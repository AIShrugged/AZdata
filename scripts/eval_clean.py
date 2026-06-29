"""Leak-free Task-2 evaluation with micro + macro + per-class P/R/F1.

Runs the RAG classifier (default: openrouter qwen3.5-35b-a3b + best_instructions)
over a split and reports BOTH micro (per-item) and macro (per-class) metrics, so
the imbalanced-class tail is visible. Requires a rebuilt train index that is
disjoint from the eval split (see scripts/make_splits.py).

  OPENROUTER_API_KEY=... python scripts/eval_clean.py --split test --k 16
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import rag  # noqa: E402

P = ROOT / "data/processed"
GROUPS = ["BAKERY", "CANNED FISH", "WIPES", "MED.SYRINGES", "TOWELS", "PUBLIC UTILITIES WATER", "DENTAL MEDICINE"]


def _prf(true: list[str], pred: list[str], classes: list[str]):
    res = {}
    for c in classes:
        tp = sum(1 for t, p in zip(true, pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(true, pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(true, pred) if t == c and p != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        res[c] = (prec, rec, f1, tp + fn)
    macro = tuple(sum(res[c][i] for c in classes) / len(classes) for i in range(3))
    return res, macro


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--provider", default="openrouter")
    ap.add_argument("--model", default="qwen/qwen3.5-35b-a3b")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--instructions", default=str(P / "best_instructions.txt"))
    args = ap.parse_args()

    emb, meta = rag.load_index(P / "train_index")
    instructions = Path(args.instructions).read_text(encoding="utf-8") if Path(args.instructions).exists() else None
    rows = list(csv.DictReader(open(P / f"{args.split}.csv", encoding="utf-8")))
    print(f"{args.split}: {len(rows)} items | index {len(meta)} | {args.provider} {args.model} k={args.k}", file=sys.stderr)

    def one(r):
        out = rag.classify_rag(r["text"], emb, meta, k=args.k, provider=args.provider, model=args.model, instructions=instructions)
        return {"text": r["text"], "true_label": r["label"], "true_group": (r.get("group") or ""),
                "pred_label": (out.get("label") or ""), "pred_group": (out.get("group") or ""), "ok": out.get("ok", False)}

    recs: list = [None] * len(rows)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(one, r): i for i, r in enumerate(rows)}
        for n, f in enumerate(as_completed(futs), 1):
            recs[futs[f]] = f.result()
            if n % 100 == 0:
                print(f"  {n}/{len(rows)}", file=sys.stderr)

    out_path = P / f"eval_clean_{args.split}.json"
    json.dump(recs, open(out_path, "w"), ensure_ascii=False, indent=1)

    N = len(recs)
    tl = [r["true_label"] for r in recs]; pl = [r["pred_label"] for r in recs]
    lab_res, lab_macro = _prf(tl, pl, ["Good", "Service"])
    lab_micro = sum(1 for r in recs if r["true_label"] == r["pred_label"]) / N
    goods = [r for r in recs if r["true_label"] == "Good"]
    grp_res, grp_macro = _prf([r["true_group"] for r in goods], [r["pred_group"] for r in goods], GROUPS)
    grp_micro = sum(1 for r in goods if r["true_group"] == r["pred_group"]) / max(1, len(goods))
    fully = sum(1 for r in recs if r["true_label"] == r["pred_label"] and (r["true_label"] == "Service" or r["true_group"] == r["pred_group"])) / N

    print(f"\n=== {args.split} ({N} items, parse-fail {sum(1 for r in recs if not r['ok'])}) ===")
    print(f"Good/Service : micro {lab_micro*100:.1f}%  macro-F1 {lab_macro[2]*100:.1f}%")
    print(f"7-group      : micro {grp_micro*100:.1f}%  MACRO-F1 {grp_macro[2]*100:.1f}%")
    for c in GROUPS:
        p, r, f, n = grp_res[c]
        print(f"   {c:24} P={p*100:5.1f} R={r*100:5.1f} F1={f*100:5.1f} (n={n})")
    print(f"Fully correct: {fully*100:.1f}%   → {out_path.name}")


if __name__ == "__main__":
    main()
