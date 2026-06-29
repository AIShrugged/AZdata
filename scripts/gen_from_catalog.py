"""Grounded synthetic generator: expand REAL catalog lines into realistic variants.

Unlike gen_synthetic.py (which invents items from the 7 group names), this takes ACTUAL
goods/services lines from the catalog (data/processed/labeled_items.csv, derived from the
e-invoice goods/services XLS), and for each seed line asks a strong model for N realistic
VARIANTS of the SAME product/service (varied vendor/size/format/typos) — so every variant
INHERITS the seed line's gold label/group (reliable labels, not invented). Then it mixes
them and (optionally) runs them through the classifier.

  OPENROUTER_API_KEY=... python scripts/gen_from_catalog.py --seeds-per-group 3 --variants 8 --eval
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import rag  # noqa: E402
from classify import GROUPS  # noqa: E402
from nlsql import call_llm  # noqa: E402

P = ROOT / "data/processed"
CATALOG = P / "labeled_items.csv"
OUT = P / "synthetic_catalog.csv"
FIELDS = ["text", "label", "group", "seed_text", "source"]


def _norm(t: str) -> str:
    return " ".join(str(t).split()).casefold()


def variant_prompt(seed_text: str, label: str, group: str, n: int) -> tuple[str, str]:
    cat = f"{label}/{group}" if group else label
    system = (
        "You are given a REAL Azerbaijani e-invoice line item and its category. Generate "
        f"{n} DISTINCT, realistic VARIANTS of THE SAME product/service as they would appear on "
        "different real invoices — vary the vendor/brand, size/quantity/units, abbreviations, word "
        "order, Azerbaijani/Russian mix, minor typos, packaging and formatting. Keep the SAME "
        "product meaning and the SAME category; do NOT drift to a different product type. "
        "Output ONLY a JSON array of strings, nothing else."
    )
    user = f"ITEM: {seed_text}\nCATEGORY: {cat}\nGenerate {n} variants as a JSON array of strings."
    return system, user


def parse_strings(raw: str) -> list[str]:
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.I | re.S)
    m = re.search(r"\[.*\]", raw, flags=re.S)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except Exception:
        return []
    return [str(x).strip() for x in data if isinstance(data, list) and str(x).strip()]


def pick_seeds(rows: list[dict], per_group: int, rng: random.Random) -> list[dict]:
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["label"], r.get("group") or "")].append(r)
    seeds: list[dict] = []
    for key in sorted(by):
        bucket = by[key]
        rng.shuffle(bucket)
        seeds.extend(bucket[:per_group])
    return seeds


def generate(args) -> list[dict]:
    rows = list(csv.DictReader(open(CATALOG, encoding="utf-8")))
    train_norm = {_norm(r["text"]) for r in csv.DictReader(open(P / "train.csv", encoding="utf-8"))}
    seeds = pick_seeds(rows, args.seeds_per_group, random.Random(7))
    print(f"{len(seeds)} real seed lines → up to {args.variants} variants each", file=sys.stderr)

    def run(seed):
        system, user = variant_prompt(seed["text"], seed["label"], seed.get("group") or "", args.variants)
        try:
            return seed, parse_strings(call_llm(system, user, args.provider, args.model))
        except Exception as exc:
            print(f"  ! seed {seed['text'][:30]}: {exc}", file=sys.stderr)
            return seed, []

    out: list[dict] = []
    seen: set[str] = set()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for seed, variants in ex.map(run, seeds):
            for v in variants:
                nv = _norm(v)
                if not v or nv in seen or nv in train_norm or nv == _norm(seed["text"]):
                    continue
                seen.add(nv)
                out.append({"text": v, "label": seed["label"], "group": (seed.get("group") or None),
                            "seed_text": seed["text"], "source": "synthetic_catalog"})
    return out


def evaluate(rows: list[dict], args) -> None:
    emb, meta = rag.load_index(P / "train_index")
    ip = P / "best_instructions.txt"
    instr = ip.read_text(encoding="utf-8") if ip.exists() else None

    def classify(r):
        o = rag.classify_rag(r["text"], emb, meta, k=16, provider=args.provider,
                             model="qwen/qwen3.5-35b-a3b", instructions=instr)
        return r, (o.get("label") or ""), (o.get("group") or "")

    res = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for r, pl, pg in ex.map(classify, rows):
            res.append((r, pl, pg))

    def ok(r, pl, pg):
        if r["label"] == "Service":
            return pl == "Service"
        return pl == "Good" and pg == (r["group"] or "")

    n = len(res)
    label_ok = sum(1 for r, pl, pg in res if pl == r["label"])
    full_ok = sum(1 for r, pl, pg in res if ok(r, pl, pg))
    print(f"\n=== grounded synthetic eval (variants of REAL catalog lines, N={n}) ===")
    print(f"  label (Good/Service): {100*label_ok/n:.1f}%   fully (label+group): {100*full_ok/n:.1f}%")
    per = collections.defaultdict(lambda: [0, 0])
    for r, pl, pg in res:
        key = r["group"] or "SERVICE"
        per[key][1] += 1
        per[key][0] += 1 if ok(r, pl, pg) else 0
    for key in sorted(per):
        c, t = per[key]
        print(f"    {key:24} {100*c/t:5.1f}%  ({c}/{t})")
    misses = [(r, pl, pg) for r, pl, pg in res if not ok(r, pl, pg)][:6]
    if misses:
        print("  sample misses (variant → predicted / gold):")
        for r, pl, pg in misses:
            print(f"    {r['text'][:44]:44} → {pl}/{pg or '-'}  (gold {r['label']}/{r['group'] or '-'})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds-per-group", type=int, default=3)
    ap.add_argument("--variants", type=int, default=8)
    ap.add_argument("--provider", default="openrouter")
    ap.add_argument("--model", default="qwen/qwen3.5-122b-a10b", help="generator model (use a strong one)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--eval", action="store_true")
    args = ap.parse_args()

    rows = generate(args)
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} variants → {OUT.name}")
    if args.eval:
        evaluate(rows, args)


if __name__ == "__main__":
    main()
