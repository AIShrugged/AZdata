"""Synthetic Azerbaijani e-invoice line-item generator for scale + robustness testing.

Generates labelled items across difficulty tiers and item kinds, using a strong model:
  - easy   : canonical, unambiguous product names
  - medium : realistic noise (brands, sizes/units, abbreviations, AZ/RU mix, typos, SKU codes)
  - hard   : ambiguous / look-alikes / borderline good-vs-service
  + special kinds:
  - MIXED  : a single line bundling a GOOD with an ancillary SERVICE (e.g. "Çörək çatdırılma ilə").
             Gold follows the PRIMARY-COMPONENT rule: principal good wins; ancillary delivery follows it.
  - OOD    : goods OUTSIDE the 7 groups (concrete, electronics…). Gold group = OTHER → tests ABSTAIN.

IMPORTANT: labels are MODEL-GENERATED (a strong model), not human ground truth. Synthetic data is for
relative/scale/robustness testing and to surface failure modes — not for absolute accuracy claims.
Generated items are de-duplicated against the real train set so this never leaks into/over training.

  OPENROUTER_API_KEY=... python scripts/gen_synthetic.py --per 4 --eval
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import rag  # noqa: E402
from classify import GROUPS, GROUP_HINTS  # noqa: E402
from nlsql import call_llm  # noqa: E402

P = ROOT / "data/processed"
OUT = P / "synthetic.csv"
FIELDS = ["text", "label", "group", "kind", "difficulty", "is_mixed", "is_ood", "trap", "source"]
DIFFICULTY = {
    "easy": "canonical, unambiguous product names; the category is obvious",
    "medium": "realistic invoice noise: brand names, sizes/units, abbreviations, mixed Azerbaijani/Russian, typos, SKU codes",
    "hard": "ambiguous or tricky: look-alikes, borderline good-vs-service, unusual phrasing",
}


def _norm(t: str) -> str:
    return " ".join(str(t).split()).casefold()


def spec_prompt(kind: str, difficulty: str, n: int) -> tuple[str, str]:
    if kind in GROUPS:
        what = f'physical GOODS in the product group "{kind}" ({GROUP_HINTS[kind]})'
        gold = f'"label":"Good","group":"{kind}","is_mixed":false,"is_ood":false'
    elif kind == "SERVICE":
        what = "SERVICES (xidmət): works/activities — construction, transport, repair, installation, consulting, utility labour"
        gold = '"label":"Service","group":null,"is_mixed":false,"is_ood":false'
    elif kind == "MIXED":
        what = (
            "a SINGLE invoice line that BUNDLES a physical good from one of these groups "
            f'[{", ".join(GROUPS)}] together with an ANCILLARY service (usually delivery/çatdırılma, '
            'sometimes installation/quraşdırma), e.g. "Çörək çatdırılma ilə"'
        )
        gold = (
            '"label":"Good","group":"<the good\'s group from the 7>","is_mixed":true,"is_ood":false,'
            '"components":[{"part":"<good>","kind":"Good"},{"part":"<service>","kind":"Service"}]'
        )
    else:  # OOD
        what = (
            f"physical GOODS that do NOT fit any of these 7 groups [{', '.join(GROUPS)}] — "
            "e.g. concrete, cement, electronics, furniture, clothing, fuel, stationery"
        )
        gold = '"label":"Good","group":"OTHER","is_mixed":false,"is_ood":true'
    system = (
        "You generate REALISTIC synthetic Azerbaijani e-invoice LINE ITEMS to stress-test a "
        "Good(Mal)/Service(Xidmət) + 7-group classifier. Write authentic Azerbaijani item text as it "
        f"appears on real invoices. Difficulty = {difficulty}: {DIFFICULTY[difficulty]}. "
        f"Generate {n} DISTINCT items that are {what}. For MIXED items the gold label uses the "
        "PRIMARY-COMPONENT rule (the principal good wins; the ancillary service follows it). "
        "Output ONLY a JSON array; each element exactly: "
        '{"text":"<azerbaijani item>", ' + gold + ', "difficulty":"' + difficulty + '", '
        '"trap":"<short phrase: why this is ' + difficulty + '>"}. Vary vendor/size/wording; never repeat an item.'
    )
    return system, f"Generate {n} {difficulty} items now as a JSON array."


def parse_array(raw: str) -> list[dict]:
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.I | re.S)
    m = re.search(r"\[.*\]", raw, flags=re.S)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def generate(args) -> list[dict]:
    kinds = list(GROUPS) + ["SERVICE"]
    specs = [(k, d) for d in ("easy", "medium", "hard") for k in kinds]
    specs += [("MIXED", "hard"), ("OOD", "hard")] * args.special  # extra hard special batches
    train_norm = {_norm(r["text"]) for r in csv.DictReader(open(P / "train.csv", encoding="utf-8"))}

    rows: list[dict] = []
    seen: set[str] = set()

    def run(spec):
        kind, diff = spec
        system, user = spec_prompt(kind, diff, args.per)
        try:
            return kind, diff, parse_array(call_llm(system, user, args.provider, args.model))
        except Exception as exc:
            print(f"  ! {kind}/{diff}: {exc}", file=sys.stderr)
            return kind, diff, []

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for kind, diff, items in ex.map(run, specs):
            for it in items:
                text = str(it.get("text", "")).strip()
                nt = _norm(text)
                if not text or nt in seen or nt in train_norm:  # de-dup vs itself + the real train set
                    continue
                seen.add(nt)
                rows.append({
                    "text": text,
                    "label": it.get("label") or ("Service" if kind == "SERVICE" else "Good"),
                    "group": (None if it.get("group") in (None, "null") else it.get("group")),
                    "kind": kind, "difficulty": diff,
                    "is_mixed": bool(it.get("is_mixed")), "is_ood": bool(it.get("is_ood")),
                    "trap": str(it.get("trap", ""))[:120], "source": "synthetic",
                })
    return rows


def evaluate(rows: list[dict], args) -> None:
    emb, meta = rag.load_index(P / "train_index")
    instr_path = P / "best_instructions.txt"
    instructions = instr_path.read_text(encoding="utf-8") if instr_path.exists() else None

    def classify(r):
        out = rag.classify_rag(r["text"], emb, meta, k=16, provider=args.provider,
                               model="qwen/qwen3.5-35b-a3b", instructions=instructions)
        return r, out

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for r, out in ex.map(classify, rows):
            results.append((r, out))

    def abstained(out):
        return bool(out.get("needs_review")) or out.get("group") == "OTHER"

    def correct(r, out):
        if r["is_ood"]:
            return abstained(out)  # correct = abstained instead of forcing a wrong group
        if r["label"] == "Service":
            return (out.get("label") or "") == "Service"
        return (out.get("label") or "") == "Good" and (out.get("group") or "") == r["group"]

    by_diff: dict = {}
    for r, out in results:
        if r["is_ood"]:
            continue
        d = r["difficulty"]
        by_diff.setdefault(d, [0, 0])
        by_diff[d][1] += 1
        by_diff[d][0] += 1 if correct(r, out) else 0

    print("\n=== synthetic eval (in-domain, primary-component gold) ===")
    for d in ("easy", "medium", "hard"):
        if d in by_diff:
            c, n = by_diff[d]
            print(f"  {d:6}: {100*c/n:5.1f}%  ({c}/{n})")

    mixed = [(r, out) for r, out in results if r["is_mixed"]]
    if mixed:
        mc = sum(1 for r, out in mixed if (out.get("label") or "") == "Good" and (out.get("group") or "") == r["group"])
        print(f"\n  MIXED (good+ancillary service → expect primary Good+group): {100*mc/len(mixed):.0f}% ({mc}/{len(mixed)})")
        for r, out in mixed[:4]:
            print(f"     {r['text'][:42]:42} → {out.get('label')}/{out.get('group') or '-'} mixed={out.get('is_mixed')} review={out.get('needs_review')}")

    ood = [(r, out) for r, out in results if r["is_ood"]]
    if ood:
        ab = sum(1 for r, out in ood if abstained(out))
        print(f"\n  OOD (not in 7 groups → SHOULD abstain): abstained {ab}/{len(ood)}  (needs_review or group=OTHER)")
        for r, out in ood[:4]:
            print(f"     {r['text'][:42]:42} → {out.get('label')}/{out.get('group') or '-'} review={out.get('needs_review')}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=4, help="items per (kind, difficulty)")
    ap.add_argument("--special", type=int, default=2, help="batches of MIXED + OOD")
    ap.add_argument("--provider", default="openrouter")
    ap.add_argument("--model", default="qwen/qwen3.5-122b-a10b", help="generator model (use a strong one)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--eval", action="store_true", help="also classify the generated set and report")
    ap.add_argument("--eval-only", action="store_true", help="skip generation; evaluate the existing synthetic.csv")
    args = ap.parse_args()

    if args.eval_only:
        rows = list(csv.DictReader(open(OUT, encoding="utf-8")))
        for r in rows:
            r["is_mixed"] = str(r.get("is_mixed")).strip().lower() == "true"
            r["is_ood"] = str(r.get("is_ood")).strip().lower() == "true"
        print(f"evaluating existing {len(rows)} items from {OUT.name}", file=sys.stderr)
        evaluate(rows, args)
        return

    print(f"generating with {args.provider}:{args.model} …", file=sys.stderr)
    rows = generate(args)
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    counts: dict = {}
    for r in rows:
        counts[r["difficulty"] + "/" + r["kind"]] = counts.get(r["difficulty"] + "/" + r["kind"], 0) + 1
    print(f"wrote {len(rows)} items → {OUT.name}")
    print("breakdown:", json.dumps(counts, ensure_ascii=False))
    if args.eval:
        evaluate(rows, args)


if __name__ == "__main__":
    main()
