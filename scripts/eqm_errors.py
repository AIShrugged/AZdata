"""Error analysis for the EQM engine: show items that missed the exact code, grouped by
WHY (retrieval miss / rerank miss / sub-code miss) with gold vs predicted descriptions.

  OPENROUTER_API_KEY=... python scripts/eqm_errors.py --file data/processed/test_hard.csv
"""
from __future__ import annotations

import argparse
import collections
import csv
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import eqm  # noqa: E402

P = ROOT / "data/processed"
CODE2DESC = {r["code"]: (r.get("description") or "") for r in csv.DictReader(open(P / "eqm_registry.csv", encoding="utf-8"))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(P / "test_hard.csv"))
    ap.add_argument("--model", default="qwen/qwen3.5-122b-a10b")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--per", type=int, default=6)
    a = ap.parse_args()

    emb, meta = eqm.load_eqm_index()
    rows = list(csv.DictReader(open(a.file, encoding="utf-8")))

    def run(r):
        res = eqm.assign_code(r["text"], emb, meta, provider="openrouter", model=a.model)
        pred = "".join(c for c in str(res.get("code", "")) if c.isdigit())
        cand_h = {str(c)[:4] for c in res.get("candidates", [])}
        return r, pred, cand_h

    misses = collections.defaultdict(list)
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for r, pred, cand_h in ex.map(run, rows):
            gold = r["hs_code"]
            if pred == gold:
                continue
            gh, ph = gold[:4], pred[:4]
            if gh not in cand_h:
                cat = "RETRIEVAL MISS — the gold heading was never even retrieved"
            elif ph != gh:
                cat = "RERANK MISS — gold heading WAS available, model picked a different heading"
            else:
                cat = "SUB-CODE MISS — right heading, wrong 10-digit (info not in the text / ambiguous)"
            misses[cat].append((r["text"], gold, CODE2DESC.get(gold, "")[:50], pred, CODE2DESC.get(pred, "")[:50]))

    total = sum(len(v) for v in misses.values())
    print(f"\n{total} exact-misses out of {len(rows)} items\n")
    for cat in sorted(misses):
        items = misses[cat]
        print(f"### {cat}  ({len(items)})")
        for text, gold, gdesc, pred, pdesc in items[: a.per]:
            print(f"  ITEM: {text[:54]}")
            print(f"     gold {gold}  {gdesc}")
            print(f"     pred {pred}  {pdesc}")
        print()


if __name__ == "__main__":
    main()
