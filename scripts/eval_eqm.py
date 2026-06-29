"""Measure EQM HS-code assignment against a gold set (top-1 + heading + candidate recall@k).

The audit noted HS-code assignment had NO ground truth and NO evaluation. This harness adds
the measurement. `data/processed/eqm_gold.csv` (columns: text, hs_code[, level]) is a STARTER
gold set seeded with items whose HS *heading* (4-digit) is well-established (medical instruments
9018, bakers' wares 1905, prepared fish 1604, oral hygiene 3306). For authoritative 10-digit
top-1 accuracy, expand it with a domain-expert-adjudicated set.

  OPENROUTER_API_KEY=... python scripts/eval_eqm.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import eqm  # noqa: E402

GOLD = ROOT / "data/processed/eqm_gold.csv"


def _digits(value: object) -> str:
    return "".join(ch for ch in str(value) if ch.isdigit())


def main() -> None:
    emb, meta = eqm.load_eqm_index()
    rows = list(csv.DictReader(open(GOLD, encoding="utf-8")))
    n = len(rows)
    exact = heading = recall = 0
    for r in rows:
        gold = _digits(r["hs_code"])
        res = eqm.assign_code(r["text"], emb, meta)
        pred = _digits(res.get("code", ""))
        cand_headings = {str(c)[:4] for c in res.get("candidates", [])}
        ex = bool(pred) and pred == gold
        hd = bool(pred) and pred[:4] == gold[:4]
        rc = gold[:4] in cand_headings
        exact += ex
        heading += hd
        recall += rc
        print(f"  {r['text'][:34]:34} gold {gold:10} pred {pred:10} heading {'Y' if hd else 'n'}  recall {'Y' if rc else 'n'}")
    print(f"\nN={n}  top-1 exact {100*exact/n:.0f}%  top-1 heading {100*heading/n:.0f}%  candidate-recall@k {100*recall/n:.0f}%")
    print("NOTE: starter gold set (heading-level) — expand with a domain-expert-labeled set for authoritative 10-digit accuracy.")


if __name__ == "__main__":
    main()
