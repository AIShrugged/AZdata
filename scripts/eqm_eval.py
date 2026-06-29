"""Eval the EQM category engine on a labelled file (text, hs_code, heading, chapter).
Reports recall@k (correct heading in candidate pool) / heading / chapter / exact.

  OPENROUTER_API_KEY=... python scripts/eqm_eval.py data/processed/test_hard.csv
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--provider", default="openrouter")
    ap.add_argument("--model", default="qwen/qwen3.5-122b-a10b")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    emb, meta = eqm.load_eqm_index()
    rows = list(csv.DictReader(open(a.file, encoding="utf-8")))
    if a.limit:
        rows = rows[: a.limit]

    def run(r):
        res = eqm.assign_code(r["text"], emb, meta, k=a.k, provider=a.provider, model=a.model)
        pred = "".join(c for c in str(res.get("code", "")) if c.isdigit())
        cands = {str(c)[:4] for c in res.get("candidates", [])}
        return r, pred, cands

    agg = collections.Counter()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for r, pred, cands in ex.map(run, rows):
            agg["n"] += 1
            agg["exact"] += pred == r["hs_code"]
            agg["heading"] += pred[:4] == r["heading"]
            agg["chapter"] += pred[:2] == r["chapter"]
            agg["recall"] += r["heading"] in cands
    n = agg["n"] or 1
    print(f"{Path(a.file).name}: n={agg['n']}  recall@k {100*agg['recall']/n:5.1f}%  "
          f"heading {100*agg['heading']/n:5.1f}%  chapter {100*agg['chapter']/n:5.1f}%  exact {100*agg['exact']/n:5.1f}%")


if __name__ == "__main__":
    main()
