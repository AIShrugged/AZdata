"""Rebuild a PROPER gold from each item (not the lazy synthetic seed): a strong model determines
the true HS chapter+heading from world knowledge (resolving brand/colloquial names like
'Kachka'=duck, using the unit as a sanity check), backing off to chapter when unsure.

Then it measures the engine against this proper gold (a fair ruler) and reports how often the
SYNTHETIC SEED itself was wrong — i.e. how much the seed-based metric was undercounting.

  OPENROUTER_API_KEY=... python scripts/gold_build.py --file data/processed/test_hard.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import eqm  # noqa: E402
from classify import extract_json  # noqa: E402
from nlsql import call_llm  # noqa: E402

P = ROOT / "data/processed"


def gold_label(item: str, model: str):
    system = (
        "You are an HS customs classification expert. Given an Azerbaijani invoice item (which may use a "
        "brand, model, or colloquial name — e.g. 'Kachka' = duck, 'McCain' = frozen potato/veg), determine "
        "the CORRECT HS classification from world knowledge. Identify the real product; use the UNIT as a "
        "sanity check ('q'/'kq' = weight, 'l' = litres, 'ədəd' = pieces). "
        'Output ONLY JSON: {"chapter":"NN","heading":"NNNN","confidence":<0..1>}. '
        "If unsure of the 4-digit heading, still give the 2-digit chapter with a low confidence."
    )
    try:
        j = extract_json(call_llm(system, f"ITEM: {item}", "openrouter", model))
        ch = "".join(c for c in str(j.get("chapter", "")) if c.isdigit())[:2]
        hd = "".join(c for c in str(j.get("heading", "")) if c.isdigit())[:4]
        cf = float(j.get("confidence", 0) or 0)
        return ch, hd, cf
    except Exception:
        return "", "", 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(P / "test_hard.csv"))
    ap.add_argument("--model", default="qwen/qwen3.5-122b-a10b")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--min-conf", type=float, default=0.6)
    a = ap.parse_args()

    emb, meta = eqm.load_eqm_index()
    rows = list(csv.DictReader(open(a.file, encoding="utf-8")))

    def run(r):
        gch, ghd, gcf = gold_label(r["text"], a.model)
        res = eqm.assign_code(r["text"], emb, meta, provider="openrouter", model=a.model)
        pred = "".join(c for c in str(res.get("code", "")) if c.isdigit())
        return r, gch, ghd, gcf, pred, bool(res.get("needs_review"))

    recs = []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for out in ex.map(run, rows):
            recs.append(out)

    confident = [x for x in recs if x[3] >= a.min_conf and x[2]]  # gold heading present + confident
    n = len(confident) or 1
    seed_ok = sum(1 for r, gch, ghd, gcf, pred, nr in confident if r["heading"] == ghd)
    eng_h = sum(1 for r, gch, ghd, gcf, pred, nr in confident if pred[:4] == ghd)
    eng_c = sum(1 for r, gch, ghd, gcf, pred, nr in confident if pred[:2] == gch)
    seed_h = sum(1 for r, gch, ghd, gcf, pred, nr in confident if pred[:4] == r["heading"])
    review = sum(1 for x in recs if x[5])

    print(f"\n=== proper-gold measurement (n={len(confident)} items where the rebuilt gold is confident) ===")
    print(f"  synthetic SEED was correct (seed heading == proper gold): {100*seed_ok/n:.0f}%   <- how good the lazy gold was")
    print(f"  engine heading  vs SEED  gold : {100*seed_h/n:.0f}%   (the old, unfair number)")
    print(f"  engine heading  vs PROPER gold: {100*eng_h/n:.0f}%   <- the FAIR number")
    print(f"  engine chapter  vs PROPER gold: {100*eng_c/n:.0f}%")
    print(f"  engine flagged needs_review (honest abstain): {100*review/len(recs):.0f}% of all {len(recs)} items")
    print("\n  examples where the SEED was wrong but the engine matched the proper gold:")
    shown = 0
    for r, gch, ghd, gcf, pred, nr in confident:
        if r["heading"] != ghd and pred[:4] == ghd and shown < 6:
            print(f"    {r['text'][:42]:42} seed {r['heading']}  proper {ghd}  engine {pred[:4]}")
            shown += 1


if __name__ == "__main__":
    main()
