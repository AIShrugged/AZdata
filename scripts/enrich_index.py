"""Enrich the EQM index: for each HS code, generate common product names / BRANDS / colloquial
synonyms (Azerbaijani/English/Russian) with a strong model, so the index embedding covers the
vocabulary real invoice items use (brand/colloquial names) — raising retrieval recall.

Writes data/processed/eqm_keywords.json; then `python src/eqm.py --build` re-embeds
description+keywords to apply it.

  OPENROUTER_API_KEY=... python scripts/enrich_index.py --batch 50
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from classify import extract_json  # noqa: E402
from nlsql import call_llm  # noqa: E402

P = ROOT / "data/processed"
REGISTRY = P / "eqm_registry.csv"
OUT = P / "eqm_keywords.json"


def batch_prompt(batch):
    lines = "\n".join(f"{c}: {d}" for c, d in batch)
    system = (
        "For each HS customs code below (given as 'code: official description'), list 4-8 COMMON product "
        "names, BRANDS, and colloquial / trade synonyms (Azerbaijani / English / Russian) that would appear "
        "on real invoices for goods in that category. Be concrete (real product and brand names). "
        'Output ONLY a JSON object mapping EVERY code to a comma-separated keyword string: '
        '{"<code>": "kw, kw, ...", ...}.'
    )
    return system, lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--model", default="qwen/qwen3.5-122b-a10b")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    rows = [r for r in csv.DictReader(open(REGISTRY, encoding="utf-8"))
            if str(r.get("active", "")).strip().lower() in {"true", "1", "t", "yes"} and r.get("code")]
    if a.limit:
        rows = rows[: a.limit]
    pairs = [(r["code"], (r.get("description") or "")[:120]) for r in rows]
    valid = {c for c, _ in pairs}
    batches = [pairs[i: i + a.batch] for i in range(0, len(pairs), a.batch)]
    print(f"enriching {len(pairs)} codes in {len(batches)} batches …", file=sys.stderr)

    def run(b):
        system, user = batch_prompt(b)
        try:
            j = extract_json(call_llm(system, user, "openrouter", a.model))
            return {str(k): str(v) for k, v in j.items()} if isinstance(j, dict) else {}
        except Exception:
            return {}

    result: dict[str, str] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for d in ex.map(run, batches):
            result.update({k: v for k, v in d.items() if k in valid})
            done += 1
            if done % 20 == 0:
                print(f"  {done}/{len(batches)} batches", file=sys.stderr)

    json.dump(result, open(OUT, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"enriched {len(result)}/{len(pairs)} codes -> {OUT.name}")


if __name__ == "__main__":
    main()
