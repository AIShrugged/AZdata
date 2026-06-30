"""Fold the learning store into the index enrichment, closing the automatic cycle.

Each TRUSTED Tier-2 / web resolution is recorded in learned_synonyms.json as {term: {keywords, code}}.
This merges those keywords into that code's entry in eqm_keywords.json, so after a re-embed the term
(and all its variants) retrieves the right code in Tier-1 — a one-time resolution becomes permanent.

Run periodically, or when the store grows past a threshold:
  python scripts/apply_learned.py --rebuild        # fold + re-embed now
  python scripts/apply_learned.py --min 25         # fold only if >= 25 unfolded entries
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
LEARNED = P / "learned_synonyms.json"
KEYWORDS = P / "eqm_keywords.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="re-embed the index after folding (python src/eqm.py --build)")
    ap.add_argument("--min", type=int, default=1, help="only fold if at least this many new entries")
    a = ap.parse_args()

    learned = json.load(open(LEARNED, encoding="utf-8")) if LEARNED.exists() else {}
    kw = json.load(open(KEYWORDS, encoding="utf-8")) if KEYWORDS.exists() else {}

    pending = []
    for term, info in learned.items():
        if not isinstance(info, dict):
            continue
        code, keys = str(info.get("code", "")), str(info.get("keywords", "")).strip()
        if code and keys and keys.lower() not in (kw.get(code, "") or "").lower():
            pending.append((code, keys))

    if len(pending) < a.min:
        print(f"{len(pending)} unfolded learned entries (< --min {a.min}); nothing to do")
        return

    for code, keys in pending:
        cur = kw.get(code, "")
        kw[code] = (cur + " | " + keys).strip(" |") if cur else keys
    json.dump(kw, open(KEYWORDS, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"folded {len(pending)} learned synonym(s) into {KEYWORDS.name} ({len(learned)} in store)")

    if a.rebuild:
        print("re-embedding the index …")
        subprocess.run([sys.executable, str(ROOT / "src/eqm.py"), "--build"], check=False)
    else:
        print("run `python src/eqm.py --build` (or pass --rebuild) to apply.")


if __name__ == "__main__":
    main()
