"""Full-taxonomy synthetic generator: seed from the ENTIRE EQM registry (ALL categories),
generate realistic invoice items per category at two quality tiers, gold = the seed HS code.

Unlike the 7-group generators, this covers the whole goods taxonomy (11,641 codes / 97 HS
chapters), stratified across chapters. Two quality tiers:
  - clean : realistic invoice product names (~ like the real test data)
  - noisy : degraded — abbreviations, truncations, typos, missing diacritics/units, OCR-like
            errors — to challenge the engine.
Each item's gold is the seed code (+ 4-digit heading, 2-digit chapter). Eval (a subset) runs
through the EQM category engine (eqm.assign_code) and reports accuracy across ALL categories,
broken down by quality tier.

Labels are model-generated (a strong model); fidelity is lower for obscure industrial
categories than for common goods — this tests coverage + robustness, not perfect ground truth.

  OPENROUTER_API_KEY=... python scripts/gen_taxonomy.py --seeds 250 --variants 20 --eval --eval-n 250
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
import eqm  # noqa: E402
from nlsql import call_llm  # noqa: E402

P = ROOT / "data/processed"
REGISTRY = P / "eqm_registry.csv"
OUT = P / "taxonomy_synth.csv"
FIELDS = ["text", "hs_code", "heading", "chapter", "quality", "description", "source"]


def _norm(t: str) -> str:
    return " ".join(str(t).split()).casefold()


def _active(rows: list[dict]) -> list[dict]:
    return [r for r in rows if str(r.get("active", "")).strip().lower() in {"true", "1", "t", "yes"} and r.get("code")]


def stratified_seeds(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    by_ch = collections.defaultdict(list)
    for r in rows:
        by_ch[str(r["code"])[:2]].append(r)
    chapters = sorted(by_ch)
    for ch in chapters:
        rng.shuffle(by_ch[ch])
    seeds, i = [], 0
    while len(seeds) < n:  # round-robin across chapters → broad coverage of all categories
        ch = chapters[i % len(chapters)]
        if by_ch[ch]:
            seeds.append(by_ch[ch].pop())
        i += 1
        if i > n * 20:
            break
    return seeds


def prompt(seed: dict, quality: str, n: int) -> tuple[str, str]:
    desc = (seed.get("description") or "").strip()
    if quality == "clean":
        how = ("realistic Azerbaijani e-invoice line items a business would actually enter for goods in "
               "THIS category — real product names, brands, sizes/units, as on a real invoice (NOT the "
               "official wording)")
    elif quality == "noisy":
        how = ("DEGRADED, messy versions as they appear in low-quality real invoices: heavy abbreviations, "
               "truncations, typos, missing Azerbaijani diacritics (ə→e, ş→s, ç→c), missing units, random "
               "ALL-CAPS or no-caps, OCR-like errors (O↔0, l↔1), partial/truncated names — still the "
               "SAME category. Do NOT include any HS code, catalogue number, or long digit string.")
    else:  # hard — SEMANTIC mismatch, stresses retrieval
        how = ("realistic items that DELIBERATELY do NOT share vocabulary with the official description — "
               "use BRAND names, trade names, model numbers, and colloquial product names a business "
               "actually types (e.g. 'Coca-Cola 0.5L' for carbonated drinks, 'MacBook Air M2' for laptops, "
               "'Holcim M400' for cement), so the wording does NOT lexically match the formal category. "
               "They must STILL genuinely belong to THIS category. No HS codes or long digit strings.")
    system = (
        f"You are given an Azerbaijani customs HS category (code + official description). Write {n} {how}. "
        "Output ONLY a JSON array of strings, nothing else."
    )
    return system, f"HS CODE: {seed['code']}\nOFFICIAL DESCRIPTION: {desc}\nGenerate {n} items as a JSON array."


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


def generate(args) -> list[dict]:
    rows = _active(list(csv.DictReader(open(REGISTRY, encoding="utf-8"))))
    seeds = stratified_seeds(rows, args.seeds, random.Random(7))
    qualities = {"both": ["clean", "noisy"], "all": ["clean", "noisy", "hard"]}.get(args.quality, [args.quality])
    per_q = max(1, args.variants // len(qualities))
    chapters = len({str(s["code"])[:2] for s in seeds})
    print(f"{len(seeds)} seed categories across {chapters} HS chapters × {qualities} × {per_q} → ~{len(seeds)*per_q*len(qualities)}", file=sys.stderr)

    tasks = [(s, q) for s in seeds for q in qualities]

    def run(task):
        seed, q = task
        system, user = prompt(seed, q, per_q)
        try:
            return seed, q, parse_strings(call_llm(system, user, args.provider, args.model))
        except Exception:
            return seed, q, []

    out, seen = [], set()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for seed, q, items in ex.map(run, tasks):
            code = str(seed["code"])
            for v in items:
                nv = _norm(v)
                if not v or nv in seen or re.search(r"\d{6,}", v):  # drop code-leaked items (HS codes are 6-10 digits)
                    continue
                seen.add(nv)
                out.append({"text": v, "hs_code": code, "heading": code[:4], "chapter": code[:2],
                            "quality": q, "description": (seed.get("description") or "")[:80], "source": "taxonomy_synth"})
    return out


def evaluate(rows: list[dict], args) -> None:
    emb, meta = eqm.load_eqm_index()
    rng = random.Random(13)
    sample = rows[:]
    rng.shuffle(sample)
    sample = sample[: args.eval_n]
    print(f"\nevaluating {len(sample)} items through the EQM category engine (assign_code) …", file=sys.stderr)

    def run(r):
        res = eqm.assign_code(r["text"], emb, meta, provider=args.provider, model=args.model)
        pred = "".join(c for c in str(res.get("code", "")) if c.isdigit())
        cands = {str(c)[:4] for c in res.get("candidates", [])}
        return r, pred, cands

    agg = collections.defaultdict(lambda: collections.Counter())
    ch_agg = collections.defaultdict(lambda: [0, 0])  # chapter → [heading-correct, total]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for r, pred, cands in ex.map(run, sample):
            q = r["quality"]
            agg[q]["n"] += 1
            agg[q]["exact"] += pred == r["hs_code"]
            agg[q]["heading"] += pred[:4] == r["heading"]
            agg[q]["chapter"] += pred[:2] == r["chapter"]
            agg[q]["recall"] += r["heading"] in cands
            ch_agg[r["chapter"]][1] += 1
            ch_agg[r["chapter"]][0] += pred[:4] == r["heading"]

    print("\n=== full-taxonomy eval (EQM category engine, across ALL categories) ===")
    print(f"  {'quality':8} {'exact':>7} {'heading':>8} {'chapter':>8} {'recall@k':>9}  (n)")
    for q in sorted(agg):
        a = agg[q]; n = a["n"] or 1
        print(f"  {q:8} {100*a['exact']/n:6.1f}% {100*a['heading']/n:7.1f}% {100*a['chapter']/n:7.1f}% {100*a['recall']/n:8.1f}%  ({a['n']})")
    worst = sorted(((c, v[0] / v[1], v[1]) for c, v in ch_agg.items() if v[1] >= 3), key=lambda x: x[1])[:6]
    if worst:
        print("  weakest HS chapters (heading acc, n>=3): " + ", ".join(f"ch{c} {100*a:.0f}%(n{n})" for c, a, n in worst))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=250, help="seed categories (stratified across chapters)")
    ap.add_argument("--variants", type=int, default=20, help="items per seed (split across quality tiers)")
    ap.add_argument("--quality", choices=["clean", "noisy", "hard", "both", "all"], default="both")
    ap.add_argument("--provider", default="openrouter")
    ap.add_argument("--model", default="qwen/qwen3.5-122b-a10b")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--eval-only", action="store_true", help="skip generation; evaluate the existing taxonomy_synth.csv")
    ap.add_argument("--eval-n", type=int, default=250)
    args = ap.parse_args()

    if args.eval_only:
        rows = list(csv.DictReader(open(OUT, encoding="utf-8")))
        print(f"loaded {len(rows)} items from {OUT.name}", file=sys.stderr)
        evaluate(rows, args)
        return

    rows = generate(args)
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    byq = collections.Counter(r["quality"] for r in rows)
    print(f"wrote {len(rows)} items → {OUT.name}  (by quality: {dict(byq)})")
    if args.eval:
        evaluate(rows, args)


if __name__ == "__main__":
    main()
