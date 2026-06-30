from __future__ import annotations

import sys; from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rag import embed_texts, _l2norm
from nlsql import call_llm, DEFAULT_MODELS, PROVIDER as DEFAULT_PROVIDER
from classify import _normalize_confidence, extract_json

import numpy as np, json, csv, os, re, argparse

ROOT = Path(__file__).resolve().parents[1]
EQM_CSV = ROOT / "data/processed/eqm_registry.csv"
INDEX_PREFIX = ROOT / "data/processed/eqm_index"


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8", newline="") as fh: return list(csv.DictReader(fh))


def _clean(value: Any) -> str:
    return "" if value is None else str(value)


def _trunc(text: str, n: int) -> str:
    return text if len(text) <= n else text[: n - 3] + "..."


def build_eqm_index() -> None:
    rows = _read_csv(EQM_CSV)
    kw_path = ROOT / "data/processed/eqm_keywords.json"
    keywords = json.load(open(kw_path, encoding="utf-8")) if kw_path.exists() else {}
    meta: list[dict[str, Any]] = []
    texts: list[str] = []  # text to EMBED = description + enrichment keywords (brands/synonyms)
    for row in rows:
        if str(row.get("active", "")).strip().lower() not in {"true", "1", "t", "yes"}:
            continue
        code = _clean(row.get("code"))
        desc = _clean(row.get("description"))
        meta.append({"code": code, "description": desc, "unit": _clean(row.get("unit"))})
        kw = str(keywords.get(code, "")).strip()
        texts.append(f"{desc} | {kw}" if kw else desc)

    emb = embed_texts(texts, batch=64).astype("float32")
    np.save(str(INDEX_PREFIX) + ".npy", emb)
    with open(str(INDEX_PREFIX) + ".meta.json", "w", encoding="utf-8") as fh: json.dump(meta, fh, ensure_ascii=False, indent=2)
    dim = int(emb.shape[1]) if emb.ndim == 2 and emb.shape[0] else 0
    print(f"indexed {len(meta)} active codes ({len(keywords)} enriched), dim {dim}")


def load_eqm_index() -> tuple[np.ndarray, list[dict[str, Any]]]:
    emb = np.load(str(INDEX_PREFIX) + ".npy").astype("float32")
    with open(str(INDEX_PREFIX) + ".meta.json", "r", encoding="utf-8") as fh: meta = json.load(fh)
    return _l2norm(emb), list(meta)


def _clean_query(text: str) -> str:
    t = re.sub(r"\([^)]*\)", " ", text)  # drop (brand)/(type) parentheticals
    t = " ".join(w for w in t.split() if not any(ch.isdigit() for ch in w))  # drop size/number tokens
    return re.sub(r"\s+", " ", t).strip() or text


def retrieve_codes(item_text: str, emb: np.ndarray, meta: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    qe = _l2norm(embed_texts([_clean_query(item_text)]))[0]
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        sims = np.nan_to_num(emb @ qe, nan=-1.0)
    kk = min(k, sims.shape[0])
    if kk <= 0:
        return []
    top = np.argpartition(-sims, kk - 1)[:kk]
    top = top[np.argsort(-sims[top])]
    return [meta[int(i)] for i in top]


def predict_headings(item_text: str, group: Optional[str], provider: str, model: Optional[str]) -> list[str]:
    system = (
        "You are an HS (Harmonized System) customs classification expert. Given an Azerbaijani "
        "product name and its category, output the 4-6 most likely 4-digit HS headings (chapter+heading), best guess first."
        'Output ONLY JSON: {"headings": ["NNNN", ...]}. Use your HS knowledge.'
    )
    user = f"PRODUCT: {item_text}\nCATEGORY: {group or ''}"
    try:
        parsed = extract_json(call_llm(system, user, provider, model))
        out: list[str] = []
        for h in parsed.get("headings", []):
            digits = "".join(ch for ch in str(h) if ch.isdigit())[:4]
            if len(digits) == 4:
                out.append(digits)
        return out
    except Exception:
        return []


def expand_query(item_text: str, provider: Optional[str], model: Optional[str]) -> str:
    """Describe a (possibly brand/colloquial) product in generic category terms so it embeds near
    the formal HS descriptions — bridges the item-name <-> description vocabulary gap."""
    system = (
        "You are an HS customs classification assistant. Given an Azerbaijani invoice product name "
        "(which may be a brand, model, or colloquial name), describe in 1-2 lines WHAT THE PRODUCT IS "
        "in generic category terms useful for customs classification: material, type, form, use, and "
        "common synonyms (Azerbaijani/English/Russian). Do NOT output any code or number. Plain text only."
    )
    try:
        return call_llm(system, f"PRODUCT: {item_text}", provider, model).strip()[:300]
    except Exception:
        return ""


_MACHINE_CHAPTERS = {"84", "85", "86", "87", "88", "89"}  # machinery/electrical/vehicles/ships/aircraft — never sold by the gram (ch.90 incl. medical syringes, so excluded)
_WEIGHT_VOL_UNITS = {"q", "qr", "qram", "kq", "kg", "g", "gr", "l", "litr", "ml"}


def _sanity_mismatch(item_text: str, code: str) -> bool:
    """A weight/volume-measured item (grams/kg/litres) that landed in a machinery/equipment chapter is
    almost certainly wrong — used to trigger Tier-2 / flag review even when the model was overconfident."""
    tokens = re.findall(r"[a-zəçşğıöü]+", item_text.lower())
    weighed = any(u in tokens for u in _WEIGHT_VOL_UNITS)
    return weighed and str(code)[:2] in _MACHINE_CHAPTERS


def assign_code(
    item_text: str,
    emb: np.ndarray,
    meta: list[dict[str, Any]],
    k: int = 60,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    group: Optional[str] = None,
    tier2: bool = True,
    web: Optional[bool] = None,
) -> dict[str, Any]:
    provider = provider or DEFAULT_PROVIDER
    model = model or DEFAULT_MODELS.get(provider)
    cands: list[dict[str, Any]] = []
    try:
        # Query expansion: describe the (possibly brand/colloquial) item in generic category terms
        # so headings + embedding retrieval find it even with no shared vocabulary.
        expanded = expand_query(item_text, provider, model)
        queries = [item_text, expanded] if expanded else [item_text]
        headings = set(predict_headings(expanded or item_text, group, provider, model))
        if headings:
            cands = [c for c in meta if str(c.get("code", ""))[:4] in headings][:50]
        # ALWAYS union with embedding retrieval over BOTH the raw item and its expansion, so a
        # single wrong predicted heading cannot drop the true code from the candidate pool.
        seen = {str(c.get("code", "")) for c in cands}
        for q in queries:
            for c in retrieve_codes(q, emb, meta, k):
                code = str(c.get("code", ""))
                if code not in seen:
                    cands.append(c)
                    seen.add(code)
        if not cands:
            raise ValueError("no candidates retrieved")
        what = f"\nWHAT IT IS: {expanded}" if expanded else ""
        codes = {str(c.get("code", "")): c for c in cands}
        system = (
            "You are an HS customs expert. First decide what the product fundamentally IS "
            "(material / type / form / use). Use COMMON SENSE and the item's UNIT as a sanity check: "
            "'q'/'kq' = grams/kilograms and 'l' = litres mean a physical good measured by weight/volume "
            "(food, material) — NOT machinery, pumps, or equipment; 'ədəd' = pieces. Then choose from the "
            "candidate list the single code whose description best matches that category — by meaning, not "
            "surface words. If NONE of the candidates plausibly fit, pick the closest BROADER match and set "
            "confidence below 0.35. Be honest — give a LOW confidence when the candidates are a poor fit. "
            'Output ONLY one JSON object: {"code": "<one of the candidate codes>", "confidence": <0..1>}. '
            "The code MUST be exactly one of the candidates."
        )
        user = "PRODUCT: " + item_text + what + "\n\nCANDIDATES:\n" + "\n".join(
            f'{c["code"]}: {c["description"]}' for c in cands
        )
        raw = call_llm(system, user, provider, model)
        parsed = extract_json(raw)
        parsed_code = str(parsed.get("code", ""))
        valid = parsed_code in codes
        chosen = parsed_code if valid else str(cands[0].get("code", ""))
        chosen_row = codes.get(chosen, {})
        # Confidence must describe the code actually returned: parse defensively (a non-numeric
        # value like "high" no longer aborts the whole result) and zero it when we fell back to cands[0].
        confidence = _normalize_confidence(parsed.get("confidence")) if valid else 0.0
        # TIER 2 (auto-resolve): if uncertain, resolve the brand/colloquial name to a generic product
        # (LLM knowledge; optional web lookup behind the privacy toggle) and re-retrieve + re-rank ONCE.
        # A successful resolution is LEARNED so the same pattern auto-resolves in Tier-1 next time.
        if tier2 and (confidence < 0.5 or _sanity_mismatch(item_text, chosen)):
            import resolver
            r = resolver.resolve(item_text, provider, model, web=web)
            product = r.get("product")
            if product:
                for c in retrieve_codes(product, emb, meta, k):
                    cc = str(c.get("code", ""))
                    if cc not in codes:
                        cands.append(c)
                        codes[cc] = c
                user2 = ("PRODUCT: " + item_text + f" (resolved: {product})" + what
                         + "\n\nCANDIDATES:\n" + "\n".join(f'{c["code"]}: {c["description"]}' for c in cands))
                try:
                    parsed2 = extract_json(call_llm(system, user2, provider, model))
                    pc = str(parsed2.get("code", ""))
                    conf2 = _normalize_confidence(parsed2.get("confidence")) if pc in codes else 0.0
                    if pc in codes and conf2 >= confidence:
                        chosen, confidence, chosen_row = pc, conf2, codes.get(pc, {})
                        resolver.learn(item_text, r.get("keywords") or r.get("product", ""))
                except Exception:
                    pass
        # Honest back-off: low confidence OR a unit/chapter sanity violation → flag for human review
        # + report the broader level we DO trust instead of a confident-but-wrong 10-digit guess.
        needs_review = confidence < 0.5 or _sanity_mismatch(item_text, chosen)
        granularity = "code" if confidence >= 0.5 else ("heading" if confidence >= 0.3 else "chapter")
        return {"code": chosen, "description": _clean(chosen_row.get("description")),
                "confidence": confidence, "code_substituted": not valid,
                "needs_review": needs_review, "granularity": granularity,
                "heading": str(chosen)[:4], "chapter": str(chosen)[:2],
                "candidates": [str(c.get("code", "")) for c in cands], "ok": True}
    except Exception as exc:
        top = cands[0] if cands else {}
        return {"code": _clean(top.get("code")), "description": _clean(top.get("description")),
                "confidence": 0.0, "candidates": [str(c.get("code", "")) for c in cands],
                "ok": False, "error": str(exc)}


def _row_text(row: dict[str, Any]) -> str:
    for key in ("text", "item_text", "description", "name"):
        value = row.get(key)
        if value: return str(value)
    return ""


def _selftest(k: int) -> None:
    emb, meta = load_eqm_index()
    rows = [r for r in _read_csv(ROOT / "data/processed/test.csv") if r.get("label") == "Good"][:6]
    for row in rows:
        text = _row_text(row)
        result = assign_code(text, emb, meta, k=k, group=row.get("group"))
        print(_trunc(text, 50), result.get("code"), _trunc(str(result.get("description", "")), 60), result.get("confidence"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for flag in ("--build", "--selftest"):
        parser.add_argument(flag, action="store_true")
    parser.add_argument("--k", type=int, default=12)
    args = parser.parse_args()

    if args.build:
        build_eqm_index()
    if args.selftest:
        _selftest(args.k)
