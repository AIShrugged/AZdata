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
    meta: list[dict[str, Any]] = []
    descriptions: list[str] = []
    for row in rows:
        if str(row.get("active", "")).strip().lower() not in {"true", "1", "t", "yes"}:
            continue
        item = {"code": _clean(row.get("code")), "description": _clean(row.get("description")), "unit": _clean(row.get("unit"))}
        meta.append(item)
        descriptions.append(item["description"])

    emb = embed_texts(descriptions, batch=64).astype("float32")
    np.save(str(INDEX_PREFIX) + ".npy", emb)
    with open(str(INDEX_PREFIX) + ".meta.json", "w", encoding="utf-8") as fh: json.dump(meta, fh, ensure_ascii=False, indent=2)
    dim = int(emb.shape[1]) if emb.ndim == 2 and emb.shape[0] else 0
    print(f"indexed {len(meta)} active codes, dim {dim}")


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
        "product name and its category, output the 1-3 most likely 4-digit HS headings (chapter+heading). "
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


def assign_code(
    item_text: str,
    emb: np.ndarray,
    meta: list[dict[str, Any]],
    k: int = 12,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    group: Optional[str] = None,
) -> dict[str, Any]:
    provider = provider or DEFAULT_PROVIDER
    model = model or DEFAULT_MODELS.get(provider)
    cands: list[dict[str, Any]] = []
    try:
        headings = set(predict_headings(item_text, group, provider, model))  # LLM predicts HS heading
        if headings:
            cands = [c for c in meta if str(c.get("code", ""))[:4] in headings][:40]
        # ALWAYS union with embedding retrieval so a single wrong predicted heading cannot
        # drop the true code from the candidate pool (was: embedding fallback only when empty).
        seen = {str(c.get("code", "")) for c in cands}
        for c in retrieve_codes(item_text, emb, meta, k):
            code = str(c.get("code", ""))
            if code not in seen:
                cands.append(c)
                seen.add(code)
        if not cands:
            raise ValueError("no candidates retrieved")
        codes = {str(c.get("code", "")): c for c in cands}
        system = (
            "You map an Azerbaijani product to its single best HS commodity code from the given candidate list. "
            "Choose the code whose description best matches the product. Output ONLY one JSON object: "
            '{"code": "<one of the candidate codes>", "confidence": <0..1>}. '
            "The code MUST be exactly one of the candidates."
        )
        user = "PRODUCT: " + item_text + "\n\nCANDIDATES:\n" + "\n".join(
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
        return {"code": chosen, "description": _clean(chosen_row.get("description")),
                "confidence": confidence, "code_substituted": not valid,
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
