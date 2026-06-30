from __future__ import annotations
import sys, argparse, csv, json
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rag, eqm

ROOT = Path(__file__).resolve().parents[1]
PROVIDER = "openrouter"
LOCAL_MODEL = "qwen/qwen3.5-35b-a3b"
STRONG_MODEL = "qwen/qwen3.5-122b-a10b"
K = 16
THRESHOLD = 0.9
REVIEW_THRESHOLD = 0.6  # final confidence below this (or group OTHER / model-flagged) → human review


def load_instructions() -> Optional[str]:
    path = ROOT / "data/processed/best_instructions.txt"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return None

def classify_route(
    item_text: str,
    rag_emb: Any,
    rag_meta: list[dict[str, Any]],
    instructions: Optional[str],
    k: int = K,
    provider: str = PROVIDER,
    local_model: str = LOCAL_MODEL,
    strong_model: str = STRONG_MODEL,
    threshold: float = THRESHOLD,
) -> dict[str, Any]:
    local = rag.classify_rag(
        item_text,
        rag_emb,
        rag_meta,
        k=k,
        provider=provider,
        model=local_model,
        instructions=instructions,
    )
    if local.get("ok") and float(local.get("confidence") or 0.0) >= threshold:
        chosen, tier, escalated = local, "local", False
    else:
        strong = rag.classify_rag(
            item_text,
            rag_emb,
            rag_meta,
            k=k,
            provider=provider,
            model=strong_model,
            instructions=instructions,
        )
        if strong.get("ok"):
            chosen, tier, escalated = strong, "strong", True
        else:
            chosen, tier, escalated = local, "local", True
    conf = float(chosen.get("confidence") or 0.0)
    needs_review = (
        bool(chosen.get("needs_review"))                          # model flagged it
        or chosen.get("group") == "OTHER"                         # out-of-taxonomy good
        or (bool(chosen.get("ok")) and conf < REVIEW_THRESHOLD)   # low confidence after routing
        or not chosen.get("ok")                                   # parse / total failure
    )
    result = {
        "label": chosen.get("label"),
        "group": chosen.get("group"),
        "confidence": chosen.get("confidence"),
        "is_mixed": bool(chosen.get("is_mixed")),
        "needs_review": needs_review,
        "components": chosen.get("components") or [],
        "tier": tier,
        "escalated": escalated,
        "local_confidence": local.get("confidence"),
        "ok": bool(chosen.get("ok")),
    }
    if not chosen.get("ok"):
        result["error"] = chosen.get("error") or "classification failed"
    return result

def classify_item(
    item_text: str,
    rag_emb: Any,
    rag_meta: list[dict[str, Any]],
    eqm_emb: Any,
    eqm_meta: list[dict[str, Any]],
    instructions: Optional[str],
    assign_hs: bool = True,
    web: Optional[bool] = None,
    **route_kwargs: Any,
) -> dict[str, Any]:
    c = classify_route(item_text, rag_emb, rag_meta, instructions, **route_kwargs)
    hs_code = hs_desc = None
    hs_review = False
    # Skip HS for review-bound items (out-of-taxonomy / low confidence): the code would be
    # unreliable and a human decides the item anyway.
    if c["label"] == "Good" and assign_hs and not c.get("needs_review"):
        hs = eqm.assign_code(
            item_text,
            eqm_emb,
            eqm_meta,
            group=c.get("group"),
            provider=route_kwargs.get("provider", PROVIDER),
            model=route_kwargs.get("strong_model", STRONG_MODEL),
            web=web,  # privacy toggle for Tier-2 web lookup
        )
        hs_code, hs_desc = hs.get("code"), hs.get("description")
        hs_review = bool(hs.get("needs_review"))  # low-confidence HS code → flag the item for review
    out = {
        "item": item_text,
        "label": c["label"],
        "group": c["group"],
        "hs_code": hs_code,
        "hs_description": hs_desc,
        "is_mixed": c.get("is_mixed", False),
        "needs_review": c.get("needs_review", False) or hs_review,
        "components": c.get("components", []),
        "tier": c["tier"],
        "escalated": c["escalated"],
        "confidence": c["confidence"],
        "ok": c.get("ok", True),
    }
    if c.get("error"):
        out["error"] = c["error"]
    return out

def load_all() -> tuple[Any, list[dict[str, Any]], Any, list[dict[str, Any]]]:
    rag_emb, rag_meta = rag.load_index(ROOT / "data/processed/train_index")
    eqm_emb, eqm_meta = eqm.load_eqm_index()
    return rag_emb, rag_meta, eqm_emb, eqm_meta

def _row_text(row: dict[str, str]) -> str:
    for key in ("text", "item_text", "description", "name", "item"):
        value = row.get(key)
        if value:
            return value
    return ""

def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))

def _demo_rows(rows: list[dict[str, str]], n: int) -> list[dict[str, str]]:
    n = max(n, 2)
    selected = rows[:n]
    service_rows = [r for r in rows if r.get("label") == "Service"]
    have_services = sum(1 for row in selected if row.get("label") == "Service")
    if service_rows and have_services < min(2, len(service_rows)):
        fill = [row for row in service_rows if row not in selected]
        for row in fill[: min(2, len(service_rows)) - have_services]:
            if len(selected) < n:
                selected.append(row)
            else:
                selected[-1 - have_services] = row
            have_services += 1
    return selected

def _trunc(text: str, n: int = 45) -> str:
    return text if len(text) <= n else text[: n - 3] + "..."

def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    args = parser.parse_args()
    if not args.demo:
        return

    rag_emb, rag_meta, eqm_emb, eqm_meta = load_all()
    instructions = load_instructions()
    rows = _demo_rows(_read_csv(ROOT / "data/processed/test.csv"), args.n)
    escalated = 0
    tiers: dict[str, int] = {}
    for row in rows:
        item = _row_text(row)
        result = classify_item(
            item,
            rag_emb,
            rag_meta,
            eqm_emb,
            eqm_meta,
            instructions,
            threshold=args.threshold,
        )
        tier = str(result.get("tier"))
        tiers[tier] = tiers.get(tier, 0) + 1
        escalated += 1 if result.get("escalated") else 0
        label = str(result.get("label"))
        group = result.get("group") or "-"
        hs_code = result.get("hs_code") or "-"
        print(f"{_trunc(item):45} | {label}/{group} | {tier} | {hs_code}")
    print(json.dumps({"escalated": escalated, "tiers": tiers}, ensure_ascii=False))

if __name__ == "__main__":
    _main()
