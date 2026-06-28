from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nlsql import DEFAULT_MODELS, PROVIDER as DEFAULT_PROVIDER, call_llm

GROUPS = [
    "BAKERY",
    "CANNED FISH",
    "WIPES",
    "MED.SYRINGES",
    "TOWELS",
    "PUBLIC UTILITIES WATER",
    "DENTAL MEDICINE",
]

GROUP_HINTS: dict[str, str] = {
    "BAKERY": "bread, cakes, baklava, pastry, confectionery (çörək, tort, paxlava, şirniyyat)",
    "CANNED FISH": "canned fish and sea products (balıq konservi, dəniz məhsulları)",
    "WIPES": "wet wipes, napkins, toilet paper, paper towels (yaş salfet, salfet, tualet kağızı)",
    "MED.SYRINGES": "medical syringes, enemas, catheters (tibbi şpris, kateter, klizma)",
    "TOWELS": "textile towels (dəsmal)",
    "PUBLIC UTILITIES WATER": "water supply / sewerage utility services billed as goods (su təchizatı, kanalizasiya)",
    "DENTAL MEDICINE": "dental medicines and materials (diş təbabəti dərmanları/materialları)",
}


def build_prompt(text: str) -> tuple[str, str]:
    group_lines = "\n".join(f"- {group}: {GROUP_HINTS[group]}" for group in GROUPS)
    system = (
        "You classify Azerbaijani invoice line items. Decide if the item is a physical GOOD (Mal) "
        "or a SERVICE (Xidmət). Services are works/activities (construction, transport, repair, "
        "installation, consulting); goods are physical products. If it is a Good, also assign exactly "
        "one product group from the list; if it is a Service, group is null. Output ONLY one JSON "
        'object, no prose, no markdown: {"label": "Good"|"Service", "group": <one of the groups '
        "or null>, \"confidence\": <0..1>}. confidence = your probability the label+group are "
        f"correct.\n\nProduct groups:\n{group_lines}"
    )
    return system, text


def _strip_fences(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.I | re.S).strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.I | re.S)
    return fence.group(1).strip() if fence else cleaned


def _first_balanced_object(text: str) -> Optional[str]:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def extract_json(text: str) -> dict[str, Any]:
    try:
        obj_text = _first_balanced_object(_strip_fences(text))
        if not obj_text:
            return {}
        obj = json.loads(obj_text)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _normalize_label(value: Any) -> Optional[str]:
    label = str(value or "").strip().casefold()
    if label.startswith(("service", "xidmət", "xidmet")):
        return "Service"
    if label.startswith(("good", "mal")):
        return "Good"
    return None


def _normalize_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except Exception:
        return 0.0
    if confidence < 0.0:
        return 0.0
    if confidence > 1.0:
        return 1.0
    return confidence


def normalize(obj: dict[str, Any]) -> dict[str, Any]:
    label = _normalize_label(obj.get("label"))
    group = obj.get("group")
    if not isinstance(group, str) or group not in GROUPS:
        group = None
    if label != "Good":
        group = None
    return {
        "label": label,
        "group": group,
        "confidence": _normalize_confidence(obj.get("confidence")),
    }


def classify(text: str, provider: Optional[str] = None, model: Optional[str] = None) -> dict[str, Any]:
    provider = provider or DEFAULT_PROVIDER
    model = model or DEFAULT_MODELS.get(provider)
    try:
        system, user = build_prompt(text)
        raw = call_llm(system, user, provider, model)
        obj = extract_json(raw)
        norm = normalize(obj)
        return {
            "label": norm["label"],
            "group": norm["group"],
            "confidence": norm["confidence"],
            "ok": bool(obj) and norm["label"] is not None,
            "raw": raw,
            "provider": provider,
            "model": model,
        }
    except Exception as exc:
        return {
            "label": None,
            "group": None,
            "confidence": 0.0,
            "ok": False,
            "raw": "",
            "provider": provider,
            "model": model,
            "error": str(exc),
        }


def classify_with_escalation(
    text: str,
    local: Optional[tuple[str, str]] = None,
    cloud: Optional[tuple[str, str]] = None,
    threshold: float = 0.75,
) -> dict[str, Any]:
    default_pair = (DEFAULT_PROVIDER, DEFAULT_MODELS.get(DEFAULT_PROVIDER) or "")
    local = local or default_pair
    cloud = cloud or default_pair
    local_result = classify(text, local[0], local[1])
    if local_result.get("ok") and float(local_result.get("confidence") or 0.0) >= threshold:
        chosen = local_result
        tier = "local"
        cloud_result = None
    else:
        cloud_result = classify(text, cloud[0], cloud[1])
        if cloud_result.get("ok"):
            chosen = cloud_result
            tier = "cloud"
        else:
            chosen = local_result
            tier = "local"
    return {
        "label": chosen.get("label"),
        "group": chosen.get("group"),
        "confidence": chosen.get("confidence"),
        "tier": tier,
        "local": local_result,
        "cloud": cloud_result,
    }
