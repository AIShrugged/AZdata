"""Tier-2 auto-resolver for uncertain HS classifications.

When Tier-1 (retrieve + rerank) is uncertain, this resolves a brand / model / colloquial item
name to its GENERIC product + likely HS area, so re-retrieval can find the right code. Two layers:

  • KNOWLEDGE resolution (always available) — the LLM already knows "Kachka"=duck, "McCain"=frozen
    potato, etc. No NEW external surface beyond the LLM you already use.
  • WEB lookup (OPTIONAL, privacy-gated) — only runs when AZDATA_WEB_SEARCH is enabled (or a request
    opts in). Sends the item text to an external search engine, so it is OFF BY DEFAULT for privacy.

Every resolution is written to a LEARNING STORE (learned_synonyms.json) so the same pattern is
folded into the index enrichment and auto-resolves in Tier-1 next time (convergence to automatic).
"""
from __future__ import annotations

import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from classify import extract_json
from nlsql import call_llm

LEARNED = ROOT / "data/processed/learned_synonyms.json"
WEB_TIMEOUT = int(os.environ.get("AZDATA_WEB_TIMEOUT", "8"))


def web_enabled(override: Optional[bool] = None) -> bool:
    """Web search is OFF by default (privacy). Enable globally via AZDATA_WEB_SEARCH, or per request."""
    if override is not None:
        return bool(override)
    return os.environ.get("AZDATA_WEB_SEARCH", "off").strip().lower() in ("on", "1", "true", "yes")


def web_lookup(query: str, web: Optional[bool] = None) -> str:
    """External web lookup — ONLY runs when web search is enabled. Best-effort, no-key (DuckDuckGo
    instant-answer); returns a short factual snippet or '' (never raises)."""
    if not web_enabled(web):
        return ""
    try:
        url = "https://api.duckduckgo.com/?format=json&no_html=1&q=" + urllib.parse.quote(query)
        with urllib.request.urlopen(url, timeout=WEB_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        snippet = data.get("AbstractText") or data.get("Definition") or data.get("Heading") or ""
        if not snippet:
            for t in data.get("RelatedTopics", [])[:1]:
                snippet = t.get("Text", "") if isinstance(t, dict) else ""
        return str(snippet)[:400]
    except Exception:
        return ""


def resolve(item_text: str, provider: str, model: Optional[str], web: Optional[bool] = None) -> dict[str, Any]:
    """Resolve a (brand/colloquial) item to its generic product + retrieval keywords.
    Uses LLM knowledge always; web facts only when enabled. Returns {product, keywords, used_web}."""
    facts = web_lookup(item_text, web)
    system = (
        "You resolve an Azerbaijani invoice line item to its REAL product for HS customs classification. "
        "The item may be a brand, model number, or colloquial/foreign name (e.g. 'Kachka' = duck, "
        "'McCain' = frozen potato/veg). Identify the GENERIC product, its FORM (live/fresh/frozen/processed), "
        "and likely HS area. Use the unit as a cue ('q' = qram = grams, 'kq' = kg, 'l' = litres) — small "
        "packaged weights are products, not live animals or machinery. "
        + (f"Web reference: {facts}\n" if facts else "")
        + 'Output ONLY JSON: {"product":"<generic product in English>","keywords":"<comma-separated '
        'synonyms in AZ/EN/RU including the original term>"}.'
    )
    try:
        j = extract_json(call_llm(system, f"ITEM: {item_text}", provider, model))
        return {"product": str(j.get("product", "")).strip(),
                "keywords": str(j.get("keywords", "")).strip(),
                "used_web": bool(facts)}
    except Exception as exc:
        return {"product": "", "keywords": "", "used_web": bool(facts), "error": str(exc)}


def learn(term: str, keywords: str) -> None:
    """Persist a resolved synonym so it can be folded into the index enrichment (re-embed) and
    auto-resolve in Tier-1 next time. Local file — no external surface."""
    term = (term or "").strip()
    if not term or not keywords:
        return
    try:
        store = json.load(open(LEARNED, encoding="utf-8")) if LEARNED.exists() else {}
        store[term.lower()] = keywords
        json.dump(store, open(LEARNED, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    except Exception:
        pass
