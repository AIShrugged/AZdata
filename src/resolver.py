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


def _brave_search(query: str, key: str) -> str:
    url = "https://api.search.brave.com/res/v1/web/search?count=3&q=" + urllib.parse.quote(query)
    req = urllib.request.Request(url, headers={"X-Subscription-Token": key, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=WEB_TIMEOUT) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    results = ((data.get("web") or {}).get("results") or [])[:3]
    parts = [(r.get("title", "") + ": " + (r.get("description", "") or "")).strip(" :") for r in results]
    return " | ".join(p for p in parts if p)[:500]


def _ddg_search(query: str) -> str:
    url = "https://api.duckduckgo.com/?format=json&no_html=1&q=" + urllib.parse.quote(query)
    with urllib.request.urlopen(url, timeout=WEB_TIMEOUT) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    snippet = data.get("AbstractText") or data.get("Definition") or data.get("Heading") or ""
    if not snippet:
        for t in data.get("RelatedTopics", [])[:1]:
            snippet = t.get("Text", "") if isinstance(t, dict) else ""
    return str(snippet)[:400]


def web_lookup(query: str, web: Optional[bool] = None) -> str:
    """External web lookup — ONLY runs when web search is enabled (privacy). Uses Brave Search when
    BRAVE_API_KEY is set, else a no-key DuckDuckGo fallback. Returns a short snippet or '' (never raises)."""
    if not web_enabled(web):
        return ""
    key = os.environ.get("BRAVE_API_KEY")
    try:
        return _brave_search(query, key) if key else _ddg_search(query)
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


def learn(term: str, keywords: str, code: Optional[str] = None) -> None:
    """Persist a TRUSTED resolution (term → keywords + the HS code it resolved to) so apply_learned.py
    can fold the keywords into that code's index entry → auto-resolve in Tier-1 next time. Local file."""
    term = (term or "").strip()
    if not term or not keywords:
        return
    try:
        store = json.load(open(LEARNED, encoding="utf-8")) if LEARNED.exists() else {}
        store[term.lower()] = {"keywords": keywords, "code": str(code or "")}
        json.dump(store, open(LEARNED, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    except Exception:
        pass


def _load_store() -> dict[str, Any]:
    try:
        return json.load(open(LEARNED, encoding="utf-8")) if LEARNED.exists() else {}
    except Exception:
        return {}


def unfolded_count() -> int:
    """How many learned (term→code) resolutions are waiting to be folded into the index.
    Drives the threshold auto-trigger (no clock — fires while the app is running)."""
    return sum(1 for v in _load_store().values() if isinstance(v, dict) and v.get("code"))


def list_learned() -> list[dict[str, Any]]:
    """The pending learned resolutions — for human review BEFORE they are folded into the index."""
    return [{"term": term, "keywords": info.get("keywords", ""), "code": info.get("code", "")}
            for term, info in _load_store().items() if isinstance(info, dict)]


def forget(term: str) -> bool:
    """Reject a learned resolution before it is applied (e.g. a wrong auto-resolution)."""
    term = (term or "").strip().lower()
    store = _load_store()
    if term in store:
        del store[term]
        try:
            json.dump(store, open(LEARNED, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            return True
        except Exception:
            pass
    return False
