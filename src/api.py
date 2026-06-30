from __future__ import annotations

from collections import OrderedDict
import json
import logging
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from catalog import build_catalog
from nlsql import PROVIDER as DEFAULT_PROVIDER, answer
import rag, eqm, router, review, resolver
import numpy as np

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("azdata")
_CACHE_MAX = int(os.environ.get("AZDATA_CACHE_MAX", "1024"))
_query_cache: "OrderedDict[tuple, Any]" = OrderedDict()
_classify_cache: "OrderedDict[tuple, Any]" = OrderedDict()
API_KEY = os.environ.get("AZDATA_API_KEY", "")  # "" = auth disabled (local dev)
CORS_ORIGINS = [o for o in os.environ.get("AZDATA_CORS_ORIGINS", "http://127.0.0.1:8642,http://localhost:8642").split(",") if o]
RATE_LIMIT = int(os.environ.get("AZDATA_RATE_LIMIT", "60"))  # requests per 60s per client
DEBUG = os.environ.get("AZDATA_DEBUG", "false").strip().lower() in ("1", "true", "yes", "on")
ALLOWED_PROVIDERS = {p for p in os.environ.get("AZDATA_ALLOWED_PROVIDERS", "openrouter,ollama").split(",") if p}
ALLOWED_MODELS = {m for m in os.environ.get("AZDATA_ALLOWED_MODELS", "qwen/qwen3.5-35b-a3b,qwen/qwen3.5-122b-a10b,qwen3.5:latest").split(",") if m}
_cache_lock = threading.Lock()
_rate_state: "dict[str, list]" = {}
_rate_lock = threading.Lock()
_index_lock = threading.Lock()


def _cache_get(cache, key):
    with _cache_lock:
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
        return None


def _cache_put(cache, key, value):
    with _cache_lock:
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > _CACHE_MAX:
            cache.popitem(last=False)


def _status_for(result: dict) -> int:
    # input/guard errors are the client's fault (400); DB/provider outages are transient (503).
    return 400 if result.get("error_kind") == "input" else 503


def _client_id(request: Request) -> str:
    return request.headers.get("x-api-key") or (request.client.host if request.client else "anon")


def _check_request(request: Request) -> None:
    # Auth (opt-in): if AZDATA_API_KEY is set, require a matching X-API-Key header.
    if API_KEY:
        if request.headers.get("x-api-key") != API_KEY:
            raise HTTPException(status_code=401, detail="invalid or missing API key")
    # Per-client sliding-window rate limit.
    cid = _client_id(request)
    now = time.time()
    with _rate_lock:
        hits = [t for t in _rate_state.get(cid, []) if now - t < 60.0]
        if len(hits) >= RATE_LIMIT:
            raise HTTPException(status_code=429, detail="rate limit exceeded")
        hits.append(now)
        _rate_state[cid] = hits


def _check_models(provider: str, *models: Optional[str]) -> None:
    if provider and provider not in ALLOWED_PROVIDERS:
        raise HTTPException(status_code=400, detail=f"provider not allowed: {provider}")
    for m in models:
        if m and m not in ALLOWED_MODELS:
            raise HTTPException(status_code=400, detail=f"model not allowed: {m}")


def _sanitize(result: dict) -> dict:
    # In non-debug mode, never leak raw model SQL or internal exception text to clients.
    out = dict(result)
    if not DEBUG:
        out.pop("raw_sql", None)
        if out.get("error"):
            cid = uuid.uuid4().hex[:12]
            log.warning("error correlation_id=%s detail=%s", cid, out.get("error"))
            out = {k: out.get(k) for k in ("error_kind",) if out.get(k) is not None}
            out["error"] = "internal error"
            out["correlation_id"] = cid
    return out


def _apply_correction_to_index(correction: Optional[dict]) -> None:
    """Live feedback: embed a human correction and append it to the in-memory RAG index so
    future similar items benefit immediately (the durable copy lives in corrections.csv)."""
    global RAG_EMB, RAG_META
    if not correction or not TASK2_READY:
        return
    try:
        vec = rag._l2norm(rag.embed_texts([correction["text"]]))  # (1, dim), normalized
        meta_row = {"text": correction["text"], "label": correction["label"], "group": (correction.get("group") or None)}
        with _index_lock:
            RAG_EMB = np.vstack([RAG_EMB, vec])
            RAG_META = RAG_META + [meta_row]
    except Exception as exc:
        log.warning("correction index append failed: %s", exc)


app = FastAPI(title="AZdata e-invoice AI", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    CATALOG: Optional[dict[str, Any]] = build_catalog()
except Exception:
    CATALOG = None

try:
    RAG_EMB, RAG_META = rag.load_index(ROOT / "data/processed/train_index")
    EQM_EMB, EQM_META = eqm.load_eqm_index()
    instruction_path = ROOT / "data/processed/best_instructions.txt"
    INSTRUCTIONS = (
        instruction_path.read_text(encoding="utf-8")
        if instruction_path.exists()
        else None
    )
    TASK2_READY = True
except Exception:
    RAG_EMB = RAG_META = EQM_EMB = EQM_META = INSTRUCTIONS = None
    TASK2_READY = False


class QueryRequest(BaseModel):
    question: str
    provider: Optional[str] = None
    model: Optional[str] = None
    ref_date: Optional[str] = None


class ClassifyRequest(BaseModel):
    text: str
    threshold: Optional[float] = None
    local_model: Optional[str] = None
    strong_model: Optional[str] = None
    assign_hs: bool = True
    web_search: Optional[bool] = None  # privacy: opt-in web lookup for hard items (None = server default, off)


class ReviewResolveRequest(BaseModel):
    id: int
    decision: str  # accept | correct | data_error
    corrected_label: Optional[str] = None
    corrected_group: Optional[str] = None
    reviewer: Optional[str] = None


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "default_provider": DEFAULT_PROVIDER,
        "task2_ready": TASK2_READY,
        "web_search_default": resolver.web_enabled(),
    }


@app.get("/catalog")
def catalog() -> dict[str, Any]:
    if CATALOG is None:
        raise HTTPException(status_code=503, detail="catalog unavailable")
    return CATALOG


@app.post("/query")
def query(request: Request, req: QueryRequest) -> Any:
    start = time.time()
    provider = req.provider or DEFAULT_PROVIDER
    _check_request(request)
    _check_models(provider, req.model)
    key = (req.question, provider, req.model, req.ref_date)
    cached = _cache_get(_query_cache, key)
    if cached is not None and not cached.get("error"):
        result = dict(cached)
        result["cached"] = True
        ms = int((time.time() - start) * 1000)
        log.info("query question=%r latency_ms=%d cache=%s provider=%s", req.question[:60], ms, "hit", provider)
        return _sanitize(result)

    result = answer(
        req.question,
        provider=provider,
        model=req.model,
        ref_date=req.ref_date,
    )
    if not result.get("error"):
        _cache_put(_query_cache, key, result)
    ms = int((time.time() - start) * 1000)
    log.info("query question=%r latency_ms=%d cache=%s provider=%s", req.question[:60], ms, "miss", provider)
    if result.get("error"):
        return JSONResponse(status_code=_status_for(result), content=_sanitize(result))
    return _sanitize(result)


@app.post("/classify")
def classify(request: Request, req: ClassifyRequest) -> Any:
    start = time.time()
    _check_request(request)
    _check_models(DEFAULT_PROVIDER, req.local_model, req.strong_model)
    key = (req.text, req.threshold, req.local_model, req.strong_model, req.assign_hs)
    cached = _cache_get(_classify_cache, key)
    if cached is not None and not cached.get("error"):
        result = dict(cached)
        result["cached"] = True
        ms = int((time.time() - start) * 1000)
        log.info("classify text=%r latency_ms=%d cache=%s tier=%s escalated=%s", req.text[:60], ms, "hit", result.get("tier"), result.get("escalated"))
        return _sanitize(result)

    if not TASK2_READY:
        ms = int((time.time() - start) * 1000)
        log.info("classify text=%r latency_ms=%d cache=%s tier=%s escalated=%s", req.text[:60], ms, "miss", None, None)
        raise HTTPException(
            status_code=503,
            detail=(
                "task 2 index not built — run scripts/make_splits.py, "
                "src/rag.py --build, src/eqm.py --build"
            ),
        )
    route_kwargs: dict[str, Any] = {}
    if req.threshold is not None:
        route_kwargs["threshold"] = req.threshold
    if req.local_model is not None:
        route_kwargs["local_model"] = req.local_model
    if req.strong_model is not None:
        route_kwargs["strong_model"] = req.strong_model
    try:
        result = router.classify_item(
            req.text,
            RAG_EMB,
            RAG_META,
            EQM_EMB,
            EQM_META,
            INSTRUCTIONS,
            assign_hs=req.assign_hs,
            web=req.web_search,
            **route_kwargs,
        )
        if result.get("needs_review"):
            try:
                review.enqueue(result)  # flagged items land in the human review queue
            except Exception as exc:
                log.warning("review enqueue failed: %s", exc)
        if result.get("label") is not None and not result.get("error"):
            _cache_put(_classify_cache, key, result)  # cache real successes only — never a failure/null
        ms = int((time.time() - start) * 1000)
        log.info("classify text=%r latency_ms=%d cache=%s tier=%s escalated=%s", req.text[:60], ms, "miss", result.get("tier"), result.get("escalated"))
        if result.get("error"):
            return JSONResponse(status_code=503, content=_sanitize(result))
        return _sanitize(result)
    except Exception as exc:
        result = {"error": str(exc), "error_kind": "upstream"}
        ms = int((time.time() - start) * 1000)
        log.info("classify text=%r latency_ms=%d cache=%s tier=%s escalated=%s", req.text[:60], ms, "miss", None, None)
        return JSONResponse(status_code=503, content=_sanitize(result))


@app.get("/review/queue")
def review_queue(request: Request, status: str = "pending", limit: int = 100) -> Any:
    _check_request(request)
    return {"items": review.list_queue(status=status, limit=min(limit, 500)), "stats": review.stats()}


@app.post("/review/resolve")
def review_resolve(request: Request, req: ReviewResolveRequest) -> Any:
    _check_request(request)
    try:
        out = review.resolve(req.id, req.decision, req.corrected_label, req.corrected_group, req.reviewer or "reviewer")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    _apply_correction_to_index(out.get("correction"))  # live RAG feedback
    out["index_updated"] = bool(out.get("correction"))
    return out


@app.get("/review/stats")
def review_stats(request: Request) -> Any:
    _check_request(request)
    return review.stats()


@app.get("/evals")
def evals() -> Any:
    fp = ROOT / "data/processed/eval_summary.json"
    if not fp.exists():
        raise HTTPException(status_code=404, detail="eval summary not built — run scripts/build_eval_summary.py")
    return json.loads(fp.read_text(encoding="utf-8"))


if (ROOT / "web").is_dir():
    app.mount("/", StaticFiles(directory=str(ROOT / "web"), html=True), name="web")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("AZDATA_API_HOST", "127.0.0.1"),
        port=int(os.environ.get("AZDATA_API_PORT", "8642")),
    )
