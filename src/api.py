from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from catalog import build_catalog
from nlsql import PROVIDER as DEFAULT_PROVIDER, answer
import rag, eqm, router


app = FastAPI(title="AZdata e-invoice AI", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "default_provider": DEFAULT_PROVIDER,
        "task2_ready": TASK2_READY,
    }


@app.get("/catalog")
def catalog() -> dict[str, Any]:
    if CATALOG is None:
        raise HTTPException(status_code=503, detail="catalog unavailable")
    return CATALOG


@app.post("/query")
def query(req: QueryRequest) -> Any:
    result = answer(
        req.question,
        provider=req.provider or DEFAULT_PROVIDER,
        model=req.model,
        ref_date=req.ref_date,
    )
    if result.get("error"):
        return JSONResponse(status_code=400, content=result)
    return result


@app.post("/classify")
def classify(req: ClassifyRequest) -> Any:
    if not TASK2_READY:
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
        return router.classify_item(
            req.text,
            RAG_EMB,
            RAG_META,
            EQM_EMB,
            EQM_META,
            INSTRUCTIONS,
            assign_hs=req.assign_hs,
            **route_kwargs,
        )
    except Exception as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})


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
