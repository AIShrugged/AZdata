from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from catalog import build_catalog
from nlsql import PROVIDER as DEFAULT_PROVIDER
from nlsql import answer


app = FastAPI(title="AZdata e-invoice NL→SQL API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    CATALOG: dict[str, Any] | None = build_catalog()
except Exception:
    CATALOG = None


class QueryRequest(BaseModel):
    question: str
    provider: Optional[str] = None
    model: Optional[str] = None
    ref_date: Optional[str] = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "default_provider": DEFAULT_PROVIDER}


@app.get("/catalog")
def catalog() -> dict[str, Any]:
    if CATALOG is None:
        raise HTTPException(status_code=503, detail="catalog unavailable")
    return CATALOG


@app.post("/query", summary="Answer a natural-language question with SQL results.")
def query(req: QueryRequest) -> Any:
    result = answer(
        req.question,
        provider=(req.provider or DEFAULT_PROVIDER),
        model=req.model,
        ref_date=req.ref_date,
    )
    if result.get("error"):
        return JSONResponse(status_code=400, content=result)
    return result


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("AZDATA_API_HOST", "127.0.0.1")
    port = int(os.environ.get("AZDATA_API_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
