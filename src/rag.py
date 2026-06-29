from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from classify import GROUPS, GROUP_HINTS, extract_json, normalize
from nlsql import DEFAULT_MODELS, PROVIDER as DEFAULT_PROVIDER, call_llm

ROOT = Path(__file__).resolve().parents[1]
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL = os.environ.get("AZDATA_EMBED_MODEL", "bge-m3")
EMBED_TIMEOUT = int(os.environ.get("AZDATA_EMBED_TIMEOUT", os.environ.get("AZDATA_LLM_TIMEOUT", "120")))
EMBED_RETRIES = int(os.environ.get("AZDATA_EMBED_RETRIES", "3"))


def embed_texts(texts: list[str], batch: int = 64) -> np.ndarray:
    embeddings: list[list[float]] = []
    for start in range(0, len(texts), batch):
        chunk = texts[start : start + batch]
        payload = {"model": EMBED_MODEL, "input": chunk}
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{OLLAMA_HOST.rstrip('/')}/api/embed",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        body = None
        for attempt in range(EMBED_RETRIES):  # timeout + retry: a stalled Ollama must not hang the API
            try:
                with urllib.request.urlopen(req, timeout=EMBED_TIMEOUT) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                break
            except Exception:
                if attempt == EMBED_RETRIES - 1:
                    raise
                time.sleep(min(2 ** attempt, 8))
        if "embeddings" not in body:
            raise RuntimeError("Ollama embed response missing 'embeddings'")
        embeddings.extend(body["embeddings"])
    return np.asarray(embeddings, dtype=np.float32)


def _l2norm(m: np.ndarray) -> np.ndarray:
    m = np.nan_to_num(np.asarray(m, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return m / norms


def build_index(rows: list[dict[str, Any]], out_prefix: Path) -> None:
    texts = [str(row["text"]) for row in rows]
    emb = embed_texts(texts)
    np.save(str(out_prefix) + ".npy", emb.astype("float32"))
    meta = [
        {"text": _json_value(row.get("text")), "label": _json_value(row.get("label")), "group": _json_value(row.get("group"))}
        for row in rows
    ]
    with open(str(out_prefix) + ".meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    dim = int(emb.shape[1]) if emb.ndim == 2 and emb.shape[0] else 0
    print(f"indexed {len(rows)} items, dim {dim}")


def _json_value(value: Any) -> Any:
    return None if pd.isna(value) else value


def load_index(out_prefix: Path) -> tuple[np.ndarray, list[dict[str, Any]]]:
    emb = np.load(str(out_prefix) + ".npy").astype("float32")
    with open(str(out_prefix) + ".meta.json", "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    return _l2norm(emb), list(meta)


def retrieve(query_text: str, emb: np.ndarray, meta: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    qe = _l2norm(embed_texts([query_text]))[0].astype(np.float64)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):  # macOS BLAS sets spurious FP flags
        sims = np.nan_to_num(emb.astype(np.float64) @ qe, nan=-1.0, posinf=-1.0, neginf=-1.0)
    idx = np.argsort(-sims)[:k]
    return [meta[int(i)] for i in idx]


DEFAULT_INSTRUCTIONS = (
    "You classify Azerbaijani invoice line items. Decide if the item is a physical GOOD (Mal) "
    "or a SERVICE (Xidmət). Services are works/activities (construction, transport, repair, "
    "installation, consulting); goods are physical products. If it is a Good, also assign exactly "
    "one product group from the list; if it is a Service, group is null. Output ONLY one JSON "
    'object, no prose, no markdown: {"label": "Good"|"Service", "group": <one of the groups '
    'or null>, "confidence": <0..1>}. confidence = your probability the label+group are correct.'
)


def build_rag_prompt(text: str, examples: list[dict[str, Any]], instructions: Optional[str] = None) -> tuple[str, str]:
    base = instructions if instructions is not None else DEFAULT_INSTRUCTIONS
    group_lines = "\n".join(f"- {group}: {GROUP_HINTS[group]}" for group in GROUPS)
    example_lines = []
    for ex in examples:
        suffix = f' / {ex["group"]}' if ex.get("group") else ""
        example_lines.append(f'- "{ex["text"]}" -> {ex["label"]}{suffix}')
    system = (
        base
        + f"\n\nProduct groups:\n{group_lines}\n\n"
        + "Similar labeled examples (item -> answer):\n"
        + "\n".join(example_lines)
    )
    return system, text


def classify_rag(
    text: str,
    emb: np.ndarray,
    meta: list[dict[str, Any]],
    k: int = 8,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    instructions: Optional[str] = None,
) -> dict[str, Any]:
    provider = provider or DEFAULT_PROVIDER
    model = model or DEFAULT_MODELS.get(provider)
    try:
        examples = retrieve(text, emb, meta, k)
        system, user = build_rag_prompt(text, examples, instructions)
        raw = call_llm(system, user, provider, model)
        norm = normalize(extract_json(raw))
        return {
            "label": norm["label"],
            "group": norm["group"],
            "confidence": norm["confidence"],
            "ok": norm["label"] is not None,
            "provider": provider,
            "model": model,
        }
    except Exception as exc:
        return {
            "label": None,
            "group": None,
            "confidence": 0.0,
            "ok": False,
            "provider": provider,
            "model": model,
            "error": str(exc),
        }


def _rows_from_csv(path: Path) -> list[dict[str, Any]]:
    return pd.read_csv(path).to_dict(orient="records")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    prefix = ROOT / "data/processed/train_index"

    if args.build:
        rows = _rows_from_csv(ROOT / "data/processed/train.csv")
        build_index(rows, prefix)
    if args.selftest:
        emb, meta = load_index(prefix)
        rows = _rows_from_csv(ROOT / "data/processed/dev.csv")[:3]
        for row in rows:
            predicted = classify_rag(str(row["text"]), emb, meta, k=8)
            print(
                json.dumps(
                    {
                        "text": row["text"],
                        "predicted": predicted,
                        "retrieved-example-count": min(8, len(meta)),
                    },
                    ensure_ascii=False,
                )
            )


if __name__ == "__main__":
    main()
