"""Consolidate Task-2 eval results into data/processed/eval_summary.json for the UI.
Metrics are the verified numbers from the held-out 1298-item test (and noted samples);
misclassified examples are loaded from the saved mis_*.json files when present."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"


def load_mis(name: str, limit: int = 12) -> list[dict]:
    fp = P / name
    if not fp.exists():
        return []
    try:
        rows = json.loads(fp.read_text(encoding="utf-8"))
    except Exception:
        return []
    out = []
    for r in rows[:limit]:
        out.append({
            "text": r.get("text", ""),
            "true": f'{r.get("true_label","")}/{r.get("true_group","") or "-"}',
            "pred": f'{r.get("pred_label","")}/{r.get("pred_group","") or "-"}',
        })
    return out


summary = {
    "task2": {
        "headline": "A 24 GB local model reaches 99% — fully private, with cloud as an optional escalation tier.",
        "test_set": "held-out test, 1298 items (train/dev/test = 6050/1295/1298, stratified)",
        "models": [
            {"name": "qwen3.5 9.7B", "where": "local", "config": "no retrieval", "label": 74.1, "group": 48.4, "fully": 60.8, "note": "baseline (166-item sample)"},
            {"name": "qwen3.5 35B", "where": "local", "config": "+ RAG (k=16)", "label": 99.08, "group": 99.18, "fully": 98.92, "note": "held-out 1298"},
            {"name": "qwen3.5 35B", "where": "local", "config": "+ RAG + prompt-opt", "label": 99.38, "group": 99.39, "fully": 99.00, "note": "held-out 1298 — BEST LOCAL", "best_local": True},
            {"name": "qwen3.5 122B", "where": "cloud/API", "config": "+ RAG (k=16)", "label": 99.46, "group": 99.39, "fully": 99.31, "note": "held-out 1298"},
            {"name": "gpt-5.5", "where": "cloud", "config": "batched", "label": 98.2, "group": 97.6, "fully": 98.2, "note": "166-item sample"},
        ],
        "journey": [
            {"stage": "35B, no retrieval", "fully": 85.0},
            {"stage": "+ RAG (few-shot)", "fully": 98.92},
            {"stage": "+ agentic prompt-opt", "fully": 99.00},
        ],
        "best_local_per_group": {"BAKERY": 99, "CANNED FISH": 99, "WIPES": 100, "MED.SYRINGES": 100, "TOWELS": 100, "PUBLIC UTILITIES WATER": 100, "DENTAL MEDICINE": 0},
        "best_local_confusion": {"Good→Good": 976, "Good→Service": 1, "Service→Service": 314, "Service→Good": 7},
        "misclassified": load_mis("mis_test_qwen3.5-35b-a3b_k16.json") or load_mis("mis_test_qwen3.5-122b-a10b_k16.json"),
        "method": "RAG retrieves the k most-similar solved items as few-shot; an agentic loop (122B optimizer) rewrites the classifier instruction from dev errors; a two-tier router escalates only low-confidence items to the cloud.",
    },
    "task1": {
        "scenario1": "‘turnover for the last 4 days, taxpayer 1234567890’ → 96000.00 (per-day 25000/18000/31000/22000), verified in EN + AZ against psql.",
        "guard": "Every generated query passes a sqlglot guard (single read-only SELECT, table/column whitelist, forced LIMIT) + runs in a read-only Postgres session.",
    },
}

out = P / "eval_summary.json"
out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"wrote {out}")
print("models:", len(summary["task2"]["models"]), "| misclassified examples:", len(summary["task2"]["misclassified"]))
