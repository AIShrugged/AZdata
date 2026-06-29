"""Human-in-the-loop review queue (SQLite) for items the classifier abstained on,
plus a durable corrections store that feeds back into the RAG index.

When the classifier returns needs_review (out-of-taxonomy OTHER, low confidence, or a
flagged item), the API enqueues it here. A reviewer then accepts / corrects / marks it a
data-error; accepted+corrected items are appended to corrections.csv and (live) to the
in-memory RAG index so future similar items classify better.
"""
from __future__ import annotations

import csv
import datetime as dt
import sqlite3
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data/review_queue.db"
CORRECTIONS = ROOT / "data/processed/corrections.csv"
_DECISION_STATUS = {"accept": "accepted", "correct": "corrected", "data_error": "data_error"}


def _now() -> str:
    return dt.datetime.utcnow().isoformat(timespec="seconds")


def _conn() -> sqlite3.Connection:
    DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _conn() as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS review_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT, text TEXT UNIQUE, reason TEXT,
                pred_label TEXT, pred_group TEXT, confidence REAL, is_mixed INTEGER,
                status TEXT DEFAULT 'pending',
                corrected_label TEXT, corrected_group TEXT, reviewer TEXT, reviewed_at TEXT
            )"""
        )


def classify_reason(result: dict) -> str:
    """Triage WHY an item needs review — drives where the reviewer routes it."""
    text = str(result.get("item") or result.get("text") or "").strip()
    conf = float(result.get("confidence") or 0.0)
    if result.get("group") == "OTHER":
        return "taxonomy_gap"          # a real good with no matching group
    if len(text) < 3 or conf < 0.3:
        return "data_error"            # garbled / placeholder / non-item
    if result.get("is_mixed"):
        return "ambiguous"             # composite / borderline
    return "low_confidence"


def enqueue(result: dict) -> Optional[int]:
    init_db()
    text = str(result.get("item") or result.get("text") or "").strip()
    if not text:
        return None
    with _conn() as conn:
        row = conn.execute("SELECT id FROM review_queue WHERE text = ?", (text,)).fetchone()
        if row:
            return int(row["id"])  # de-dup: same item already queued
        cur = conn.execute(
            "INSERT INTO review_queue (created_at,text,reason,pred_label,pred_group,confidence,is_mixed,status)"
            " VALUES (?,?,?,?,?,?,?,'pending')",
            (_now(), text, classify_reason(result), result.get("label"), result.get("group"),
             float(result.get("confidence") or 0.0), 1 if result.get("is_mixed") else 0),
        )
        return int(cur.lastrowid)


def list_queue(status: str = "pending", limit: int = 100) -> list[dict]:
    init_db()
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM review_queue WHERE status = ? ORDER BY id DESC LIMIT ?", (status, limit)
        ).fetchall()
        return [dict(r) for r in rows]


def resolve(item_id: int, decision: str, corrected_label: Optional[str] = None,
            corrected_group: Optional[str] = None, reviewer: str = "reviewer") -> dict:
    """decision ∈ {accept, correct, data_error}. Returns the captured correction (if any)."""
    init_db()
    status = _DECISION_STATUS.get(decision)
    if status is None:
        raise ValueError(f"unknown decision: {decision}")
    with _conn() as conn:
        row = conn.execute("SELECT * FROM review_queue WHERE id = ?", (item_id,)).fetchone()
        if row is None:
            raise ValueError(f"no queue item {item_id}")
        final_label = corrected_label if decision == "correct" else row["pred_label"]
        final_group = corrected_group if decision == "correct" else row["pred_group"]
        conn.execute(
            "UPDATE review_queue SET status=?, corrected_label=?, corrected_group=?, reviewer=?, reviewed_at=? WHERE id=?",
            (status, final_label, final_group, reviewer, _now(), item_id),
        )
    correction = None
    if decision in ("accept", "correct") and final_label:  # data_error is NOT fed back
        correction = {"text": row["text"], "label": final_label, "group": (final_group or "")}
        _append_correction(correction)
    return {"id": item_id, "status": status, "correction": correction}


def _append_correction(corr: dict) -> None:
    CORRECTIONS.parent.mkdir(parents=True, exist_ok=True)
    is_new = not CORRECTIONS.exists()
    with open(CORRECTIONS, "a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if is_new:
            writer.writerow(["text", "label", "group"])
        writer.writerow([corr["text"], corr["label"], corr["group"]])


def stats() -> dict[str, Any]:
    init_db()
    with _conn() as conn:
        by_status = {r["status"]: r["n"] for r in conn.execute("SELECT status, count(*) n FROM review_queue GROUP BY status")}
        by_reason = {r["reason"]: r["n"] for r in conn.execute("SELECT reason, count(*) n FROM review_queue GROUP BY reason")}
    reviewed = sum(v for k, v in by_status.items() if k != "pending")
    corrected = by_status.get("corrected", 0)
    n_corrections = (sum(1 for _ in open(CORRECTIONS, encoding="utf-8")) - 1) if CORRECTIONS.exists() else 0
    return {
        "pending": by_status.get("pending", 0),
        "reviewed": reviewed,
        "override_rate": round(corrected / reviewed, 3) if reviewed else 0.0,
        "by_status": by_status,
        "by_reason": by_reason,
        "corrections_captured": max(0, n_corrections),
    }
