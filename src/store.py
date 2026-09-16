"""SQLite persistence for cases (async approval workflow — no interrupt)."""
from __future__ import annotations

import json
import sqlite3
import time

from .config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    denial_text TEXT,
    case_text TEXT,
    decision TEXT,
    output_text TEXT,
    edited_text TEXT,
    result_json TEXT,
    status TEXT DEFAULT 'pending_approval',
    confidence REAL,
    created_at REAL,
    reviewed_at REAL
);
"""


def _connect() -> sqlite3.Connection:
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.db_path)
    conn.execute(SCHEMA)
    return conn


def save_case(state: dict) -> int:
    conn = _connect()
    output = state.get("output") or {}
    output_text = output.get("letter") or output.get("explanation") or output.get("request") or ""
    result = {
        "decision": state.get("decision"),
        "criteria_spec": state.get("criteria_spec"),
        "evaluation": state.get("evaluation"),
        "output": output,
        "verdict": state.get("verdict"),
        "confidence": state.get("confidence_score"),
        "evidence": state.get("retrieved_evidence", []),
        "audit_trail": state.get("audit_trail", []),
    }
    conn.execute(
        "INSERT INTO cases (denial_text, case_text, decision, output_text, result_json, status, confidence, created_at) "
        "VALUES (?, ?, ?, ?, ?, 'pending_approval', ?, ?)",
        (state.get("denial_text", ""), state.get("case_text", ""), state.get("decision", ""),
         output_text, json.dumps(result), state.get("confidence_score", 0.0), time.time()),
    )
    conn.commit()
    row_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.close()
    return row_id


def list_cases(status: str | None = None) -> list[dict]:
    conn = _connect()
    conn.row_factory = sqlite3.Row
    if status:
        rows = conn.execute(
            "SELECT id, decision, status, confidence, created_at, output_text FROM cases "
            "WHERE status = ? ORDER BY id DESC", (status,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, decision, status, confidence, created_at, output_text FROM cases ORDER BY id DESC"
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def approve(case_id: int, edited_text: str | None = None) -> None:
    _update_status(case_id, "approved", edited_text)


def reject(case_id: int, edited_text: str | None = None) -> None:
    _update_status(case_id, "rejected", edited_text)


def _update_status(case_id: int, status: str, edited_text: str | None) -> None:
    conn = _connect()
    if edited_text is not None:
        conn.execute(
            "UPDATE cases SET status = ?, edited_text = ?, reviewed_at = ? WHERE id = ?",
            (status, edited_text, time.time(), case_id),
        )
    else:
        conn.execute(
            "UPDATE cases SET status = ?, reviewed_at = ? WHERE id = ?",
            (status, time.time(), case_id),
        )
    conn.commit()
    conn.close()
