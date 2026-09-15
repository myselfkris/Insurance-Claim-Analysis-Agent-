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
    letter TEXT,
    edited_letter TEXT,
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


def save_case(state: dict, trace: list | None = None) -> int:
    conn = _connect()
    result = {
        "denial_type": (state.get("plan") or {}).get("denial_type"),
        "codes": state.get("codes"),
        "verdict": state.get("verdict"),
        "confidence": state.get("confidence_score"),
        "evidence": state.get("retrieved_evidence", []),
        "audit_trail": state.get("audit_trail", []),
    }
    letter = (state.get("letter_draft") or {}).get("letter", "")
    conn.execute(
        "INSERT INTO cases (denial_text, case_text, letter, result_json, status, confidence, created_at) "
        "VALUES (?, ?, ?, ?, 'pending_approval', ?, ?)",
        (state.get("denial_text", ""), state.get("case_text", ""), letter,
         json.dumps(result), state.get("confidence_score", 0.0), time.time()),
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
            "SELECT id, status, confidence, created_at, letter FROM cases WHERE status = ? ORDER BY id DESC",
            (status,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, status, confidence, created_at, letter FROM cases ORDER BY id DESC"
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def approve(case_id: int, edited_letter: str | None = None) -> None:
    _update_status(case_id, "approved", edited_letter)


def reject(case_id: int, edited_letter: str | None = None) -> None:
    _update_status(case_id, "rejected", edited_letter)


def _update_status(case_id: int, status: str, edited_letter: str | None) -> None:
    conn = _connect()
    if edited_letter is not None:
        conn.execute(
            "UPDATE cases SET status = ?, edited_letter = ?, reviewed_at = ? WHERE id = ?",
            (status, edited_letter, time.time(), case_id),
        )
    else:
        conn.execute(
            "UPDATE cases SET status = ?, reviewed_at = ? WHERE id = ?",
            (status, time.time(), case_id),
        )
    conn.commit()
    conn.close()
