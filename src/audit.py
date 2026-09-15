"""Audit-trail helper.

Each node returns ``{"audit_trail": [entry(...)]}`` so the graph accumulates a
complete, traceable log of every agent step (the human's sign-off evidence).
"""
from __future__ import annotations

import time
from typing import Any


def entry(node: str, role: str | None, **summary: Any) -> dict:
    """Build one audit-trail record."""
    return {
        "ts": time.time(),
        "node": node,
        "role": role,
        **summary,
    }
