"""Hybrid search fusion: cosine similarity + reciprocal rank fusion (RRF)."""
from __future__ import annotations

import math


def cosine(a, b) -> float:
    """Cosine similarity between two equal-length vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def rrf_fuse(ranked_lists, k: int = 60):
    """Fuse multiple ranked lists of (idx, score) with reciprocal rank fusion.

    Returns a single list of (idx, fused_score) sorted best-first.
    """
    rrf: dict[int, float] = {}
    for hits in ranked_lists:
        for rank, (idx, _score) in enumerate(hits, start=1):
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(rrf.items(), key=lambda x: -x[1])
