"""Search tools the Research agent calls."""
from __future__ import annotations


def search_policy(retriever, query: str, k: int = 5) -> list[dict]:
    return retriever.search(query, k=k, source_type="policy")


def search_emr(retriever, query: str, k: int = 5) -> list[dict]:
    return retriever.search(query, k=k, source_type="emr")
