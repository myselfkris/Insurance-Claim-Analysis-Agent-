"""Shared graph state and structured-output schemas.

Complex objects are stored as plain dicts in the graph state (pydantic models
are used for LLM structured output and validated before being dumped to dict).
List fields use ``operator.add`` so nodes can return incremental entries.
"""
from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Structured-output schemas (used with .with_structured_output)
# --------------------------------------------------------------------------- #
class Evidence(BaseModel):
    citation_key: str = Field(..., description="Stable key, e.g. POL-001 or EMR-002")
    source_id: str
    source_type: Literal["policy", "emr"]
    text: str
    score: float = 0.0


class DenialPlan(BaseModel):
    denial_type: str
    reasoning: str
    criteria_needed: list[str]
    initial_queries: list[str]


class SufficiencyCheck(BaseModel):
    sufficient: bool
    missing: list[str] = Field(default_factory=list)
    reformulated_queries: list[str] = Field(default_factory=list)


class CodeMapping(BaseModel):
    icd10: list[str] = Field(default_factory=list)
    cpt: list[str] = Field(default_factory=list)
    notes: str = ""


class LetterDraft(BaseModel):
    letter: str
    citations: list[str] = Field(default_factory=list, description="citation keys used in the letter")
    codes: CodeMapping


class CitationCheck(BaseModel):
    citation_key: str
    supported: bool
    reason: str


class Verdict(BaseModel):
    verdict: Literal["pass", "fail"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    feedback: str
    reason: Literal["ready", "evidence_insufficient", "writing_citation"]
    citation_checks: list[CitationCheck] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Graph state
# --------------------------------------------------------------------------- #
class GraphState(TypedDict, total=False):
    denial_text: str
    case_text: str
    plan: dict | None
    retrieved_evidence: Annotated[list[dict], operator.add]
    retrieval_queries: Annotated[list[str], operator.add]
    codes: dict | None
    letter_draft: dict | None
    verdict: dict | None
    confidence_score: float
    cycle_count: int
    audit_trail: Annotated[list[dict], operator.add]
