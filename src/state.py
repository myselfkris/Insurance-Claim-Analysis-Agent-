"""Shared graph state and structured-output schemas.

Complex objects are stored as plain dicts in the graph state (pydantic models
are used for LLM structured output and validated before being dumped to dict).
List fields use ``operator.add`` so nodes can return incremental entries.
"""
from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field


class Evidence(BaseModel):
    citation_key: str
    source_id: str
    source_type: Literal["policy", "emr"]
    text: str
    score: float = 0.0


class Criterion(BaseModel):
    id: str
    description: str


class CriteriaSpec(BaseModel):
    denial_type: str
    criteria: list[Criterion]
    expression: str
    policy_citations: list[str] = Field(default_factory=list)


class CriterionVerdict(BaseModel):
    criterion_id: str
    status: Literal["met", "not_met", "unverified"]
    reasoning: str
    citations: list[str] = Field(default_factory=list)


class Evaluation(BaseModel):
    criteria: list[CriterionVerdict]


class AppealLetter(BaseModel):
    letter: str
    citations: list[str] = Field(default_factory=list)


class UpholdExplanation(BaseModel):
    explanation: str
    unmet_criteria: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)


class DocumentRequest(BaseModel):
    request: str
    missing_criteria: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)


class CitationCheck(BaseModel):
    citation_key: str
    supported: bool
    reason: str


class JudgeRecheck(BaseModel):
    criterion_id: str
    original_status: str
    correct: bool
    corrected_status: Literal["met", "not_met", "unverified"] | None = None
    reasoning: str


class Rechecks(BaseModel):
    checks: list[JudgeRecheck] = Field(default_factory=list)


class Verdict(BaseModel):
    verdict: Literal["pass", "fail"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    feedback: str
    reason: Literal["ready", "judge_error", "grounding_error"]
    citation_checks: list[CitationCheck] = Field(default_factory=list)
    judge_rechecks: list[JudgeRecheck] = Field(default_factory=list)


class GraphState(TypedDict, total=False):
    denial_text: str
    case_text: str
    criteria_spec: dict | None
    retrieved_evidence: Annotated[list[dict], operator.add]
    evaluation: dict | None
    decision: str
    output: dict | None
    verdict: dict | None
    confidence_score: float
    cycle_count: int
    audit_trail: Annotated[list[dict], operator.add]
