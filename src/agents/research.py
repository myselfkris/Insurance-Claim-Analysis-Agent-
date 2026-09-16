"""Research: retrieve facts NEUTRALLY and extract coverage criteria.

Neutral means we gather the facts (both directions), NOT "find evidence to help
the patient". That neutrality is what lets the evaluator honestly reach NOT MET.
"""
from __future__ import annotations

from ..audit import entry
from ..llm import invoke_structured
from ..state import CriteriaSpec, GraphState

MOCK_CRITERIA = {
    "denial_type": "medical necessity",
    "criteria": [
        {"id": "A", "description": "knee pain that limits daily activities"},
        {"id": "B", "description": "at least six weeks of conservative care (physical therapy or anti-inflammatory medicine)"},
        {"id": "C", "description": "X-ray showing meniscal tear or joint space narrowing"},
    ],
    "expression": "A AND B AND C",
    "policy_citations": ["POL-KNEE-MRI-1"],
}

# Neutral, generic section queries — not "help the patient" queries.
NEUTRAL_QUERIES = ["symptoms", "treatment history", "imaging", "medications", "diagnosis"]


def make_research_node(retriever):
    def node(state: GraphState) -> dict:
        denial = state.get("denial_text", "")
        case = state.get("case_text", "")

        # 1. Find the relevant policy (source of the coverage criteria).
        policy_hits = retriever.search(denial, k=6, source_type="policy")

        # 2. Retrieve the patient's facts neutrally, across all record sections.
        emr_hits: list[dict] = []
        seen: set[str] = set()
        for q in NEUTRAL_QUERIES:
            for hit in retriever.search(q, k=4, source_type="emr"):
                if hit["citation_key"] not in seen:
                    seen.add(hit["citation_key"])
                    emr_hits.append(hit)

        evidence = policy_hits + emr_hits

        # 3. Extract criteria + boolean expression from the policy.
        policy_block = "\n\n".join(f"[{h['citation_key']}] {h['text']}" for h in policy_hits)
        prompt = (
            f"Denial:\n{denial}\n\n"
            f"Policy (source of truth for the criteria):\n{policy_block}\n\n"
            "Extract the coverage criteria. Give each criterion a short id (A, B, C, ...) "
            "and write the boolean expression using AND / OR / NOT / parentheses "
            "(e.g. 'A AND B AND C' or '(A AND B) OR C')."
        )
        spec = invoke_structured("research", CriteriaSpec, prompt, mock=MOCK_CRITERIA)

        return {
            "criteria_spec": spec.model_dump(),
            "retrieved_evidence": evidence,
            "audit_trail": [entry(
                "research", "research",
                denial_type=spec.denial_type,
                criteria=[c.id for c in spec.criteria],
                expression=spec.expression,
                n_evidence=len(evidence),
            )],
        }

    return node
