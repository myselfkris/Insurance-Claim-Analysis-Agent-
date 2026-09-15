"""Research agent: plan the appeal, retrieve evidence, check sufficiency, re-query."""
from __future__ import annotations

from ..audit import entry
from ..config import settings
from ..llm import invoke_structured
from ..state import DenialPlan, GraphState, SufficiencyCheck

MOCK_PLAN = {
    "denial_type": "medical necessity",
    "reasoning": (
        "The denial says the knee MRI is not medically necessary, but the record "
        "shows eight weeks of conservative care and an abnormal X-ray."
    ),
    "criteria_needed": [
        "knee pain limiting daily activities",
        "at least six weeks of conservative care",
        "X-ray showing meniscal tear or joint space narrowing",
    ],
    "initial_queries": [
        "knee MRI coverage criteria",
        "conservative care six weeks physical therapy anti-inflammatory medicine",
    ],
}

MOCK_SUFFICIENCY = {"sufficient": True, "missing": [], "reformulated_queries": []}


def make_research_node(retriever):
    def node(state: GraphState) -> dict:
        denial = state.get("denial_text", "")
        case = state.get("case_text", "")
        prompt = f"Denial:\n{denial}\n\nCase note:\n{case}\n\nReturn a plan."
        plan = invoke_structured("research", DenialPlan, prompt, mock=MOCK_PLAN)

        queries = list(plan.initial_queries)
        evidence: list[dict] = []
        seen: set[str] = set()
        for _ in range(settings.max_research_rounds):
            for q in queries:
                for hit in retriever.search(q, k=4):
                    if hit["citation_key"] not in seen:
                        seen.add(hit["citation_key"])
                        evidence.append(hit)
            check_prompt = (
                f"Criteria needed:\n{plan.criteria_needed}\n\nEvidence found:\n"
                + "\n".join(f"[{e['citation_key']}] {e['text'][:120]}" for e in evidence)
            )
            check = invoke_structured("research", SufficiencyCheck, check_prompt, mock=MOCK_SUFFICIENCY)
            if check.sufficient:
                break
            queries = check.reformulated_queries or [f"{plan.denial_type} criteria"]

        return {
            "plan": plan.model_dump(),
            "retrieved_evidence": evidence,
            "retrieval_queries": queries,
            "audit_trail": [entry("research", "research", denial_type=plan.denial_type, n_evidence=len(evidence))],
        }

    return node
