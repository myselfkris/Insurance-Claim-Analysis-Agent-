"""Uphold path: explain, honestly and with evidence, WHY the denial is correct."""
from __future__ import annotations

from ..audit import entry
from ..llm import invoke_structured
from ..state import GraphState, UpholdExplanation

MOCK_UPHOLD = {
    "explanation": (
        "The denial appears CORRECT. The patient did not satisfy the required "
        "criterion for conservative care: the record shows the patient declined "
        "physical therapy, so the six-week conservative-treatment requirement is "
        "not met [EMR-KNEE-002-1]. An appeal is unlikely to succeed."
    ),
    "unmet_criteria": ["B"],
    "citations": ["EMR-KNEE-002-1"],
}


def make_uphold_node():
    def node(state: GraphState) -> dict:
        denial = state.get("denial_text", "")
        case = state.get("case_text", "")
        spec = state.get("criteria_spec", {})
        evaluation = state.get("evaluation", {})
        evidence = state.get("retrieved_evidence", [])
        evidence_block = "\n\n".join(f"[{e['citation_key']}] {e['text']}" for e in evidence)
        prompt = (
            f"Denial:\n{denial}\n\nCase:\n{case}\n\n"
            f"Criteria:\n{spec.get('criteria')}\n"
            f"Evaluation:\n{evaluation}\n\n"
            f"Evidence:\n{evidence_block}\n\n"
            "Explain, honestly and with cited evidence, why the denial is correct "
            "(which criteria are NOT met)."
        )
        out = invoke_structured("uphold", UpholdExplanation, prompt, mock=MOCK_UPHOLD)
        return {
            "output": out.model_dump(),
            "audit_trail": [entry("uphold", "uphold", unmet=out.unmet_criteria)],
        }

    return node
