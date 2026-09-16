"""More-info path: request the specific missing document/information."""
from __future__ import annotations

from ..audit import entry
from ..llm import invoke_structured
from ..state import DocumentRequest, GraphState

MOCK_REQUEST = {
    "request": (
        "Please provide documentation of conservative treatment for the right "
        "knee (e.g. physical therapy notes or prescribed anti-inflammatory "
        "medication). This is needed to verify coverage criterion B."
    ),
    "missing_criteria": ["B"],
    "citations": [],
}


def make_request_node():
    def node(state: GraphState) -> dict:
        denial = state.get("denial_text", "")
        case = state.get("case_text", "")
        spec = state.get("criteria_spec", {})
        evaluation = state.get("evaluation", {})
        prompt = (
            f"Denial:\n{denial}\n\nCase:\n{case}\n\n"
            f"Criteria:\n{spec.get('criteria')}\n"
            f"Evaluation:\n{evaluation}\n\n"
            "Which information/document is missing? Write a clear request naming "
            "exactly what the hospital should provide."
        )
        out = invoke_structured("request", DocumentRequest, prompt, mock=MOCK_REQUEST)
        return {
            "output": out.model_dump(),
            "audit_trail": [entry("request", "request", missing=out.missing_criteria)],
        }

    return node
