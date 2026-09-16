"""Draft (appeal path): write the appeal letter, citing only retrieved evidence."""
from __future__ import annotations

from ..audit import entry
from ..llm import invoke_structured
from ..state import AppealLetter, GraphState

MOCK_APPEAL = {
    "letter": (
        "Dear Medical Review Board,\n\n"
        "We appeal the denial of the knee MRI. The patient meets every coverage "
        "criterion: (A) knee pain limits daily activities, (B) eight weeks of "
        "conservative care with physical therapy and anti-inflammatory medication, "
        "and (C) an X-ray shows joint space narrowing and a possible meniscal tear "
        "[POL-KNEE-MRI-1]. We respectfully request reconsideration."
    ),
    "citations": ["POL-KNEE-MRI-1"],
}


def make_draft_node():
    def node(state: GraphState) -> dict:
        denial = state.get("denial_text", "")
        case = state.get("case_text", "")
        spec = state.get("criteria_spec", {})
        evaluation = state.get("evaluation", {})
        evidence = state.get("retrieved_evidence", [])
        evidence_block = "\n\n".join(f"[{e['citation_key']}] {e['text']}" for e in evidence)
        prompt = (
            f"Denial:\n{denial}\n\nCase:\n{case}\n\n"
            f"Criteria (all MET):\n{spec.get('criteria')}\n"
            f"Evaluation:\n{evaluation}\n\n"
            f"Evidence (cite ONLY these keys):\n{evidence_block}\n\n"
            "Write the appeal letter. Cite only the provided keys."
        )
        letter = invoke_structured("draft", AppealLetter, prompt, mock=MOCK_APPEAL)
        return {
            "output": letter.model_dump(),
            "audit_trail": [entry("draft", "draft", citations=letter.citations)],
        }

    return node
