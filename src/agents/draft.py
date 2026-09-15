"""Draft agent: write the appeal letter, citing only retrieved evidence keys."""
from __future__ import annotations

from ..audit import entry
from ..llm import invoke_structured
from ..state import GraphState, LetterDraft

MOCK_LETTER = {
    "letter": (
        "Dear Medical Review Board,\n\n"
        "We appeal the denial of the MRI of the right knee (CPT 73721). "
        "The patient meets all coverage criteria: (1) knee pain limits daily activities, "
        "(2) the patient completed eight weeks of conservative care with physical therapy "
        "and anti-inflammatory medication without improvement, and (3) an X-ray shows "
        "medial joint space narrowing and a possible meniscal tear [POL-KNEE-MRI-1].\n\n"
        "We respectfully request reconsideration of coverage."
    ),
    "citations": ["POL-KNEE-MRI-1"],
    "codes": {"icd10": ["M23.31"], "cpt": ["73721"], "notes": "medial meniscal tear"},
}


def make_draft_node():
    def node(state: GraphState) -> dict:
        denial = state.get("denial_text", "")
        case = state.get("case_text", "")
        evidence = state.get("retrieved_evidence", [])
        evidence_block = "\n\n".join(f"[{e['citation_key']}] {e['text']}" for e in evidence)
        prompt = (
            f"Denial:\n{denial}\n\nCase:\n{case}\n\n"
            f"Evidence (cite ONLY these keys; ignore irrelevant passages):\n{evidence_block}\n\n"
            "Write the appeal letter."
        )
        letter = invoke_structured("draft", LetterDraft, prompt, mock=MOCK_LETTER)
        return {
            "letter_draft": letter.model_dump(),
            "codes": letter.codes.model_dump(),
            "audit_trail": [entry("draft", "draft", citations=letter.citations)],
        }

    return node
