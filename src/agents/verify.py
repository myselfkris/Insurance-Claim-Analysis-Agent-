"""Verify agent: score the letter and check citation faithfulness."""
from __future__ import annotations

from ..audit import entry
from ..llm import invoke_structured
from ..state import GraphState, Verdict

MOCK_VERDICT = {
    "verdict": "pass",
    "confidence": 0.9,
    "feedback": "The letter cites the correct policy criterion and the clinical evidence supports it.",
    "reason": "ready",
    "citation_checks": [
        {"citation_key": "POL-KNEE-MRI-1", "supported": True, "reason": "criterion matches the clinical note"}
    ],
}


def make_verify_node():
    def node(state: GraphState) -> dict:
        letter = state.get("letter_draft", {})
        evidence = state.get("retrieved_evidence", [])
        evidence_block = "\n\n".join(f"[{e['citation_key']}] {e['text']}" for e in evidence)
        prompt = (
            f"Letter:\n{letter.get('letter', '')}\n\n"
            f"Evidence:\n{evidence_block}\n\n"
            "Score the letter and check that every citation is supported."
        )
        verdict = invoke_structured("verify", Verdict, prompt, mock=MOCK_VERDICT)
        verdict_dict = verdict.model_dump()

        # Structural grounding guard: every cited key must exist in evidence.
        evidence_keys = {e["citation_key"] for e in evidence}
        cited = set((state.get("letter_draft") or {}).get("citations", []))
        hallucinated = sorted(cited - evidence_keys)
        if hallucinated:
            verdict_dict["verdict"] = "fail"
            verdict_dict["reason"] = "writing_citation"
            verdict_dict["confidence"] = min(float(verdict_dict.get("confidence", 0.5)), 0.4)
            verdict_dict["feedback"] = f"Hallucinated citations not in evidence: {hallucinated}"

        return {
            "verdict": verdict_dict,
            "confidence_score": verdict_dict.get("confidence", 0.0),
            "cycle_count": state.get("cycle_count", 0) + 1,
            "audit_trail": [entry("verify", "verify", verdict=verdict_dict["verdict"], confidence=verdict_dict.get("confidence"))],
        }

    return node
