"""Criterion Evaluator: an LLM that judges each criterion MET / NOT MET / UNVERIFIED.

The retriever finds facts; the evaluator reads those facts and decides what they
mean against each criterion. The NOT MET vs UNVERIFIED distinction needs reading,
not keyword matching.
"""
from __future__ import annotations

from ..audit import entry
from ..config import settings
from ..llm import invoke_structured
from ..logic import decide
from ..state import Evaluation, GraphState


def _mock_evaluate(spec: dict, case_text: str) -> dict:
    """Deterministic keyword mock — a placeholder for the real LLM judge."""
    text = case_text.lower()
    verdicts = []
    for c in spec.get("criteria", []):
        desc = c["description"].lower()
        if "conservative" in desc or "therap" in desc or "treatment" in desc:
            if "declined" in text or "refused" in text or "did not" in text:
                status = "not_met"
            elif "physical therapy" in text or "ibuprofen" in text or "weeks" in text:
                status = "met"
            else:
                status = "unverified"
        elif "x-ray" in desc or "imaging" in desc:
            status = "met" if "x-ray" in text else "unverified"
        elif "pain" in desc or "symptom" in desc:
            status = "met" if "pain" in text else "unverified"
        else:
            status = "unverified"
        verdicts.append({
            "criterion_id": c["id"],
            "status": status,
            "reasoning": f"mock keyword check for: {c['description']}",
            "citations": [],
        })
    return {"criteria": verdicts}


def make_evaluator_node():
    def node(state: GraphState) -> dict:
        spec = state.get("criteria_spec", {})
        case = state.get("case_text", "")
        evidence = state.get("retrieved_evidence", [])

        if settings.llm_provider == "mock":
            evaluation = _mock_evaluate(spec, case)
        else:
            evidence_block = "\n\n".join(f"[{e['citation_key']}] {e['text']}" for e in evidence)
            prompt = (
                f"Coverage criteria:\n{spec.get('criteria')}\n"
                f"Boolean expression: {spec.get('expression')}\n\n"
                f"Patient record:\n{case}\n\n"
                f"Retrieved facts:\n{evidence_block}\n\n"
                "For EACH criterion judge its status: met / not_met / unverified.\n"
                "  met       = the record has positive evidence for it\n"
                "  not_met   = the record has evidence AGAINST it (e.g. 'declined')\n"
                "  unverified= the record says nothing either way\n"
                "Give reasoning and cite supporting evidence keys."
            )
            evaluation = invoke_structured("evaluator", Evaluation, prompt).model_dump()

        decision = decide(spec.get("expression", ""), evaluation["criteria"])

        return {
            "evaluation": evaluation,
            "decision": decision,
            "audit_trail": [entry(
                "evaluator", "evaluator",
                decision=decision,
                statuses={v["criterion_id"]: v["status"] for v in evaluation["criteria"]},
            )],
        }

    return node
