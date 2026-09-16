"""Verify: ground the output AND double-check the evaluator's judgment.

Two checks:
  1. Grounding (code)  — every cited key must exist in the evidence (no hallucination).
  2. Judge recheck (LLM) — re-validate each criterion's MET / NOT MET / UNVERIFIED.
"""
from __future__ import annotations

from ..audit import entry
from ..config import settings
from ..llm import invoke_structured
from ..state import GraphState, Rechecks


def make_verify_node():
    def node(state: GraphState) -> dict:
        output = state.get("output", {})
        evaluation = state.get("evaluation", {})
        evidence = state.get("retrieved_evidence", [])
        case = state.get("case_text", "")

        # 1. Grounding guard (code): citations must exist in evidence.
        evidence_keys = {e["citation_key"] for e in evidence}
        cited = set(output.get("citations", []))
        hallucinated = sorted(cited - evidence_keys)

        # 2. Double-check the judge.
        if settings.llm_provider == "mock":
            # The mock judge is deterministic; the recheck simply agrees.
            rechecks = {"checks": [
                {"criterion_id": v["criterion_id"], "original_status": v["status"],
                 "correct": True, "corrected_status": None,
                 "reasoning": f"mock recheck agrees with {v['status']}"}
                for v in evaluation.get("criteria", [])
            ]}
        else:
            checks_prompt = (
                f"Case record:\n{case}\n\n"
                f"Evaluator's per-criterion judgments:\n{evaluation.get('criteria')}\n\n"
                f"Evidence:\n" +
                "\n".join(f"[{e['citation_key']}] {e['text'][:150]}" for e in evidence) + "\n\n"
                "For EACH criterion, re-check whether the evaluator's status is correct. "
                "If wrong, set correct=False and give the corrected_status."
            )
            rechecks = invoke_structured("verify", Rechecks, checks_prompt).model_dump()

        wrong = [r for r in rechecks["checks"] if not r["correct"]]

        if hallucinated:
            verdict_dict = {
                "verdict": "fail",
                "confidence": 0.2,
                "feedback": f"Hallucinated citations not in evidence: {hallucinated}",
                "reason": "grounding_error",
                "citation_checks": [{"citation_key": c, "supported": False, "reason": "not in evidence"}
                                   for c in hallucinated],
                "judge_rechecks": rechecks["checks"],
            }
        elif wrong:
            verdict_dict = {
                "verdict": "fail",
                "confidence": 0.3,
                "feedback": "The evaluator mis-judged: " + "; ".join(
                    f"{r['criterion_id']} was {r['original_status']}, should be {r['corrected_status']}"
                    for r in wrong
                ),
                "reason": "judge_error",
                "citation_checks": [],
                "judge_rechecks": rechecks["checks"],
            }
        else:
            verdict_dict = {
                "verdict": "pass",
                "confidence": 0.9,
                "feedback": "Output is grounded and the evaluator's judgments were confirmed.",
                "reason": "ready",
                "citation_checks": [{"citation_key": c, "supported": True, "reason": "in evidence"}
                                   for c in cited],
                "judge_rechecks": rechecks["checks"],
            }

        return {
            "verdict": verdict_dict,
            "confidence_score": verdict_dict["confidence"],
            "cycle_count": state.get("cycle_count", 0) + 1,
            "audit_trail": [entry("verify", "verify", verdict=verdict_dict["verdict"],
                                   reason=verdict_dict["reason"], n_wrong=len(wrong))],
        }

    return node
