"""LangGraph wiring for the honest-triage architecture.

research -> evaluator -> decision gate -> [draft | uphold | request] -> verify
verify loops back to evaluator once if it catches a judge error.
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .agents.draft import make_draft_node
from .agents.evaluator import make_evaluator_node
from .agents.request import make_request_node
from .agents.research import make_research_node
from .agents.uphold import make_uphold_node
from .agents.verify import make_verify_node
from .config import settings
from .retrieval.policy_store import HybridRetriever
from .state import GraphState

_DECISION_TO_NODE = {"appeal": "draft", "uphold": "uphold", "more_info": "request"}


def route_decision(state: GraphState) -> str:
    return _DECISION_TO_NODE.get(state.get("decision", "appeal"), "draft")


def route_after_verify(state: GraphState) -> str:
    verdict = state.get("verdict", {})
    cycle = state.get("cycle_count", 0)
    if verdict.get("verdict") == "pass":
        return END
    if verdict.get("reason") == "judge_error" and cycle < settings.max_cycles:
        return "evaluator"  # double-check caught a judge error -> re-evaluate once
    return END  # grounding error or cycle cap -> surface to human


def build_graph(retriever=None):
    retriever = retriever or HybridRetriever()
    builder = StateGraph(GraphState)
    builder.add_node("research", make_research_node(retriever))
    builder.add_node("evaluator", make_evaluator_node())
    builder.add_node("draft", make_draft_node())
    builder.add_node("uphold", make_uphold_node())
    builder.add_node("request", make_request_node())
    builder.add_node("verify", make_verify_node())

    builder.add_edge(START, "research")
    builder.add_edge("research", "evaluator")
    builder.add_conditional_edges(
        "evaluator",
        route_decision,
        {"draft": "draft", "uphold": "uphold", "request": "request"},
    )
    builder.add_edge("draft", "verify")
    builder.add_edge("uphold", "verify")
    builder.add_edge("request", "verify")
    builder.add_conditional_edges("verify", route_after_verify, {"evaluator": "evaluator", END: END})
    return builder.compile()
