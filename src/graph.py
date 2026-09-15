"""LangGraph wiring: research -> draft -> verify (loop) -> end. No interrupt."""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .agents.draft import make_draft_node
from .agents.research import make_research_node
from .agents.verify import make_verify_node
from .config import settings
from .retrieval.policy_store import HybridRetriever
from .state import GraphState


def route_after_verify(state: GraphState) -> str:
    verdict = state.get("verdict", {})
    cycle = state.get("cycle_count", 0)
    if verdict.get("verdict") == "pass":
        return END
    if cycle >= settings.max_cycles:
        return END
    if verdict.get("reason") == "evidence_insufficient":
        return "research"
    return "draft"


def build_graph(retriever=None):
    retriever = retriever or HybridRetriever()
    research = make_research_node(retriever)
    draft = make_draft_node()
    verify = make_verify_node()

    builder = StateGraph(GraphState)
    builder.add_node("research", research)
    builder.add_node("draft", draft)
    builder.add_node("verify", verify)
    builder.add_edge(START, "research")
    builder.add_edge("research", "draft")
    builder.add_edge("draft", "verify")
    builder.add_conditional_edges(
        "verify",
        route_after_verify,
        {"research": "research", "draft": "draft", END: END},
    )
    return builder.compile()
