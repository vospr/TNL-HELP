"""Node seam: RAG node delegates graph call to RAG agent stub."""

from typing import Any

from concierge.agents.rag_agent import RAGAgent
from concierge.nodes.error_handling import build_node_error_update
from concierge.state import ConciergeState
from concierge.trace import trace


def rag_node(state: ConciergeState) -> dict[str, Any]:
    trace("rag_agent", turn_id=state.get("turn_id"), status="start")
    try:
        return RAGAgent().run(state)
    except Exception as exc:
        return build_node_error_update(exc)
