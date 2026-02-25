"""Node seam: research node delegates graph call to research agent stub."""

from typing import Any

from concierge.agents.research_agent import ResearchAgent
from concierge.nodes.error_handling import build_node_error_update
from concierge.state import ConciergeState
from concierge.trace import trace


def research_node(state: ConciergeState) -> dict[str, Any]:
    trace("research_agent", turn_id=state.get("turn_id"), status="start")
    try:
        return ResearchAgent().run(state)
    except Exception as exc:
        return build_node_error_update(exc)
