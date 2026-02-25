"""Node seam: synthesis node delegates graph call to response synthesis agent stub."""

from typing import Any

from concierge.agents.response_synthesis import ResponseSynthesisAgent
from concierge.nodes.error_handling import build_node_error_update
from concierge.state import ConciergeState
from concierge.trace import trace


def synthesis_node(state: ConciergeState) -> dict[str, Any]:
    trace("response_synthesis", turn_id=state.get("turn_id"), status="start")
    try:
        return ResponseSynthesisAgent().run(state)
    except Exception as exc:
        return build_node_error_update(exc)
