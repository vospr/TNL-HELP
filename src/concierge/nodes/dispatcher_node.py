"""Node seam: dispatcher node delegates graph call to dispatcher agent stub."""

from typing import Any

from concierge.agents.dispatcher import DispatcherAgent
from concierge.nodes.error_handling import build_node_error_update
from concierge.state import ConciergeState
from concierge.trace import trace


def dispatcher_node(state: ConciergeState) -> dict[str, Any]:
    trace("dispatcher", turn_id=state.get("turn_id"), status="start")
    try:
        return DispatcherAgent().run(state)
    except Exception as exc:
        return build_node_error_update(exc)
