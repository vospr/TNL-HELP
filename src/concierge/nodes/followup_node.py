"""Node seam: follow-up node delegates graph call to follow-up agent stub."""

from typing import Any

from concierge.agents.followup import FollowUpAgent
from concierge.nodes.error_handling import build_node_error_update
from concierge.state import ConciergeState
from concierge.trace import trace


def followup_node(state: ConciergeState) -> dict[str, Any]:
    trace("followup", turn_id=state.get("turn_id"), status="start")
    try:
        return FollowUpAgent().run(state)
    except Exception as exc:
        return build_node_error_update(exc)
