"""Node seam: guardrail node delegates graph call to guardrail agent stub."""

from typing import Any

from concierge.agents.guardrail import GuardrailAgent
from concierge.nodes.error_handling import build_node_error_update
from concierge.state import ConciergeState
from concierge.trace import trace


def guardrail_node(state: ConciergeState) -> dict[str, Any]:
    trace("guardrail", turn_id=state.get("turn_id"), status="start")
    try:
        return GuardrailAgent().run(state)
    except Exception as exc:
        return build_node_error_update(exc)
