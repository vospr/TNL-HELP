from __future__ import annotations

from concierge.state import ConciergeState
from concierge.trace import trace


FOLLOWUP_SUGGESTION = (
    "Based on your interest in Southeast Asia, you might also want to explore mountain retreats. "
    "Should I research that for you?"
)


class FollowUpAgent:
    def run(self, state: ConciergeState) -> ConciergeState:
        if str(state.get("route") or "") != "research":
            return {}
        if bool(state.get("human_handoff")):
            return {}
        if not bool(state.get("guardrail_passed")):
            return {}

        existing_response = str(state.get("current_response") or "").strip()
        if existing_response:
            combined = f"{existing_response} {FOLLOWUP_SUGGESTION}".strip()
        else:
            combined = FOLLOWUP_SUGGESTION

        trace("followup", event="suggestion_generated")
        return {
            "proactive_suggestion": FOLLOWUP_SUGGESTION,
            "current_response": combined,
        }
