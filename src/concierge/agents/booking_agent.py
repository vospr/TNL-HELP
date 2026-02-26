from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from concierge.state import ConciergeState
from concierge.trace import trace


class BookingAgentResponse(BaseModel):
    status: str
    message: str
    integration_point: str
    required_env_vars: list[str]


class BookingAgent:
    def run(self, state: ConciergeState) -> ConciergeState:
        del state
        trace("booking_agent", event="unavailable", message="stub_integration_contract")
        response = BookingAgentResponse(
            status="unavailable",
            message=(
                "Booking is not available in this version. "
                "A human travel specialist can assist - shall I connect you?"
            ),
            integration_point="Replace with BedrockBookingAPI(region=X, api_key=...)",
            required_env_vars=["BOOKING_API_KEY", "BOOKING_REGION"],
        )
        return {"current_response": response}
