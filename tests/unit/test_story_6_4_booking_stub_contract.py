from __future__ import annotations

import pytest

from concierge.graph import compiled_graph
from concierge.state import initialize_state


def test_booking_agent_response_model_contract_fields() -> None:
    from concierge.agents.booking_agent import BookingAgentResponse

    fields = set(BookingAgentResponse.model_fields.keys())
    assert fields == {"status", "message", "integration_point", "required_env_vars"}


def test_booking_agent_returns_unavailable_contract_and_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from concierge.agents import booking_agent as booking_module

    trace_calls: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        booking_module,
        "trace",
        lambda node_name, outcome=None, **fields: trace_calls.append((node_name, dict(fields))),
    )

    state = initialize_state(
        "alex",
        "session-6-4-booking",
        "Can you book me a flight to Bali?",
        turn_id=1,
    )
    update = booking_module.BookingAgent().run(state)
    response = update["current_response"]

    assert response.status == "unavailable"
    assert response.message == (
        "Booking is not available in this version. "
        "A human travel specialist can assist - shall I connect you?"
    )
    assert response.integration_point == "Replace with BedrockBookingAPI(region=X, api_key=...)"
    assert response.required_env_vars == ["BOOKING_API_KEY", "BOOKING_REGION"]
    assert trace_calls == [
        ("booking_agent", {"event": "unavailable", "message": "stub_integration_contract"})
    ]


def test_booking_route_synthesis_includes_integration_contract_for_visibility() -> None:
    state = initialize_state(
        "alex",
        "session-6-4-graph",
        "Can you book me a flight to Bali?",
        turn_id=1,
    )
    state["route"] = "booking_stub"

    result = compiled_graph.invoke(state)

    assert "Booking is not available in this version." in result["current_response"]
    assert "Integration point: Replace with BedrockBookingAPI(region=X, api_key=...)." in result[
        "current_response"
    ]
    assert "Required env vars: BOOKING_API_KEY, BOOKING_REGION." in result["current_response"]
    assert result.get("_executed_nodes") == [
        "dispatcher",
        "booking_stub",
        "guardrail",
        "synthesis",
    ]
