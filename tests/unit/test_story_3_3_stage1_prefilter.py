from __future__ import annotations

from concierge.agents.dispatcher import DispatcherAgent
from concierge.state import initialize_state


def test_stage1_rules_load_on_dispatcher_startup() -> None:
    agent = DispatcherAgent()

    assert isinstance(agent.escalation_threshold, float)
    assert 0.0 <= agent.escalation_threshold <= 1.0
    assert agent.routing_rules, "Stage 1 rules should be loaded at startup"

    for rule in agent.routing_rules:
        assert rule.intent
        assert rule.route
        assert isinstance(rule.confidence, float)
        assert 0.0 <= rule.confidence <= 1.0


def test_stage1_booking_keyword_routes_without_escalation(monkeypatch) -> None:
    traces: list[tuple[str, dict[str, object]]] = []

    def _capture_trace(node_name: str, outcome: str | None = None, **fields: object) -> None:
        del outcome
        traces.append((node_name, fields))

    monkeypatch.setattr("concierge.agents.dispatcher.trace", _capture_trace)

    state = initialize_state(
        user_id="alex",
        session_id="session-3-3-booking",
        current_input="I want to book a flight",
        turn_id=1,
    )
    state["route"] = None

    update = DispatcherAgent().run(state)

    assert update["intent"] == "booking_intent"
    assert update["confidence"] == 0.95
    assert update["route"] == "booking_stub"
    assert traces == [
        (
            "dispatcher",
            {
                "intent": "booking_intent",
                "confidence": 0.95,
                "route": "booking_stub",
                "stage": "pre_filter",
            },
        )
    ]


def test_stage1_no_keyword_match_escalates_with_zero_confidence() -> None:
    state = initialize_state(
        user_id="alex",
        session_id="session-3-3-escalate",
        current_input="something else",
        turn_id=2,
    )
    state["route"] = None

    update = DispatcherAgent().run(state)

    assert update["intent"] is None
    assert update["confidence"] == 0.0
    assert update["route"] is None
