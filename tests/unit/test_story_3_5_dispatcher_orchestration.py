from __future__ import annotations

from concierge.agents.dispatcher import DispatcherAgent
from concierge.state import initialize_state


def test_run_resets_turn_state_appends_history_and_routes_stage1(monkeypatch) -> None:
    def _stage1(self: DispatcherAgent, current_input: str) -> tuple[str | None, float, str | None]:
        assert current_input == "I want to book a flight"
        return "booking_intent", 0.95, "booking_stub"

    def _stage2(self: DispatcherAgent, current_input: str) -> tuple[str | None, float | None, str | None]:
        del self, current_input
        raise AssertionError("Stage 2 must not run when Stage 1 resolves route")

    traces: list[tuple[str, dict[str, object]]] = []

    def _capture_trace(node_name: str, outcome: str | None = None, **fields: object) -> None:
        del outcome
        traces.append((node_name, fields))

    monkeypatch.setattr(DispatcherAgent, "_evaluate_stage1", _stage1)
    monkeypatch.setattr(DispatcherAgent, "_evaluate_stage2", _stage2)
    monkeypatch.setattr("concierge.agents.dispatcher.trace", _capture_trace)

    state = initialize_state(
        user_id="alex",
        session_id="session-3-5-stage1",
        current_input="I want to book a flight",
        turn_id=1,
    )
    state["route"] = None
    state["rag_results"] = [{"id": "stale"}]
    state["research_results"] = [{"id": "stale-web"}]
    state["source_attribution"] = ["[RAG] stale"]
    state["clarification_needed"] = True
    state["clarification_question"] = "stale?"
    state["human_handoff"] = True
    state["error"] = "stale error"
    state["conversation_history"] = [{"role": "assistant", "content": "hello"}]

    update = DispatcherAgent().run(state)

    assert update["intent"] == "booking_intent"
    assert update["confidence"] == 0.95
    assert update["route"] == "booking_stub"
    assert update["current_response"] is None
    assert update["rag_results"] is None
    assert update["research_results"] is None
    assert update["source_attribution"] == []
    assert update["clarification_needed"] is False
    assert update["clarification_question"] is None
    assert update["human_handoff"] is False
    assert update["error"] is None
    assert update["conversation_history"] == [
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "I want to book a flight"},
    ]
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


def test_run_escalates_to_stage2_when_stage1_does_not_route(monkeypatch) -> None:
    call_order: list[str] = []

    def _stage1(self: DispatcherAgent, current_input: str) -> tuple[str | None, float, str | None]:
        del self
        assert current_input == "something else"
        call_order.append("stage1")
        return None, 0.0, None

    def _stage2(self: DispatcherAgent, current_input: str) -> tuple[str | None, float | None, str | None]:
        del self
        assert current_input == "something else"
        call_order.append("stage2")
        return "destination_research", 0.84, "research"

    traces: list[tuple[str, dict[str, object]]] = []

    def _capture_trace(node_name: str, outcome: str | None = None, **fields: object) -> None:
        del outcome
        traces.append((node_name, fields))

    monkeypatch.setattr(DispatcherAgent, "_evaluate_stage1", _stage1)
    monkeypatch.setattr(DispatcherAgent, "_evaluate_stage2", _stage2)
    monkeypatch.setattr("concierge.agents.dispatcher.trace", _capture_trace)

    state = initialize_state(
        user_id="alex",
        session_id="session-3-5-stage2",
        current_input="something else",
        turn_id=2,
    )
    state["route"] = None

    update = DispatcherAgent().run(state)

    assert call_order == ["stage1", "stage2"]
    assert update["intent"] == "destination_research"
    assert update["confidence"] == 0.84
    assert update["route"] == "research"
    assert update["current_response"] is None
    token_budget_traces = [fields for node_name, fields in traces if node_name == "token_budget"]
    dispatcher_traces = [fields for node_name, fields in traces if node_name == "dispatcher"]

    assert len(token_budget_traces) == 1
    assert token_budget_traces[0]["event"] == "check_performed"
    assert token_budget_traces[0]["threshold"] == 6000

    assert len(dispatcher_traces) == 1
    assert dispatcher_traces[0]["stage"] == "llm_escalation"


def test_dispatcher_owns_full_conversation_history_across_turns(monkeypatch) -> None:
    def _stage1(self: DispatcherAgent, current_input: str) -> tuple[str | None, float, str | None]:
        del self
        return "booking_intent", 0.95, "booking_stub"

    monkeypatch.setattr(DispatcherAgent, "_evaluate_stage1", _stage1)
    monkeypatch.setattr(
        DispatcherAgent,
        "_evaluate_stage2",
        lambda self, current_input: (None, None, None),
    )
    monkeypatch.setattr("concierge.agents.dispatcher.trace", lambda *args, **kwargs: None)

    turn_1 = initialize_state(
        user_id="alex",
        session_id="session-3-5-history",
        current_input="turn one",
        turn_id=1,
    )
    turn_1["route"] = None
    turn_1["conversation_history"] = [{"role": "assistant", "content": "welcome"}]

    update_1 = DispatcherAgent().run(turn_1)

    turn_2 = dict(turn_1)
    turn_2.update(update_1)
    turn_2["current_input"] = "turn two"
    turn_2["route"] = None

    update_2 = DispatcherAgent().run(turn_2)

    assert update_2["conversation_history"] == [
        {"role": "assistant", "content": "welcome"},
        {"role": "user", "content": "turn one"},
        {"role": "user", "content": "turn two"},
    ]
