from __future__ import annotations

import pytest

from concierge.agents.guardrail import CLARIFICATION_QUESTION, GuardrailAgent
from concierge.graph import compiled_graph
from concierge.state import initialize_state


HANDOFF_MESSAGE = (
    "I'm having trouble understanding your request. "
    "A human travel specialist can assist - please wait or contact support."
)


def test_guardrail_triggers_handoff_after_max_clarification_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from concierge.agents import guardrail as guardrail_module

    trace_calls: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        guardrail_module,
        "trace",
        lambda node_name, outcome=None, **fields: trace_calls.append((node_name, dict(fields))),
    )

    state = initialize_state("alex", "session-6-3-max", "still not clear", turn_id=4)
    state["confidence"] = 0.42
    state["conversation_history"] = _history_with_clarifications(3)

    update = GuardrailAgent().run(state)

    assert update["human_handoff"] is True
    assert update["guardrail_passed"] is False
    assert update["clarification_needed"] is False
    assert update["clarification_question"] is None
    assert update["current_response"] == HANDOFF_MESSAGE
    assert update["max_clarifications"] == 3
    assert trace_calls == [
        (
            "guardrail",
            {
                "event": "human_handoff",
                "clarification_count": 3,
                "session_id": "session-6-3-max",
            },
        )
    ]


def test_guardrail_asks_clarification_when_below_max_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from concierge.agents import guardrail as guardrail_module

    monkeypatch.setattr(guardrail_module, "trace", lambda *args, **kwargs: None)

    state = initialize_state("alex", "session-6-3-below", "still unclear", turn_id=3)
    state["confidence"] = 0.42
    state["conversation_history"] = _history_with_clarifications(2)

    update = GuardrailAgent().run(state)

    assert update["human_handoff"] is False
    assert update["clarification_needed"] is True
    assert update["clarification_question"] == CLARIFICATION_QUESTION
    assert update["max_clarifications"] == 3


def test_graph_emits_terminal_handoff_message_without_crashing() -> None:
    state = initialize_state("alex", "session-6-3-graph", "still unclear", turn_id=5)
    state["route"] = "fallback"
    state["confidence"] = 0.42
    state["conversation_history"] = _history_with_clarifications(3)

    result = compiled_graph.invoke(state)

    assert result["human_handoff"] is True
    assert result["current_response"].endswith(HANDOFF_MESSAGE)
    assert result.get("_executed_nodes") == [
        "dispatcher",
        "guardrail",
        "synthesis",
    ]


def _history_with_clarifications(count: int) -> list[dict[str, str]]:
    history: list[dict[str, str]] = []
    for idx in range(count):
        history.append({"role": "user", "content": f"Ambiguous request {idx + 1}"})
        history.append({"role": "assistant", "content": CLARIFICATION_QUESTION})
    return history
