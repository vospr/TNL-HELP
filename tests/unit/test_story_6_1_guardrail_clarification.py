from __future__ import annotations

import pytest

from concierge.agents.dispatcher import DispatcherAgent
from concierge.agents.guardrail import GuardrailAgent
from concierge.state import initialize_state


CLARIFICATION_QUESTION = (
    "Of course - are you looking to research a destination, check on a booking, or something else?"
)


def test_guardrail_fires_single_clarification_when_confidence_below_dispatcher_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from concierge.agents import guardrail as guardrail_module

    trace_calls: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        guardrail_module,
        "trace",
        lambda node_name, outcome=None, **fields: trace_calls.append((node_name, dict(fields))),
    )

    state = initialize_state("alex", "session-6-1-low", "not sure", turn_id=1)
    state["confidence"] = 0.61

    update = GuardrailAgent().run(state)

    assert update["guardrail_passed"] is False
    assert update["clarification_needed"] is True
    assert update["clarification_question"] == CLARIFICATION_QUESTION
    assert update["clarification_question"].count("?") == 1
    assert trace_calls == [
        ("guardrail", {"event": "clarification_fired", "confidence": 0.61, "threshold": 0.75})
    ]


def test_guardrail_passes_without_clarification_when_confidence_meets_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from concierge.agents import guardrail as guardrail_module

    monkeypatch.setattr(guardrail_module, "trace", lambda *args, **kwargs: None)

    state = initialize_state("alex", "session-6-1-high", "beach ideas", turn_id=1)
    state["confidence"] = 0.84

    update = GuardrailAgent().run(state)

    assert update["guardrail_passed"] is True
    assert update["clarification_needed"] is False
    assert update["clarification_question"] is None


def test_dispatcher_appends_user_reply_on_next_turn_after_clarification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("concierge.agents.dispatcher.trace", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        DispatcherAgent,
        "_evaluate_stage1",
        lambda self, current_input: ("destination_research", 0.91, "research"),
    )

    state = initialize_state(
        "alex",
        "session-6-1-next-turn",
        "I want destination research",
        turn_id=2,
    )
    state["conversation_history"] = [
        {"role": "user", "content": "help"},
        {"role": "assistant", "content": CLARIFICATION_QUESTION},
    ]

    update = DispatcherAgent().run(state)

    assert update["conversation_history"][-1] == {
        "role": "user",
        "content": "I want destination research",
    }
    assert update["route"] == "research"
