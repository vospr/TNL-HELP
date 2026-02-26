from __future__ import annotations

import pytest

from concierge.agents.guardrail import GuardrailAgent
from concierge.graph import compiled_graph
from concierge.state import initialize_state


def test_guardrail_out_of_domain_deflection_for_weather_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from concierge.agents import guardrail as guardrail_module

    trace_calls: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        guardrail_module,
        "trace",
        lambda node_name, outcome=None, **fields: trace_calls.append((node_name, dict(fields))),
    )

    state = initialize_state("alex", "session-6-2-weather", "What's the weather in London?", turn_id=1)
    state["intent"] = "out_of_domain"
    state["confidence"] = 0.32

    update = GuardrailAgent().run(state)

    assert update["guardrail_passed"] is False
    assert update["clarification_needed"] is False
    assert update["clarification_question"] is None
    assert update["human_handoff"] is False
    assert "I specialize in travel planning and concierge services." in update["current_response"]
    assert "best time of year to visit London" in update["current_response"]
    assert trace_calls == [
        (
            "guardrail",
            {"event": "out_of_domain", "confidence": 0.32, "query": "weather london"},
        )
    ]


def test_graph_returns_deflection_without_ending_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from concierge.agents import guardrail as guardrail_module

    monkeypatch.setattr(guardrail_module, "trace", lambda *args, **kwargs: None)

    state = initialize_state("alex", "session-6-2-graph", "What's the weather in London?", turn_id=1)
    state["route"] = "fallback"
    state["intent"] = "out_of_domain"
    state["confidence"] = 0.32

    result = compiled_graph.invoke(state)

    assert result["guardrail_passed"] is False
    assert result["human_handoff"] is False
    assert "travel planning and concierge services" in result["current_response"]
    assert result.get("_executed_nodes") == [
        "dispatcher",
        "guardrail",
        "synthesis",
    ]
