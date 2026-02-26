from __future__ import annotations

import time

import pytest

from concierge.agents.dispatcher import DispatcherAgent
from concierge.state import initialize_state


def _patch_direct_rag_route(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        DispatcherAgent,
        "_evaluate_stage1",
        lambda self, current_input: ("property_lookup", 0.95, "rag"),
    )
    monkeypatch.setattr(
        DispatcherAgent,
        "_evaluate_stage2",
        lambda self, current_input: (None, None, None),
    )
    monkeypatch.setattr("concierge.agents.dispatcher.trace", lambda *args, **kwargs: None)


def test_dispatcher_resets_turn_scoped_subagent_fields_from_previous_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_direct_rag_route(monkeypatch)

    state = initialize_state("alex", "session-3-7-reset", "show me bali options", turn_id=2)
    state["route"] = None
    state["rag_results"] = [{"id": "stale-rag"}]
    state["research_results"] = [{"id": "stale-web"}]
    state["source_attribution"] = ["[RAG] stale"]
    state["proactive_suggestion"] = "stale suggestion"
    state["clarification_needed"] = True
    state["clarification_question"] = "stale question"
    state["human_handoff"] = True
    state["error"] = "stale error"

    update = DispatcherAgent().run(state)

    assert update["route"] == "rag"
    assert update["rag_results"] is None
    assert update["research_results"] is None
    assert update["source_attribution"] == []
    assert update["proactive_suggestion"] is None
    assert update["clarification_needed"] is False
    assert update["clarification_question"] is None
    assert update["human_handoff"] is False
    assert update["error"] is None


def test_rag_node_receives_fresh_rag_results_on_turn2_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from concierge.agents.rag_agent import RAGAgent
    from concierge.graph import compiled_graph

    _patch_direct_rag_route(monkeypatch)

    observed: dict[str, object] = {}

    def _capture_rag_input(self, state):  # noqa: ANN001
        del self
        observed["rag_results"] = state.get("rag_results")
        return {}

    monkeypatch.setattr(RAGAgent, "run", _capture_rag_input)

    state = initialize_state("alex", "session-3-7-rag-fresh", "wdh phuket", turn_id=2)
    state["route"] = None
    state["rag_results"] = [{"id": "turn-1-stale"}]

    result = compiled_graph.invoke(state)

    assert observed["rag_results"] is None
    assert result.get("_executed_nodes") == [
        "dispatcher",
        "rag",
        "guardrail",
        "synthesis",
    ]


def test_dispatcher_appends_current_message_before_specialist_routing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from concierge.agents.rag_agent import RAGAgent
    from concierge.graph import compiled_graph

    _patch_direct_rag_route(monkeypatch)

    observed: dict[str, object] = {}

    def _capture_rag_input(self, state):  # noqa: ANN001
        del self
        observed["history"] = list(state.get("conversation_history") or [])
        return {}

    monkeypatch.setattr(RAGAgent, "run", _capture_rag_input)

    prior_history = [
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "turn-1"},
    ]
    state = initialize_state("alex", "session-3-7-history", "turn-2", turn_id=2)
    state["route"] = None
    state["conversation_history"] = list(prior_history)

    compiled_graph.invoke(state)

    assert observed["history"] == [
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "turn-1"},
        {"role": "user", "content": "turn-2"},
    ]


def test_dispatch_overhead_stays_under_2_seconds_without_llm_latency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_direct_rag_route(monkeypatch)

    state = initialize_state("alex", "session-3-7-performance", "quick route", turn_id=2)
    state["route"] = None

    agent = DispatcherAgent()
    durations: list[float] = []
    for _ in range(100):
        started = time.perf_counter()
        agent.run(state)
        durations.append(time.perf_counter() - started)

    assert max(durations) < 2.0
    assert sum(durations) / len(durations) < 0.1
