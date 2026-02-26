from __future__ import annotations

import pytest

from concierge.agents.response_synthesis import ResponseSynthesisAgent
from concierge.graph import compiled_graph
from concierge.state import initialize_state


LABEL = "[WEB SEARCH UNAVAILABLE — serving from internal KB only]"


def test_research_agent_timeout_degrades_with_label_and_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from concierge.agents import research_agent as research_agent_module

    trace_calls: list[tuple[str, dict[str, object]]] = []

    def _raise_timeout(query: str, max_results: int = 5):  # noqa: ANN001
        del query, max_results
        raise TimeoutError("request timed out")

    monkeypatch.setattr(research_agent_module, "search_duckduckgo", _raise_timeout)
    monkeypatch.setattr(
        research_agent_module,
        "trace",
        lambda node_name, outcome=None, **fields: trace_calls.append((node_name, dict(fields))),
    )

    update = research_agent_module.ResearchAgent().run(
        initialize_state("alex", "session-5-4-timeout", "latest travel trends", turn_id=1)
    )

    assert update["research_results"] == []
    assert update["degradation_label"] == LABEL
    assert update["current_response"] == LABEL
    assert trace_calls == [
        (
            "research_agent",
            {"event": "search_unavailable", "reason": "timeout"},
        )
    ]


def test_research_agent_rate_limit_degrades_without_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from concierge.agents import research_agent as research_agent_module

    trace_calls: list[tuple[str, dict[str, object]]] = []

    def _raise_rate_limit(query: str, max_results: int = 5):  # noqa: ANN001
        del query, max_results
        raise RuntimeError("rate limit reached")

    monkeypatch.setattr(research_agent_module, "search_duckduckgo", _raise_rate_limit)
    monkeypatch.setattr(
        research_agent_module,
        "trace",
        lambda node_name, outcome=None, **fields: trace_calls.append((node_name, dict(fields))),
    )

    update = research_agent_module.ResearchAgent().run(
        initialize_state("alex", "session-5-4-rate-limit", "latest travel trends", turn_id=1)
    )

    assert update["research_results"] == []
    assert update["degradation_label"] == LABEL
    assert trace_calls == [
        (
            "research_agent",
            {"event": "search_unavailable", "reason": "rate_limit"},
        )
    ]


def test_graph_research_route_does_not_crash_on_search_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from concierge.agents import research_agent as research_agent_module

    def _raise_timeout(query: str, max_results: int = 5):  # noqa: ANN001
        del query, max_results
        raise TimeoutError("request timed out")

    monkeypatch.setattr(research_agent_module, "search_duckduckgo", _raise_timeout)
    state = initialize_state("alex", "session-5-4-graph", "latest travel trends", turn_id=1)
    state["route"] = "research"
    state["rag_results"] = [{"id": "dest-bali", "name": "Bali"}]

    result = compiled_graph.invoke(state)

    assert isinstance(result, dict)
    assert result.get("_executed_nodes") == [
        "dispatcher",
        "research",
        "guardrail",
        "synthesis",
        "followup",
    ]


def test_response_synthesis_includes_degradation_label_for_kb_only_path() -> None:
    state = initialize_state("alex", "session-5-4-synthesis", "any", turn_id=1)
    state["rag_results"] = [{"id": "dest-bali", "name": "Bali"}]
    state["research_results"] = []
    state["degradation_label"] = LABEL

    update = ResponseSynthesisAgent().run(state)

    assert update["current_response"].startswith(LABEL)
    assert "[RAG]" in update["current_response"]
