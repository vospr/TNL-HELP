from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from concierge.state import initialize_state


def test_research_agent_calls_ddg_with_query_parses_results_and_traces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from concierge.agents import research_agent as research_agent_module

    captured: dict[str, object] = {}
    trace_calls: list[tuple[str, dict[str, object]]] = []

    def _fake_search(query: str, max_results: int = 5) -> list[dict[str, str]]:
        captured["query"] = query
        captured["max_results"] = max_results
        return [
            {"title": "Trend 1", "href": "https://example.com/1", "body": "Snippet 1"},
            {"title": "Trend 2", "href": "https://example.com/2", "body": "Snippet 2"},
            {"title": "Trend 3", "href": "https://example.com/3", "body": "Snippet 3"},
        ]

    monkeypatch.setattr(research_agent_module, "search_duckduckgo", _fake_search)
    monkeypatch.setattr(
        research_agent_module,
        "trace",
        lambda node_name, outcome=None, **fields: trace_calls.append((node_name, dict(fields))),
    )

    state = initialize_state(
        user_id="alex",
        session_id="session-5-3-basic",
        current_input="latest travel trends to Southeast Asia",
        turn_id=1,
    )
    update = research_agent_module.ResearchAgent().run(state)

    assert captured["query"] == "latest travel trends to Southeast Asia"
    assert captured["max_results"] == 5
    assert update["research_results"] == [
        {"title": "Trend 1", "link": "https://example.com/1", "snippet": "Snippet 1"},
        {"title": "Trend 2", "link": "https://example.com/2", "snippet": "Snippet 2"},
        {"title": "Trend 3", "link": "https://example.com/3", "snippet": "Snippet 3"},
    ]
    assert trace_calls == [
        ("research_agent", {"event": "search_complete", "results_count": 3})
    ]


def test_research_agent_scopes_context_to_last_three_turns_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from concierge.agents import research_agent as research_agent_module

    captured_query: dict[str, str] = {}

    def _fake_search(query: str, max_results: int = 5) -> list[dict[str, str]]:
        del max_results
        captured_query["value"] = query
        return []

    monkeypatch.setattr(research_agent_module, "search_duckduckgo", _fake_search)

    state = initialize_state(
        user_id="alex",
        session_id="session-5-3-scope",
        current_input="latest travel trends to Southeast Asia",
        turn_id=4,
    )
    state["conversation_history"] = [
        {"role": "user", "content": "old-turn-1"},
        {"role": "assistant", "content": "old-turn-2"},
        {"role": "user", "content": "keep-turn-3"},
        {"role": "assistant", "content": "keep-turn-4"},
        {"role": "user", "content": "keep-turn-5"},
    ]

    research_agent_module.ResearchAgent().run(state)
    scoped_query = captured_query["value"]

    assert "latest travel trends to Southeast Asia" in scoped_query
    assert "keep-turn-3" in scoped_query
    assert "keep-turn-4" in scoped_query
    assert "keep-turn-5" in scoped_query
    assert "old-turn-1" not in scoped_query
    assert "old-turn-2" not in scoped_query


def test_research_agent_results_count_follows_search_payload_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from concierge.agents import research_agent as research_agent_module

    monkeypatch.setattr(
        research_agent_module,
        "search_duckduckgo",
        lambda query, max_results=5: [
            {"title": "One", "href": "https://example.com/1", "body": "A"},
            {"title": "Two", "href": "https://example.com/2", "body": "B"},
            {"title": "Three", "href": "https://example.com/3", "body": "C"},
            {"title": "Four", "href": "https://example.com/4", "body": "D"},
        ],
    )

    update_many = research_agent_module.ResearchAgent().run(
        initialize_state("alex", "session-5-3-many", "travel trends", turn_id=1)
    )
    assert len(update_many["research_results"]) >= 3

    monkeypatch.setattr(
        research_agent_module,
        "search_duckduckgo",
        lambda query, max_results=5: [
            {"title": "One", "href": "https://example.com/1", "body": "A"},
            {"title": "Two", "href": "https://example.com/2", "body": "B"},
        ],
    )
    update_few = research_agent_module.ResearchAgent().run(
        initialize_state("alex", "session-5-3-few", "travel trends", turn_id=1)
    )
    assert len(update_few["research_results"]) == 2


def test_research_agent_llm_ranking_enforces_policy_max_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from concierge.agents import research_agent as research_agent_module

    monkeypatch.delenv("FAST_MODE", raising=False)
    monkeypatch.setenv("RESEARCH_AGENT_LLM_RANKING", "1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(
        research_agent_module,
        "search_duckduckgo",
        lambda query, max_results=5: [
            {"title": "One", "href": "https://example.com/1", "body": "A"},
            {"title": "Two", "href": "https://example.com/2", "body": "B"},
            {"title": "Three", "href": "https://example.com/3", "body": "C"},
        ],
    )

    captured_kwargs: dict[str, object] = {}

    class _FakeMessages:
        def create(self, **kwargs: object) -> object:
            captured_kwargs.update(kwargs)
            return SimpleNamespace(content=[SimpleNamespace(text="[2, 0, 1]")])

    class _FakeAnthropicClient:
        def __init__(self) -> None:
            self.messages = _FakeMessages()

    monkeypatch.setitem(
        sys.modules,
        "anthropic",
        SimpleNamespace(Anthropic=lambda: _FakeAnthropicClient()),
    )

    update = research_agent_module.ResearchAgent().run(
        initialize_state("alex", "session-5-3-ranking", "travel trends", turn_id=1)
    )

    assert captured_kwargs["max_tokens"] == 1024
    assert captured_kwargs["model"] == "claude-sonnet-4-6"
    assert update["research_results"][0]["title"] == "Three"
