from __future__ import annotations

import pytest

from concierge.agents.response_synthesis import ResponseSynthesisAgent
from concierge.state import initialize_state


def test_response_synthesis_excludes_system_summary_messages_from_filtered_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from concierge.agents import response_synthesis as module

    captured_history: list[dict[str, str]] = []

    def _fake_build(self, rag_results, research_results, filtered_history):  # noqa: ANN001
        del self, rag_results, research_results
        captured_history.extend(filtered_history)
        return "filtered output"

    monkeypatch.setattr(module.ResponseSynthesisAgent, "_build_response_text", _fake_build)
    monkeypatch.setattr(module, "trace", lambda *args, **kwargs: None)

    state = initialize_state("alex", "session-6-6-filter", "destination help", turn_id=5)
    state["conversation_history"] = [
        {"role": "system_summary", "content": "[SUMMARY] old summary must never leak"},
        {"role": "assistant", "content": "How can I help?"},
        {"role": "user", "content": "Show me Southeast Asia ideas"},
    ]
    state["rag_results"] = [{"id": "dest-bali", "name": "Bali"}]
    state["research_results"] = [{"title": "Trend", "link": "https://example.com", "snippet": "x"}]

    update = ResponseSynthesisAgent().run(state)

    assert update["current_response"] == "filtered output"
    assert all(item.get("role") != "system_summary" for item in captured_history)
    assert "[SUMMARY]" not in " ".join(item.get("content", "") for item in captured_history)


def test_response_synthesis_output_has_no_summary_markers_or_summary_text() -> None:
    state = initialize_state("alex", "session-6-6-output", "top destinations", turn_id=5)
    state["conversation_history"] = [
        {"role": "system_summary", "content": "[SUMMARY] private memory compression content"},
        {"role": "user", "content": "Please suggest beach destinations"},
    ]
    state["rag_results"] = [{"id": "dest-phuket", "name": "Phuket"}]
    state["research_results"] = [
        {"title": "Beach trend report", "link": "https://example.com/beach", "snippet": "Demand rising"}
    ]

    update = ResponseSynthesisAgent().run(state)
    response_text = update["current_response"]

    assert "[SUMMARY]" not in response_text
    assert "private memory compression content" not in response_text


def test_system_summary_filtering_applies_each_turn_independently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from concierge.agents import response_synthesis as module

    captured_turn_histories: list[list[dict[str, str]]] = []

    def _fake_build(self, rag_results, research_results, filtered_history):  # noqa: ANN001
        del self, rag_results, research_results
        captured_turn_histories.append(list(filtered_history))
        return "turn response"

    monkeypatch.setattr(module.ResponseSynthesisAgent, "_build_response_text", _fake_build)
    monkeypatch.setattr(module, "trace", lambda *args, **kwargs: None)

    turn5 = initialize_state("alex", "session-6-6-turns", "turn5", turn_id=5)
    turn5["conversation_history"] = [
        {"role": "system_summary", "content": "Turn5 summary"},
        {"role": "user", "content": "turn5 user input"},
    ]
    turn5["rag_results"] = [{"id": "dest-bali", "name": "Bali"}]
    turn5["research_results"] = [{"title": "T5", "link": "https://example.com/t5", "snippet": "x"}]

    turn6 = initialize_state("alex", "session-6-6-turns", "turn6", turn_id=6)
    turn6["conversation_history"] = [
        {"role": "system_summary", "content": "Turn6 summary"},
        {"role": "assistant", "content": "previous answer"},
        {"role": "user", "content": "turn6 user input"},
    ]
    turn6["rag_results"] = [{"id": "dest-bangkok", "name": "Bangkok"}]
    turn6["research_results"] = [{"title": "T6", "link": "https://example.com/t6", "snippet": "y"}]

    ResponseSynthesisAgent().run(turn5)
    ResponseSynthesisAgent().run(turn6)

    assert len(captured_turn_histories) == 2
    for filtered_history in captured_turn_histories:
        assert all(item.get("role") != "system_summary" for item in filtered_history)
