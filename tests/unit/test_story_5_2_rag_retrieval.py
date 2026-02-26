from __future__ import annotations

import pytest

from concierge.agents.rag_agent import RAGAgent
from concierge.state import initialize_state


def test_rag_agent_retrieves_southeast_asia_beach_destinations() -> None:
    state = initialize_state(
        user_id="alex",
        session_id="session-5-2-beach",
        current_input="Best beach destinations in Southeast Asia",
        turn_id=1,
    )

    update = RAGAgent().run(state)
    results = update["rag_results"]

    assert isinstance(results, list)
    assert results
    names = {str(entry["name"]) for entry in results}
    assert {"Bali", "Phuket"}.issubset(names)
    for entry in results:
        assert str(entry["region"]) == "Southeast Asia"
        assert "beach" in [str(item).lower() for item in entry["amenities"]]


def test_rag_agent_uses_current_input_not_full_conversation_history() -> None:
    state = initialize_state(
        user_id="alex",
        session_id="session-5-2-scope",
        current_input="city destinations",
        turn_id=2,
    )
    state["conversation_history"] = [
        {"role": "user", "content": "best beach destinations in Southeast Asia"}
    ]

    update = RAGAgent().run(state)
    names = {str(entry["name"]) for entry in update["rag_results"]}

    assert {"Tokyo", "Bangkok"}.issubset(names)
    assert "Bali" not in names
    assert "Phuket" not in names


def test_rag_agent_emits_retrieval_complete_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from concierge.agents import rag_agent as rag_agent_module

    trace_calls: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        rag_agent_module,
        "trace",
        lambda node_name, outcome=None, **fields: trace_calls.append((node_name, dict(fields))),
    )

    state = initialize_state(
        user_id="alex",
        session_id="session-5-2-trace",
        current_input="beach",
        turn_id=1,
    )
    update = rag_agent_module.RAGAgent().run(state)

    assert isinstance(update["rag_results"], list)
    assert trace_calls == [
        (
            "rag_agent",
            {"event": "retrieval_complete", "results_count": len(update["rag_results"])},
        )
    ]


def test_rag_agent_no_match_returns_empty_list_and_fallback_message() -> None:
    state = initialize_state(
        user_id="alex",
        session_id="session-5-2-empty",
        current_input="polar expeditions in antarctica",
        turn_id=3,
    )

    update = RAGAgent().run(state)

    assert update["rag_results"] == []
    assert update["current_response"] == "No matching internal KB destinations found."
