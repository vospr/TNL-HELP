from __future__ import annotations

from pathlib import Path

import pytest

from concierge.graph import compiled_graph
from concierge.state import initialize_state


FOLLOWUP_SUGGESTION = (
    "Based on your interest in Southeast Asia, you might also want to explore mountain retreats. "
    "Should I research that for you?"
)


def test_followup_agent_generates_suggestion_and_trace_for_research_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from concierge.agents import followup as followup_module

    trace_calls: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        followup_module,
        "trace",
        lambda node_name, outcome=None, **fields: trace_calls.append((node_name, dict(fields))),
    )

    state = initialize_state("alex", "session-6-5-followup", "research ideas", turn_id=1)
    state["route"] = "research"
    state["guardrail_passed"] = True
    state["current_response"] = "Research findings [Web]."

    update = followup_module.FollowUpAgent().run(state)

    assert update["proactive_suggestion"] == FOLLOWUP_SUGGESTION
    assert FOLLOWUP_SUGGESTION in update["current_response"]
    assert trace_calls == [("followup", {"event": "suggestion_generated"})]


def test_graph_runs_followup_only_for_research_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from concierge.agents import research_agent as research_module

    monkeypatch.setattr(research_module, "search_duckduckgo", lambda query, max_results=5: [])

    research = initialize_state("alex", "session-6-5-research", "hello", turn_id=1)
    research["route"] = "research"
    research_result = compiled_graph.invoke(research)
    assert research_result.get("_executed_nodes") == [
        "dispatcher",
        "research",
        "guardrail",
        "synthesis",
        "followup",
    ]
    assert research_result.get("proactive_suggestion") == FOLLOWUP_SUGGESTION

    rag = initialize_state("alex", "session-6-5-rag", "hello", turn_id=2)
    rag["route"] = "rag"
    rag_result = compiled_graph.invoke(rag)
    assert rag_result.get("_executed_nodes") == [
        "dispatcher",
        "rag",
        "guardrail",
        "synthesis",
    ]
    assert rag_result.get("proactive_suggestion") is None

    booking = initialize_state("alex", "session-6-5-booking", "hello", turn_id=3)
    booking["route"] = "booking_stub"
    booking_result = compiled_graph.invoke(booking)
    assert booking_result.get("_executed_nodes") == [
        "dispatcher",
        "booking_stub",
        "guardrail",
        "synthesis",
    ]
    assert booking_result.get("proactive_suggestion") is None


def test_graph_source_documents_research_only_followup_conditional_edge() -> None:
    graph_source = (
        Path(__file__).resolve().parents[2] / "src" / "concierge" / "graph" / "__init__.py"
    ).read_text(encoding="utf-8")

    assert "Follow-Up is intentionally conditioned on research route only" in graph_source
