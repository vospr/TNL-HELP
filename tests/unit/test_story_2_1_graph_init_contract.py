from __future__ import annotations

from copy import deepcopy


EXPECTED_GRAPH_NODES = {
    "dispatcher",
    "rag",
    "research",
    "synthesis",
    "guardrail",
    "followup",
    "booking_stub",
}


def test_initialize_state_sets_optional_defaults() -> None:
    from concierge.state import initialize_state

    state = initialize_state(user_id="alex", session_id="session-test", current_input="hello")

    assert state["rag_results"] is None
    assert state["research_results"] is None
    assert state["degradation_label"] is None
    assert state["proactive_suggestion"] is None
    assert state["clarification_question"] is None
    assert state["error"] is None
    assert state["source_attribution"] == []
    assert state["guardrail_passed"] is True
    assert state["clarification_needed"] is False
    assert state["human_handoff"] is False


def test_compiled_graph_is_importable_and_runnable() -> None:
    from concierge.graph import compiled_graph
    from concierge.state import initialize_state

    state = initialize_state(user_id="alex", session_id="session-test", current_input="hello")
    result = compiled_graph.invoke(state)

    assert isinstance(result, dict)
    assert result["user_id"] == "alex"


def test_graph_exposes_all_7_nodes_and_expected_edges() -> None:
    from concierge.graph import CONDITIONAL_EDGES, GRAPH_EDGES, GRAPH_NODES

    assert set(GRAPH_NODES) == EXPECTED_GRAPH_NODES

    expected_edges = {
        ("dispatcher", "rag"),
        ("dispatcher", "research"),
        ("dispatcher", "booking_stub"),
        ("dispatcher", "guardrail"),
        ("rag", "synthesis"),
        ("research", "synthesis"),
    }
    assert expected_edges.issubset(set(GRAPH_EDGES))
    assert "dispatcher" in CONDITIONAL_EDGES
    assert "synthesis" in CONDITIONAL_EDGES


def test_graph_invoke_does_not_mutate_input_object_and_returns_modified_copy() -> None:
    from concierge.graph import compiled_graph
    from concierge.state import initialize_state

    initial = initialize_state(user_id="alex", session_id="session-test", current_input="hello")
    before = deepcopy(initial)

    result = compiled_graph.invoke(initial)

    assert initial == before, "compiled_graph.invoke must not mutate input state in place"
    assert result is not initial
    assert result != before, "invoke should return a modified state copy"
