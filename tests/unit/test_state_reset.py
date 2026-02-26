from __future__ import annotations

from copy import deepcopy

import validate_config
from concierge.agents.dispatcher import reset_turn_state


def test_reset_turn_state_returns_expected_turn_reset_contract(
    fresh_concierge_state: dict[str, object],
) -> None:
    state = deepcopy(fresh_concierge_state)
    state["intent"] = "research_intent"
    state["confidence"] = 0.91
    state["route"] = "research"
    state["rag_results"] = [{"id": "KB-1"}]
    state["research_results"] = [{"id": "WEB-1"}]
    state["source_attribution"] = ["[RAG] KB-1"]
    state["degradation_label"] = "[WEB SEARCH UNAVAILABLE — serving from internal KB only]"
    state["proactive_suggestion"] = "Try Thailand."
    state["clarification_needed"] = True
    state["clarification_question"] = "Do you mean flights?"
    state["human_handoff"] = True
    state["guardrail_passed"] = False
    state["error"] = "timeout"

    reset = reset_turn_state(state)

    assert reset == {
        "intent": None,
        "confidence": None,
        "route": None,
        "rag_results": None,
        "research_results": None,
        "source_attribution": [],
        "degradation_label": None,
        "guardrail_passed": True,
        "proactive_suggestion": None,
        "clarification_needed": False,
        "clarification_question": None,
        "human_handoff": False,
        "error": None,
    }


def test_stale_rag_results_do_not_bleed_between_turns(
    fresh_concierge_state: dict[str, object],
) -> None:
    turn_1 = deepcopy(fresh_concierge_state)
    turn_1["route"] = "rag"
    turn_1["rag_results"] = [{"id": "KB Entry"}]
    turn_1["source_attribution"] = ["[RAG] KB Entry"]

    turn_2 = deepcopy(turn_1)
    turn_2.update(reset_turn_state(turn_2))
    turn_2["route"] = "research"

    assert turn_1["rag_results"] == [{"id": "KB Entry"}]
    assert turn_2["rag_results"] is None
    assert turn_2["route"] == "research"
    assert turn_2["source_attribution"] == []


def test_validate_config_static_check_enforces_reset_contract() -> None:
    errors = validate_config.check_reset_turn_state_contract_non_blocking()
    assert errors == []
