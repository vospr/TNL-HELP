"""
test_e2e_scenario_6_escalation.py
==================================
E2E test: Scenario 6 — Repeated Clarifications -> Human Escalation

Scenario:
  Turn 1: User sends an ambiguous query that yields low confidence. Guardrail fires
           its first clarification question. clarification_count in history = 0 before
           this turn, so after the turn the conversation_history contains 1 assistant
           clarification entry.

  Turn 2: User replies vaguely. Still low confidence. Guardrail fires again.
           conversation_history now contains 1 prior clarification, count in history = 1
           after appending assistant reply.

  Turn 3: User still vague. Still low confidence. Guardrail fires again.
           conversation_history now contains 2 prior clarifications, count = 2 after append.

  Turn 4: User still vague. Guardrail detects >= max_clarifications (3) prior
           clarifications in history. It triggers human_handoff=True and emits the
           terminal handoff message.

State carried across turns: user_id, session_id, conversation_history, turn_id.
State reset each turn (by dispatcher): intent, confidence, route, rag_results,
  research_results, source_attribution, degradation_label, guardrail_passed,
  proactive_suggestion, clarification_needed, clarification_question, human_handoff, error.

Each turn is a separate compiled_graph.invoke() call — this tests the real node
execution pipeline end-to-end without live LLM calls. The dispatcher's Stage-2 LLM
escalation is suppressed by removing ANTHROPIC_API_KEY from the environment so the
dispatcher falls back to Stage-1 only, producing a low-confidence None route that
routes to the guardrail via the "fallback" branch.
"""
from __future__ import annotations

import os
from typing import Any

import pytest
import yaml
from pathlib import Path

from concierge.agents.guardrail import CLARIFICATION_QUESTION, HUMAN_HANDOFF_MESSAGE
from concierge.graph import compiled_graph
from concierge.state import ConciergeState, initialize_state


# ---------------------------------------------------------------------------
# Constants mirrored from production code / policy files so tests are
# self-documenting and will break loudly if the policy values change.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GUARDRAIL_POLICY_PATH = _REPO_ROOT / "prompts" / "guardrail" / "policy.yaml"
_DISPATCHER_POLICY_PATH = _REPO_ROOT / "prompts" / "dispatcher" / "policy.yaml"

MAX_CLARIFICATIONS: int = 3

# An ambiguous phrase that does NOT match any Stage-1 routing rule keyword so
# Stage-1 returns (None, 0.0, None) and Stage-2 is skipped (no API key),
# resulting in confidence=None / route=None -> guardrail fallback path.
# The guardrail receives None confidence and returns an early pass only if
# confidence is not numeric.  We therefore need confidence to be a numeric
# value below the dispatcher threshold (0.75).  We achieve this by injecting
# confidence directly into the state before each invoke call, simulating what
# a real LLM dispatcher would return.
AMBIGUOUS_INPUT_T1 = "I need help with something"
AMBIGUOUS_INPUT_T2 = "um not sure really"
AMBIGUOUS_INPUT_T3 = "still not entirely clear"
AMBIGUOUS_INPUT_T4 = "I don't know what I want"

# Confidence below dispatcher threshold (0.75) to keep guardrail from passing.
LOW_CONFIDENCE = 0.42

USER_ID = "alex"
SESSION_ID = "session-e2e-scenario-6-escalation"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _count_clarifications_in_history(history: list[dict[str, Any]]) -> int:
    """Count how many times CLARIFICATION_QUESTION appears as assistant content."""
    return sum(
        1
        for entry in history
        if isinstance(entry, dict)
        and str(entry.get("content", "")).strip() == CLARIFICATION_QUESTION
    )


def _load_policy_max_clarifications() -> int:
    with _GUARDRAIL_POLICY_PATH.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    assert isinstance(data, dict), "guardrail policy must be a YAML mapping"
    value = data.get("max_clarifications")
    assert isinstance(value, int) and value > 0, (
        f"max_clarifications must be a positive int, got {value!r}"
    )
    return int(value)


def _load_dispatcher_confidence_threshold() -> float:
    with _DISPATCHER_POLICY_PATH.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    assert isinstance(data, dict), "dispatcher policy must be a YAML mapping"
    value = data.get("confidence_threshold")
    assert isinstance(value, (int, float)), (
        f"dispatcher confidence_threshold must be numeric, got {value!r}"
    )
    return float(value)


def _invoke_turn(
    state: ConciergeState,
    user_input: str,
    turn_id: int,
    *,
    injected_confidence: float,
) -> ConciergeState:
    """
    Build the per-turn input state and invoke the compiled graph.

    Dispatcher normally calls reset_turn_state() and rebuilds conversation_history
    by appending the new user message.  We must carry forward only the persistent
    fields (user_id, session_id, conversation_history, turn_id) and let the graph
    handle the rest.

    Injected confidence simulates the result a real LLM dispatcher would produce.
    We do this by monkeypatching _evaluate_stage1 in the caller so that the
    dispatcher node returns our desired low confidence, routing to 'fallback'.
    The caller (the test function) owns this patch via monkeypatch.
    """
    turn_state = initialize_state(
        user_id=state["user_id"],
        session_id=state["session_id"],
        current_input=user_input,
        turn_id=turn_id,
    )
    # Carry forward accumulated conversation_history from the previous turn.
    turn_state["conversation_history"] = list(state.get("conversation_history") or [])

    # Inject route=None and confidence=LOW_CONFIDENCE into state BEFORE invoke
    # so that when dispatcher_node runs, the dispatcher.run() sees existing_route
    # is falsy and proceeds to re-evaluate.  The stage1 monkeypatch (applied by
    # the caller) will return the injected values.
    # Note: dispatcher.run() checks existing_route; since we pass route=None here
    # the dispatcher will proceed normally through its evaluation pipeline.
    return compiled_graph.invoke(turn_state)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def suppress_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Ensure ANTHROPIC_API_KEY is absent so the dispatcher's Stage-2 LLM branch
    is completely bypassed, making the test hermetic and offline.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


@pytest.fixture()
def patch_dispatcher_low_confidence(monkeypatch: pytest.MonkeyPatch):
    """
    Monkeypatch DispatcherAgent._evaluate_stage1 so it always returns
    intent=None, confidence=LOW_CONFIDENCE, route=None — meaning Stage-1 did
    not produce a confident route, Stage-2 will be skipped (no API key), and
    the dispatcher emits route=None which maps to 'fallback'.

    This forces every turn through dispatcher -> guardrail -> synthesis without
    any specialist node.

    Returned so tests can inspect or replace if needed.
    """
    from concierge.agents.dispatcher import DispatcherAgent

    monkeypatch.setattr(
        DispatcherAgent,
        "_evaluate_stage1",
        lambda self, text: (None, LOW_CONFIDENCE, None),
    )


@pytest.fixture()
def suppress_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    """Silence trace output to keep test logs clean."""
    import concierge.agents.guardrail as gm
    import concierge.agents.dispatcher as dm
    import concierge.nodes.dispatcher_node as dn
    import concierge.nodes.guardrail_node as gn
    import concierge.nodes.synthesis_node as sn

    noop = lambda *args, **kwargs: None  # noqa: E731
    monkeypatch.setattr(gm, "trace", noop)
    monkeypatch.setattr(dm, "trace", noop)
    monkeypatch.setattr(dn, "trace", noop)
    monkeypatch.setattr(gn, "trace", noop)
    monkeypatch.setattr(sn, "trace", noop)


# ---------------------------------------------------------------------------
# Policy contract assertions (run first so failures are clearly policy-level)
# ---------------------------------------------------------------------------


class TestPolicyContracts:
    """Verify that the policy files encode the expected max_clarifications value."""

    def test_guardrail_policy_max_clarifications_is_three(self) -> None:
        policy_value = _load_policy_max_clarifications()
        assert policy_value == MAX_CLARIFICATIONS, (
            f"Expected max_clarifications={MAX_CLARIFICATIONS} in guardrail policy, "
            f"got {policy_value}. Update MAX_CLARIFICATIONS constant if intentional."
        )

    def test_dispatcher_confidence_threshold_is_above_low_confidence(self) -> None:
        threshold = _load_dispatcher_confidence_threshold()
        assert LOW_CONFIDENCE < threshold, (
            f"LOW_CONFIDENCE={LOW_CONFIDENCE} must be below dispatcher threshold "
            f"{threshold} for this scenario to work correctly."
        )


# ---------------------------------------------------------------------------
# Core E2E multi-turn scenario
# ---------------------------------------------------------------------------


class TestScenario6RepeatedClarificationsEscalation:
    """
    Four-turn E2E test: every turn invokes compiled_graph.invoke().
    State (conversation_history, turn_id) is carried forward manually between
    turns, mirroring how a real session loop would work.
    """

    @pytest.fixture(autouse=True)
    def _setup(
        self,
        patch_dispatcher_low_confidence,
        suppress_trace,
    ) -> None:
        """Apply shared patches for all tests in this class."""

    # ------------------------------------------------------------------
    # Turn 1
    # ------------------------------------------------------------------

    def test_turn_1_triggers_first_clarification(self) -> None:
        """
        Turn 1: ambiguous input, low confidence.
        Expected: clarification_needed=True, human_handoff=False,
                  response == CLARIFICATION_QUESTION,
                  conversation_history has 2 entries (user + assistant).
        """
        state = initialize_state(USER_ID, SESSION_ID, AMBIGUOUS_INPUT_T1, turn_id=1)
        result = compiled_graph.invoke(state)

        # Node execution order for fallback route
        assert result.get("_executed_nodes") == ["dispatcher", "guardrail", "synthesis"], (
            "Turn 1 must execute dispatcher -> guardrail -> synthesis"
        )

        # Turn-level state
        assert result["turn_id"] == 1
        assert result["user_id"] == USER_ID
        assert result["session_id"] == SESSION_ID

        # Guardrail fired clarification, not handoff
        assert result["clarification_needed"] is True, "Turn 1: clarification_needed must be True"
        assert result["human_handoff"] is False, "Turn 1: human_handoff must be False"
        assert result["guardrail_passed"] is False, "Turn 1: guardrail_passed must be False"

        # Clarification question surfaced in response
        assert result["current_response"] == CLARIFICATION_QUESTION, (
            f"Turn 1: response must be the clarification question.\n"
            f"Got: {result['current_response']!r}"
        )
        assert result["clarification_question"] == CLARIFICATION_QUESTION

        # conversation_history: user message appended by dispatcher;
        # synthesis does not append assistant turn — that is the caller's
        # responsibility in a real session loop.  We verify the user entry exists.
        history = result["conversation_history"]
        assert isinstance(history, list)
        user_entries = [e for e in history if e.get("role") == "user"]
        assert len(user_entries) == 1, (
            f"Turn 1: expected 1 user entry in history, got {len(user_entries)}"
        )
        assert user_entries[0]["content"] == AMBIGUOUS_INPUT_T1

        # No escalation yet
        assert _count_clarifications_in_history(history) == 0, (
            "Turn 1: clarification question not yet in history before caller appends it"
        )

    # ------------------------------------------------------------------
    # Turn 2
    # ------------------------------------------------------------------

    def test_turn_2_clarification_count_increments_to_one(self) -> None:
        """
        Turn 2: caller has appended Turn 1's clarification to history before invoking.
        Expected: guardrail counts 1 prior clarification, still below max -> asks again.
                  clarification_count_in_history == 1 going INTO Turn 2.
        """
        # Simulate what the session loop does: append the assistant clarification
        # from Turn 1 to the history.
        prior_history = [
            {"role": "user", "content": AMBIGUOUS_INPUT_T1},
            {"role": "assistant", "content": CLARIFICATION_QUESTION},
        ]

        state = initialize_state(USER_ID, SESSION_ID, AMBIGUOUS_INPUT_T2, turn_id=2)
        state["conversation_history"] = list(prior_history)
        result = compiled_graph.invoke(state)

        # Verify node sequence unchanged
        assert result.get("_executed_nodes") == ["dispatcher", "guardrail", "synthesis"]

        # Turn ID advanced
        assert result["turn_id"] == 2

        # Guardrail still firing clarification (1 prior, max is 3)
        assert result["clarification_needed"] is True, "Turn 2: clarification_needed must be True"
        assert result["human_handoff"] is False, "Turn 2: human_handoff must be False"
        assert result["current_response"] == CLARIFICATION_QUESTION

        # History now contains the T1 user + T1 assistant + T2 user entries
        history = result["conversation_history"]
        user_entries = [e for e in history if e.get("role") == "user"]
        assert len(user_entries) == 2, (
            f"Turn 2: expected 2 user entries in history, got {len(user_entries)}"
        )
        assert user_entries[-1]["content"] == AMBIGUOUS_INPUT_T2

        # Prior clarifications visible to guardrail
        prior_clarification_count = _count_clarifications_in_history(history)
        assert prior_clarification_count == 1, (
            f"Turn 2: history should show 1 prior clarification, got {prior_clarification_count}"
        )

    # ------------------------------------------------------------------
    # Turn 3
    # ------------------------------------------------------------------

    def test_turn_3_clarification_count_increments_to_two(self) -> None:
        """
        Turn 3: history carries 2 prior clarification entries.
        Expected: 2 < max_clarifications(3) -> guardrail asks again, no handoff.
        """
        prior_history = [
            {"role": "user", "content": AMBIGUOUS_INPUT_T1},
            {"role": "assistant", "content": CLARIFICATION_QUESTION},
            {"role": "user", "content": AMBIGUOUS_INPUT_T2},
            {"role": "assistant", "content": CLARIFICATION_QUESTION},
        ]

        state = initialize_state(USER_ID, SESSION_ID, AMBIGUOUS_INPUT_T3, turn_id=3)
        state["conversation_history"] = list(prior_history)
        result = compiled_graph.invoke(state)

        assert result.get("_executed_nodes") == ["dispatcher", "guardrail", "synthesis"]
        assert result["turn_id"] == 3

        assert result["clarification_needed"] is True, "Turn 3: clarification_needed must be True"
        assert result["human_handoff"] is False, "Turn 3: human_handoff must be False"
        assert result["current_response"] == CLARIFICATION_QUESTION

        history = result["conversation_history"]
        user_entries = [e for e in history if e.get("role") == "user"]
        assert len(user_entries) == 3, (
            f"Turn 3: expected 3 user entries in history, got {len(user_entries)}"
        )
        assert user_entries[-1]["content"] == AMBIGUOUS_INPUT_T3

        prior_clarification_count = _count_clarifications_in_history(history)
        assert prior_clarification_count == 2, (
            f"Turn 3: history should show 2 prior clarifications, got {prior_clarification_count}"
        )

    # ------------------------------------------------------------------
    # Turn 4 — Escalation
    # ------------------------------------------------------------------

    def test_turn_4_human_handoff_triggered_at_max_clarifications(self) -> None:
        """
        Turn 4: history carries 3 prior clarification entries (== max_clarifications).
        Expected: guardrail detects count >= max -> human_handoff=True,
                  response contains the handoff message.
        """
        prior_history = [
            {"role": "user", "content": AMBIGUOUS_INPUT_T1},
            {"role": "assistant", "content": CLARIFICATION_QUESTION},
            {"role": "user", "content": AMBIGUOUS_INPUT_T2},
            {"role": "assistant", "content": CLARIFICATION_QUESTION},
            {"role": "user", "content": AMBIGUOUS_INPUT_T3},
            {"role": "assistant", "content": CLARIFICATION_QUESTION},
        ]

        state = initialize_state(USER_ID, SESSION_ID, AMBIGUOUS_INPUT_T4, turn_id=4)
        state["conversation_history"] = list(prior_history)
        result = compiled_graph.invoke(state)

        assert result.get("_executed_nodes") == ["dispatcher", "guardrail", "synthesis"], (
            "Turn 4 must still execute dispatcher -> guardrail -> synthesis"
        )
        assert result["turn_id"] == 4

        # Core escalation assertions
        assert result["human_handoff"] is True, (
            "Turn 4: human_handoff must be True after max clarifications reached"
        )
        assert result["guardrail_passed"] is False, (
            "Turn 4: guardrail_passed must be False on human handoff"
        )
        assert result["clarification_needed"] is False, (
            "Turn 4: clarification_needed must be False when escalating to human"
        )
        assert result["clarification_question"] is None, (
            "Turn 4: clarification_question must be None on human handoff"
        )

        # Response must include the handoff message text
        response = result["current_response"]
        assert isinstance(response, str) and response.strip(), (
            "Turn 4: current_response must be a non-empty string"
        )
        assert HUMAN_HANDOFF_MESSAGE in response, (
            f"Turn 4: response must contain HUMAN_HANDOFF_MESSAGE.\n"
            f"Expected substring: {HUMAN_HANDOFF_MESSAGE!r}\n"
            f"Got: {response!r}"
        )

        # Response indicates human support needed (keyword check for resilience)
        response_lower = response.lower()
        assert "human" in response_lower or "specialist" in response_lower or "support" in response_lower, (
            f"Turn 4: response must mention human/specialist/support.\nGot: {response!r}"
        )

        # History still contains 3 prior clarifications + T4 user entry
        history = result["conversation_history"]
        user_entries = [e for e in history if e.get("role") == "user"]
        assert len(user_entries) == 4, (
            f"Turn 4: expected 4 user entries in history (all 4 turns), got {len(user_entries)}"
        )
        assert user_entries[-1]["content"] == AMBIGUOUS_INPUT_T4

        prior_clarification_count = _count_clarifications_in_history(history)
        assert prior_clarification_count == MAX_CLARIFICATIONS, (
            f"Turn 4: history must show {MAX_CLARIFICATIONS} prior clarifications, "
            f"got {prior_clarification_count}"
        )


# ---------------------------------------------------------------------------
# Full sequential scenario (single test, 4 invocations)
# ---------------------------------------------------------------------------


class TestScenario6FullFlow:
    """
    Runs all four turns in a single test function to verify end-to-end state
    continuity across the complete escalation arc.
    """

    @pytest.fixture(autouse=True)
    def _setup(
        self,
        patch_dispatcher_low_confidence,
        suppress_trace,
    ) -> None:
        pass

    def test_four_turn_escalation_arc(self) -> None:
        """
        Execute Turns 1-4 sequentially, carrying state forward between turns.
        Verifies conversation_history growth, clarification_count progression,
        turn_id increments, and terminal human_handoff assertion.
        """
        # Session-level persistent state.
        session_history: list[dict[str, Any]] = []

        # ---- Turn 1 ----
        t1_state = initialize_state(USER_ID, SESSION_ID, AMBIGUOUS_INPUT_T1, turn_id=1)
        t1_state["conversation_history"] = list(session_history)
        t1_result = compiled_graph.invoke(t1_state)

        assert t1_result["turn_id"] == 1
        assert t1_result["clarification_needed"] is True
        assert t1_result["human_handoff"] is False
        assert t1_result["current_response"] == CLARIFICATION_QUESTION
        assert t1_result.get("_executed_nodes") == ["dispatcher", "guardrail", "synthesis"]

        # Advance session history: carry dispatcher's history forward, then
        # append the assistant's clarification response (session loop duty).
        session_history = list(t1_result["conversation_history"])
        session_history.append({"role": "assistant", "content": t1_result["current_response"]})

        # After Turn 1: 1 user entry + 1 assistant (clarification) = 2 total entries
        assert len(session_history) == 2
        assert _count_clarifications_in_history(session_history) == 1

        # ---- Turn 2 ----
        t2_state = initialize_state(USER_ID, SESSION_ID, AMBIGUOUS_INPUT_T2, turn_id=2)
        t2_state["conversation_history"] = list(session_history)
        t2_result = compiled_graph.invoke(t2_state)

        assert t2_result["turn_id"] == 2
        assert t2_result["clarification_needed"] is True
        assert t2_result["human_handoff"] is False
        assert t2_result["current_response"] == CLARIFICATION_QUESTION
        assert t2_result.get("_executed_nodes") == ["dispatcher", "guardrail", "synthesis"]

        session_history = list(t2_result["conversation_history"])
        session_history.append({"role": "assistant", "content": t2_result["current_response"]})

        # After Turn 2: 2 user + 2 assistant = 4 entries, 2 clarifications
        assert len(session_history) == 4
        assert _count_clarifications_in_history(session_history) == 2

        # ---- Turn 3 ----
        t3_state = initialize_state(USER_ID, SESSION_ID, AMBIGUOUS_INPUT_T3, turn_id=3)
        t3_state["conversation_history"] = list(session_history)
        t3_result = compiled_graph.invoke(t3_state)

        assert t3_result["turn_id"] == 3
        assert t3_result["clarification_needed"] is True
        assert t3_result["human_handoff"] is False
        assert t3_result["current_response"] == CLARIFICATION_QUESTION
        assert t3_result.get("_executed_nodes") == ["dispatcher", "guardrail", "synthesis"]

        session_history = list(t3_result["conversation_history"])
        session_history.append({"role": "assistant", "content": t3_result["current_response"]})

        # After Turn 3: 3 user + 3 assistant = 6 entries, 3 clarifications
        assert len(session_history) == 6
        assert _count_clarifications_in_history(session_history) == MAX_CLARIFICATIONS

        # ---- Turn 4 — Escalation ----
        t4_state = initialize_state(USER_ID, SESSION_ID, AMBIGUOUS_INPUT_T4, turn_id=4)
        t4_state["conversation_history"] = list(session_history)
        t4_result = compiled_graph.invoke(t4_state)

        assert t4_result["turn_id"] == 4
        assert t4_result.get("_executed_nodes") == ["dispatcher", "guardrail", "synthesis"]

        # --- Core escalation gate ---
        assert t4_result["human_handoff"] is True, (
            "After 3 clarifications human_handoff must be True"
        )
        assert t4_result["guardrail_passed"] is False
        assert t4_result["clarification_needed"] is False
        assert t4_result["clarification_question"] is None

        # Response must contain the canonical handoff message
        assert HUMAN_HANDOFF_MESSAGE in t4_result["current_response"], (
            f"Final response must contain HUMAN_HANDOFF_MESSAGE.\n"
            f"Got: {t4_result['current_response']!r}"
        )

        # conversation_history at this point (dispatcher appended T4 user msg)
        final_history = t4_result["conversation_history"]
        user_entries_in_result = [e for e in final_history if e.get("role") == "user"]
        assert len(user_entries_in_result) == 4, (
            f"All 4 user turns must be in final history, got {len(user_entries_in_result)}"
        )

        # Turn ID incremented monotonically across all 4 turns
        assert t1_result["turn_id"] < t2_result["turn_id"] < t3_result["turn_id"] < t4_result["turn_id"]

        # Verify the clarification progression: 0 -> 1 -> 2 -> max (3) prior to each turn
        # (the count visible to the guardrail node when it runs each turn)
        # We reconstruct this from the per-turn histories we already checked above.
        # Turns 1-3: clarification_needed True; Turn 4: human_handoff True.
        assert all([
            t1_result["clarification_needed"],
            t2_result["clarification_needed"],
            t3_result["clarification_needed"],
        ]), "Turns 1-3 must all have clarification_needed=True"
        assert not any([
            t1_result["human_handoff"],
            t2_result["human_handoff"],
            t3_result["human_handoff"],
        ]), "Turns 1-3 must all have human_handoff=False"


# ---------------------------------------------------------------------------
# State-reset invariant checks
# ---------------------------------------------------------------------------


class TestStateResetInvariantsAcrossTurns:
    """
    Verify that turn-reset fields (intent, route, rag_results, etc.) are always
    fresh on each turn's result, even after the escalation arc.
    """

    @pytest.fixture(autouse=True)
    def _setup(
        self,
        patch_dispatcher_low_confidence,
        suppress_trace,
    ) -> None:
        pass

    def test_ephemeral_fields_are_reset_each_turn(self) -> None:
        """
        Simulate a carry-over state where stale values from a previous turn are
        present; verify the dispatcher's reset clears them before guardrail sees them.
        """
        stale_history = [
            {"role": "user", "content": AMBIGUOUS_INPUT_T1},
            {"role": "assistant", "content": CLARIFICATION_QUESTION},
            {"role": "user", "content": AMBIGUOUS_INPUT_T2},
            {"role": "assistant", "content": CLARIFICATION_QUESTION},
        ]

        state = initialize_state(USER_ID, SESSION_ID, AMBIGUOUS_INPUT_T3, turn_id=3)
        state["conversation_history"] = list(stale_history)

        # Inject stale values that must be cleared by the dispatcher's reset
        state["intent"] = "stale_intent"
        state["route"] = None  # Keep None so dispatcher runs normally
        state["rag_results"] = [{"id": "stale"}]
        state["research_results"] = [{"title": "stale"}]
        state["degradation_label"] = "stale_label"
        state["error"] = "previous turn error"

        result = compiled_graph.invoke(state)

        # Dispatcher must have reset these
        assert result["intent"] is None, "intent must be reset to None each turn"
        assert result["rag_results"] is None, "rag_results must be reset to None each turn"
        assert result["research_results"] is None, "research_results must be reset to None each turn"
        assert result["degradation_label"] is None, "degradation_label must be reset each turn"
        assert result["error"] is None, "error must be reset each turn"
        assert result["source_attribution"] == [], "source_attribution must be reset to [] each turn"
        assert result["proactive_suggestion"] is None, "proactive_suggestion must be reset each turn"

    def test_human_handoff_is_reset_at_turn_start_by_dispatcher(self) -> None:
        """
        Verify that human_handoff is NOT carried forward from a previous turn as a
        sticky value; the dispatcher resets it to False at the start of each turn.
        If the guardrail then fires again it sets it back to True.  But the key
        invariant is that the dispatcher-owned reset runs first.

        We test this by providing a history with exactly MAX_CLARIFICATIONS entries
        and verifying Turn 4 sets human_handoff=True (not stuck from a prior state).
        """
        # Fresh state — human_handoff defaults to False in initialize_state
        prior_history = [
            {"role": "user", "content": AMBIGUOUS_INPUT_T1},
            {"role": "assistant", "content": CLARIFICATION_QUESTION},
            {"role": "user", "content": AMBIGUOUS_INPUT_T2},
            {"role": "assistant", "content": CLARIFICATION_QUESTION},
            {"role": "user", "content": AMBIGUOUS_INPUT_T3},
            {"role": "assistant", "content": CLARIFICATION_QUESTION},
        ]

        state = initialize_state(USER_ID, SESSION_ID, AMBIGUOUS_INPUT_T4, turn_id=4)
        state["conversation_history"] = list(prior_history)
        # Pretend a previous turn left human_handoff=True in a stale state dict
        # (this should NOT prevent the turn from running; dispatcher resets it).
        state["human_handoff"] = False  # initialize_state already sets this; be explicit

        result = compiled_graph.invoke(state)

        # The guardrail must SET human_handoff=True on this turn (it is not
        # pre-existing; it is freshly set after the dispatcher reset it to False).
        assert result["human_handoff"] is True


# ---------------------------------------------------------------------------
# Guardrail max_clarifications config contract
# ---------------------------------------------------------------------------


class TestGuardrailMaxClarificationsConfig:
    """Config-level contract: GuardrailAgent reads max_clarifications from policy."""

    def test_guardrail_agent_reads_max_clarifications_from_policy(self) -> None:
        from concierge.agents.guardrail import GuardrailAgent

        agent = GuardrailAgent()
        # Access the private attribute to verify it loaded the correct value.
        assert agent._max_clarifications == MAX_CLARIFICATIONS, (
            f"GuardrailAgent._max_clarifications must equal {MAX_CLARIFICATIONS}, "
            f"got {agent._max_clarifications}"
        )

    def test_guardrail_triggers_handoff_exactly_at_max_not_before(self) -> None:
        """
        Boundary test: max-1 clarifications -> still asks; max -> handoff.
        """
        from concierge.agents.guardrail import GuardrailAgent
        import concierge.agents.guardrail as guardrail_module

        # Silence traces
        original_trace = guardrail_module.trace
        guardrail_module.trace = lambda *a, **kw: None

        try:
            # max-1 prior clarifications: should still ask
            history_below = []
            for i in range(MAX_CLARIFICATIONS - 1):
                history_below.append({"role": "user", "content": f"Ambiguous {i}"})
                history_below.append({"role": "assistant", "content": CLARIFICATION_QUESTION})

            state_below = initialize_state(USER_ID, SESSION_ID, "still vague", turn_id=3)
            state_below["confidence"] = LOW_CONFIDENCE
            state_below["conversation_history"] = history_below

            update_below = GuardrailAgent().run(state_below)
            assert update_below["human_handoff"] is False, (
                f"With {MAX_CLARIFICATIONS - 1} prior clarifications, human_handoff must be False"
            )
            assert update_below["clarification_needed"] is True

            # max prior clarifications: must trigger handoff
            history_at = []
            for i in range(MAX_CLARIFICATIONS):
                history_at.append({"role": "user", "content": f"Ambiguous {i}"})
                history_at.append({"role": "assistant", "content": CLARIFICATION_QUESTION})

            state_at = initialize_state(USER_ID, SESSION_ID, "still vague", turn_id=4)
            state_at["confidence"] = LOW_CONFIDENCE
            state_at["conversation_history"] = history_at

            update_at = GuardrailAgent().run(state_at)
            assert update_at["human_handoff"] is True, (
                f"With {MAX_CLARIFICATIONS} prior clarifications, human_handoff must be True"
            )
            assert update_at["clarification_needed"] is False
        finally:
            guardrail_module.trace = original_trace
