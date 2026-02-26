"""
E2E Scenario 9 — API Key Missing: Graceful Degradation
=======================================================

Scenario: ANTHROPIC_API_KEY is absent from the environment.

Expected behaviour
------------------
- Stage 1 (regex/pattern routing) works independently of any API key.
- Stage 2 (LLM escalation) detects the missing key via
  ``DispatcherAgent._evaluate_stage2`` and returns ``(None, None, None)``
  immediately — no import attempt, no network call, no raised exception.
- RAG retrieval (MockKnowledgeBase) is purely file-based; it is likewise
  independent of the API key.
- RAG LLM ranking is skipped when the key is absent (``_rank_results_with_llm``
  short-circuits) — results are still returned in insertion order.
- ResponseSynthesisAgent is purely deterministic string assembly; it never
  touches the Anthropic SDK.
- For a *strong-pattern* user input the full pipeline completes and returns a
  well-formed, non-error response.
- For an *ambiguous* user input that matches no Stage-1 rule the pipeline
  gracefully falls back (route=None → guardrail → synthesis) and produces a
  response rather than crashing.

Coverage matrix
---------------
Test                                       | Input type  | Checks
test_stage1_pattern_match_no_api_key       | strong      | route, intent, confidence
test_stage2_skipped_when_no_api_key        | strong      | _evaluate_stage2 returns (None,None,None)
test_rag_agent_works_without_api_key       | strong      | rag_results non-empty, no error key
test_rag_llm_ranking_skipped_no_api_key    | strong      | ranking NOT called; results still returned
test_synthesis_works_without_api_key       | strong      | current_response valid, no API error text
test_full_pipeline_strong_pattern          | strong      | end-to-end graph invoke, no crash
test_full_pipeline_ambiguous_input         | ambiguous   | graceful fallback, no crash, response set
test_no_anthropic_import_error_strong      | strong      | anthropic module never imported during run
test_no_anthropic_import_error_ambiguous   | ambiguous   | anthropic module never imported during run
test_response_contains_no_api_key_error    | strong      | "API_KEY" absent from current_response
test_confidence_may_be_none_no_crash       | ambiguous   | no exception even with None confidence
test_system_continues_after_no_key         | strong→ambig| two sequential turns, no crash on either
"""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock

import pytest

from concierge.agents.dispatcher import DispatcherAgent
from concierge.agents.rag_agent import RAGAgent
from concierge.agents.response_synthesis import ResponseSynthesisAgent
from concierge.graph import compiled_graph
from concierge.nodes.error_handling import INVALID_API_KEY_ERROR_MESSAGE
from concierge.state import initialize_state

# ---------------------------------------------------------------------------
# Constants shared across tests
# ---------------------------------------------------------------------------

# "resort" (singular) matches Stage-1 rule: pattern='\b(wyndham|property|resort)\b',
# intent=property_lookup, route=rag, score=0.91.
# 0.91 >= escalation_threshold (0.72) → Stage 1 resolves route immediately,
# Stage 2 is never reached.
#
# NOTE: "resorts" (plural) does NOT match because \b requires a non-word char
# after "t" — "resorts" has "s" after "t", which is still a word char.
# We use "resort" (singular) to guarantee the pattern fires.
STRONG_INPUT = "Tell me about the resort in Bali"

# No Stage-1 keyword match → Stage 1 returns (None, 0.0, None).
# Stage 2 would run normally but is skipped when ANTHROPIC_API_KEY is absent.
AMBIGUOUS_INPUT = "Show me something nice"

# Text that must NEVER appear in a non-error response.
API_KEY_ERROR_PHRASES = [
    "ANTHROPIC_API_KEY",
    "API_KEY not set",
    "api key",
    "authentication",
    "unauthorized",
    "invalid key",
    INVALID_API_KEY_ERROR_MESSAGE,
]


# ---------------------------------------------------------------------------
# Helper: build a minimal state dict
# ---------------------------------------------------------------------------

def _make_state(
    session_id: str,
    current_input: str,
    turn_id: int = 1,
) -> dict[str, Any]:
    state = dict(
        initialize_state(
            user_id="e2e-test-user",
            session_id=session_id,
            current_input=current_input,
            turn_id=turn_id,
        )
    )
    state["route"] = None
    return state


# ---------------------------------------------------------------------------
# Helper: assert response is valid (non-empty, no API-key error text)
# ---------------------------------------------------------------------------

def _assert_valid_response(result: dict[str, Any]) -> None:
    response = str(result.get("current_response") or "")
    lowered = response.lower()
    for phrase in API_KEY_ERROR_PHRASES:
        assert phrase.lower() not in lowered, (
            f"Response must not contain API-key error phrase '{phrase}'. "
            f"Got: {response!r}"
        )


# ===========================================================================
# Group A — Dispatcher: Stage 1 and Stage 2 behaviour without API key
# ===========================================================================


class TestDispatcherWithoutApiKey:
    """Unit-scope tests that drive DispatcherAgent directly."""

    def test_stage1_pattern_match_no_api_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Stage 1 must resolve route='rag' for STRONG_INPUT without any API key."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        state = _make_state("e2e-s9-stage1-strong", STRONG_INPUT)
        update = DispatcherAgent().run(state)

        # Correct route without LLM
        assert update["route"] == "rag", (
            f"Expected route='rag', got {update['route']!r}"
        )
        assert update["intent"] == "property_lookup"
        # Confidence must meet or exceed the escalation threshold (0.72)
        assert isinstance(update["confidence"], float)
        assert update["confidence"] >= 0.72, (
            f"Stage-1 confidence {update['confidence']} is below escalation threshold"
        )

    def test_stage2_skipped_when_no_api_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """_evaluate_stage2 must return (None, None, None) when the key is absent."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        agent = DispatcherAgent()
        intent, confidence, route = agent._evaluate_stage2(AMBIGUOUS_INPUT)

        assert intent is None, f"Expected intent=None, got {intent!r}"
        assert confidence is None, f"Expected confidence=None, got {confidence!r}"
        assert route is None, f"Expected route=None, got {route!r}"

    def test_stage2_does_not_attempt_import_when_no_api_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No Anthropic SDK import must be attempted when API key is absent.

        If _evaluate_stage2 short-circuits on the env-var check (line 253 in
        dispatcher.py), the ``import anthropic`` statement is never reached.
        We verify this by poisoning sys.modules so that any import of
        'anthropic' raises ImportError — the call must still succeed silently.
        """
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        # Remove any cached real module so the poison takes effect
        monkeypatch.delitem(sys.modules, "anthropic", raising=False)
        monkeypatch.setitem(
            sys.modules,
            "anthropic",
            None,  # type: ignore[arg-type]  # None causes ImportError on access
        )

        agent = DispatcherAgent()
        # Must not raise even though sys.modules["anthropic"] is None
        intent, confidence, route = agent._evaluate_stage2(AMBIGUOUS_INPUT)
        assert intent is None
        assert confidence is None
        assert route is None

    def test_stage2_skipped_for_strong_input_no_api_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Stage 1 resolves STRONG_INPUT before Stage 2 is even considered.

        We confirm Stage 2 is never entered by patching it to raise — the run
        must complete without triggering the patch.
        """
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        stage2_called: list[bool] = []

        def _stage2_sentinel(
            self: DispatcherAgent,
            current_input: str,
        ) -> tuple[str | None, float | None, str | None]:
            stage2_called.append(True)
            raise AssertionError(
                "Stage 2 must NOT be called when Stage 1 resolves the route"
            )

        monkeypatch.setattr(DispatcherAgent, "_evaluate_stage2", _stage2_sentinel)

        state = _make_state("e2e-s9-stage2-skip", STRONG_INPUT)
        update = DispatcherAgent().run(state)

        assert not stage2_called, "Stage 2 was invoked unexpectedly"
        assert update["route"] == "rag"


# ===========================================================================
# Group B — RAG agent: independent of API key
# ===========================================================================


class TestRAGAgentWithoutApiKey:
    """RAG retrieval must work fully without ANTHROPIC_API_KEY."""

    def test_rag_agent_returns_results_without_api_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MockKnowledgeBase queries are file-based; no API key required."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        state = _make_state("e2e-s9-rag-results", STRONG_INPUT)
        update = RAGAgent().run(state)

        # "resort" and "bali" both appear in the Bali knowledge-base entry
        rag_results = update.get("rag_results")
        assert isinstance(rag_results, list), (
            f"rag_results must be a list, got {type(rag_results)}"
        )
        assert len(rag_results) > 0, "Expected at least one RAG result for STRONG_INPUT"

        # Bali is an exact match for both "resort" and "bali" tokens
        names = {str(entry.get("name", "")).strip() for entry in rag_results}
        assert "Bali" in names, (
            f"Expected 'Bali' in rag_results names, got {names}"
        )

    def test_rag_agent_no_error_key_without_api_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """RAG agent must not set 'error' when there is no API key."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        state = _make_state("e2e-s9-rag-noerr", STRONG_INPUT)
        update = RAGAgent().run(state)

        assert update.get("error") is None, (
            f"RAG agent must not set 'error'; got {update.get('error')!r}"
        )

    def test_rag_llm_ranking_skipped_when_no_api_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """LLM ranking must be skipped (not errored) when key is absent.

        The RAGAgent._rank_results_with_llm guard checks both
        ``self._llm_ranking_enabled`` and the API key. Here we enable the flag
        but clear the key; results must still be returned in their original order.
        """
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        # Force the LLM-ranking branch on
        monkeypatch.setenv("RAG_AGENT_LLM_RANKING", "1")

        # Poison the anthropic module to guarantee no SDK call reaches the
        # network layer — if the guard fails, this MagicMock call would still
        # succeed, but we verify results are consistent with the unranked path.
        mock_anthropic = MagicMock()
        monkeypatch.setitem(sys.modules, "anthropic", mock_anthropic)

        state = _make_state("e2e-s9-rag-ranking-skip", STRONG_INPUT)
        agent = RAGAgent()

        sample_results = [
            {"id": "dest-bali", "name": "Bali", "region": "Southeast Asia"},
            {"id": "dest-phuket", "name": "Phuket", "region": "Southeast Asia"},
        ]
        # _rank_results_with_llm must short-circuit on missing API key and
        # return the input list unchanged.
        ranked = agent._rank_results_with_llm(STRONG_INPUT, sample_results)

        assert ranked == sample_results, (
            "Results must be returned unchanged when LLM ranking is skipped "
            f"due to missing API key. Got: {ranked}"
        )
        # The SDK constructor must not have been called
        mock_anthropic.Anthropic.assert_not_called()


# ===========================================================================
# Group C — ResponseSynthesisAgent: purely deterministic, no API key needed
# ===========================================================================


class TestSynthesisWithoutApiKey:
    """Synthesis assembles text from state fields; never touches the Anthropic SDK."""

    def test_synthesis_produces_valid_response_from_rag_results(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Synthesis must produce a [RAG]-tagged response from pre-populated results."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        state = _make_state("e2e-s9-synth-rag", STRONG_INPUT)
        state["rag_results"] = [
            {"id": "dest-bali", "name": "Bali", "region": "Southeast Asia"},
        ]
        state["research_results"] = []

        update = ResponseSynthesisAgent().run(state)

        response = str(update.get("current_response") or "")
        assert response, "Synthesis must produce a non-empty response"
        assert "[RAG]" in response, (
            f"Expected '[RAG]' tag in synthesis response. Got: {response!r}"
        )
        _assert_valid_response(update)

    def test_synthesis_produces_fallback_when_no_rag_results(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Synthesis must not crash when rag_results is empty.

        With empty results and no existing current_response the agent returns
        an empty dict update (no-op), leaving the existing current_response
        untouched. Either that or a pre-set fallback string is acceptable.
        """
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        state = _make_state("e2e-s9-synth-empty", AMBIGUOUS_INPUT)
        state["rag_results"] = []
        state["research_results"] = []
        state["current_response"] = None

        # Must not raise
        update = ResponseSynthesisAgent().run(state)
        # update may be {} or contain current_response — either is acceptable
        # as long as no exception is raised and no API-key error text leaks in.
        if update.get("current_response"):
            _assert_valid_response(update)


# ===========================================================================
# Group D — Full pipeline (graph.invoke) integration
# ===========================================================================


class TestFullPipelineWithoutApiKey:
    """End-to-end graph invocations without ANTHROPIC_API_KEY."""

    def test_full_pipeline_strong_pattern_no_api_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Strong-pattern input → rag → guardrail → synthesis, no crash, valid response.

        The graph must resolve to:
          dispatcher (route=rag) → rag → guardrail → synthesis
        """
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        state = _make_state("e2e-s9-full-strong", STRONG_INPUT)
        result = compiled_graph.invoke(state)

        # Route resolved by Stage 1
        assert result.get("route") == "rag", (
            f"Expected route='rag', got {result.get('route')!r}"
        )
        assert result.get("intent") == "property_lookup"
        assert result.get("confidence") == pytest.approx(0.91)

        # RAG must have run and found results
        rag_results = result.get("rag_results")
        assert isinstance(rag_results, list) and rag_results, (
            "rag_results must be a non-empty list"
        )

        # Final response must be present and valid
        response = str(result.get("current_response") or "")
        assert response, "current_response must be non-empty after full pipeline"
        _assert_valid_response(result)

        # No error flag must be set
        assert result.get("error") is None, (
            f"'error' must be None; got {result.get('error')!r}"
        )
        assert result.get("human_handoff") is False, (
            "human_handoff must be False for a clean rag run"
        )

        # Execution breadcrumbs
        executed = result.get("_executed_nodes") or []
        assert "dispatcher" in executed
        assert "rag" in executed
        assert "synthesis" in executed

    def test_full_pipeline_ambiguous_input_no_api_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ambiguous input without API key → graceful fallback, no crash.

        Stage 1 returns (None, 0.0, None).
        Stage 2 is skipped (no API key) → returns (None, None, None).
        Dispatcher sets route=None.
        Graph defaults route to 'fallback' → guardrail → synthesis.
        No exception must be raised. The pipeline must complete.
        """
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        state = _make_state("e2e-s9-full-ambiguous", AMBIGUOUS_INPUT)
        result = compiled_graph.invoke(state)

        # Must not crash — primary assertion
        # Route is None (stage1/stage2 both miss) → graph treats as 'fallback'
        # The stored route field reflects what the dispatcher set, which is None.
        assert result.get("route") is None or result.get("route") == "fallback", (
            f"Expected route None or 'fallback', got {result.get('route')!r}"
        )

        # Confidence and intent may be None — that is acceptable
        confidence = result.get("confidence")
        intent = result.get("intent")
        # No type assertion: both may be None or a valid value
        _ = confidence
        _ = intent

        # Execution must have reached synthesis
        executed = result.get("_executed_nodes") or []
        assert "dispatcher" in executed, "Dispatcher must have run"
        assert "synthesis" in executed, "Synthesis must have run even on fallback path"

        # No API-key error must appear in the response
        if result.get("current_response"):
            _assert_valid_response(result)

    def test_response_contains_no_api_key_error_string(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify ANTHROPIC_API_KEY error text never reaches current_response."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        state = _make_state("e2e-s9-noerr-str", STRONG_INPUT)
        result = compiled_graph.invoke(state)

        response = str(result.get("current_response") or "")
        for phrase in API_KEY_ERROR_PHRASES:
            assert phrase.lower() not in response.lower(), (
                f"API-key error phrase '{phrase}' must not appear in response. "
                f"Got: {response!r}"
            )

    def test_no_anthropic_import_error_strong_pattern(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Anthropic SDK must never be imported during a strong-pattern run.

        We poison sys.modules['anthropic'] = None so that any ``import anthropic``
        inside the pipeline raises ImportError. The pipeline must still complete
        without raising.
        """
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delitem(sys.modules, "anthropic", raising=False)
        # Setting the module to None causes ``import anthropic`` to raise ImportError
        monkeypatch.setitem(sys.modules, "anthropic", None)  # type: ignore[arg-type]

        state = _make_state("e2e-s9-noimport-strong", STRONG_INPUT)
        # Must not raise ImportError or any other exception
        result = compiled_graph.invoke(state)

        assert result.get("route") == "rag"
        assert result.get("error") is None

    def test_no_anthropic_import_error_ambiguous_input(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Anthropic SDK must never be imported during an ambiguous-input run.

        Stage 2 checks the API key first and short-circuits before the
        ``import anthropic`` line, so poisoning the module must not cause a crash.
        """
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delitem(sys.modules, "anthropic", raising=False)
        monkeypatch.setitem(sys.modules, "anthropic", None)  # type: ignore[arg-type]

        state = _make_state("e2e-s9-noimport-ambig", AMBIGUOUS_INPUT)
        # Must not raise
        result = compiled_graph.invoke(state)

        executed = result.get("_executed_nodes") or []
        assert "dispatcher" in executed
        assert "synthesis" in executed

    def test_confidence_may_be_none_no_crash(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ambiguous input leaves confidence=None; no component must crash on that."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        state = _make_state("e2e-s9-none-conf", AMBIGUOUS_INPUT)
        # Must not raise
        result = compiled_graph.invoke(state)

        # State-level confidence may be None (dispatcher sets None when both
        # Stage 1 and Stage 2 miss)
        assert "confidence" in result  # key present
        # Value is None or a float — both are acceptable
        confidence = result["confidence"]
        assert confidence is None or isinstance(confidence, float), (
            f"confidence must be None or float, got {type(confidence)}"
        )

        # No exception path triggered
        assert result.get("error") is None or INVALID_API_KEY_ERROR_MESSAGE not in str(
            result.get("error") or ""
        ), "API key error must not appear in the error field"

    def test_system_continues_across_sequential_turns_without_api_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two sequential turns — strong then ambiguous — must both complete.

        Validates that the system stays operational (does not enter a broken
        state) after processing a turn without an API key.
        """
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        # Turn 1: strong pattern
        state_turn1 = _make_state("e2e-s9-sequential-t1", STRONG_INPUT, turn_id=1)
        result_turn1 = compiled_graph.invoke(state_turn1)

        assert result_turn1.get("route") == "rag", (
            f"Turn 1 must route to 'rag'. Got {result_turn1.get('route')!r}"
        )
        assert result_turn1.get("error") is None

        # Turn 2: ambiguous — must also complete without crash
        state_turn2 = _make_state("e2e-s9-sequential-t2", AMBIGUOUS_INPUT, turn_id=2)
        # Carry conversation history forward (simulating real session)
        state_turn2["conversation_history"] = list(
            result_turn1.get("conversation_history") or []
        )

        result_turn2 = compiled_graph.invoke(state_turn2)

        executed_t2 = result_turn2.get("_executed_nodes") or []
        assert "dispatcher" in executed_t2
        assert "synthesis" in executed_t2
        # Must not raise or set an API-key error
        if result_turn2.get("current_response"):
            _assert_valid_response(result_turn2)


# ===========================================================================
# Group E — Isolation / regression: empty vs. missing key are equivalent
# ===========================================================================


class TestEmptyKeyEquivalentToMissingKey:
    """An empty string value for ANTHROPIC_API_KEY must behave identically to absent."""

    @pytest.mark.parametrize("key_value", ["", "   "])
    def test_stage2_skipped_for_blank_api_key(
        self,
        monkeypatch: pytest.MonkeyPatch,
        key_value: str,
    ) -> None:
        """Stage 2 guard uses ``os.environ.get(...).strip()`` — blank == absent."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", key_value)

        agent = DispatcherAgent()
        intent, confidence, route = agent._evaluate_stage2(AMBIGUOUS_INPUT)

        assert intent is None
        assert confidence is None
        assert route is None

    @pytest.mark.parametrize("key_value", ["", "   "])
    def test_rag_llm_ranking_skipped_for_blank_api_key(
        self,
        monkeypatch: pytest.MonkeyPatch,
        key_value: str,
    ) -> None:
        """RAG LLM ranking guard uses the same ``os.environ.get(...).strip()`` pattern."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", key_value)
        monkeypatch.setenv("RAG_AGENT_LLM_RANKING", "1")

        mock_anthropic = MagicMock()
        monkeypatch.setitem(sys.modules, "anthropic", mock_anthropic)

        agent = RAGAgent()
        sample = [{"id": "dest-bali", "name": "Bali"}]
        ranked = agent._rank_results_with_llm("resort in Bali", sample)

        assert ranked == sample
        mock_anthropic.Anthropic.assert_not_called()
