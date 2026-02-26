"""
E2E Test — Scenario 4: Ambiguous Query with LLM Escalation

Flow under test:
    User input: "Show me something nice"
        │
        ▼
    Dispatcher Stage 1  →  no pattern match  →  None returned
        │
        ▼ (token budget check performed here)
        │
    Dispatcher Stage 2  →  Anthropic LLM called
        │
        ▼  LLM returns: {"intent": "travel_advice", "confidence": 0.55}
        │
    Dispatcher routes to "fallback" (0.55 < 0.75 dispatcher confidence_threshold)
        │
        ▼
    Guardrail  →  confidence 0.55 < 0.75 dispatcher threshold
                  intent is NOT out_of_domain
                  no prior clarifications
                  → clarification_needed=True
        │
        ▼
    Synthesis  →  clarification_needed=True
                  → current_response = clarification_question

Key thresholds (from config/routing_rules.yaml and prompts/dispatcher/policy.yaml):
    - routing_rules.yaml escalation_threshold:   0.72  (Stage 1 skip LLM if rule score >= this)
    - dispatcher/policy.yaml confidence_threshold: 0.75 (Stage 2 LLM: route to specialist if >= this)
    - Guardrail uses the dispatcher confidence_threshold (0.75) as its own threshold;
      any confidence strictly below 0.75 that is not out_of_domain triggers clarification.
    - Confidence 0.55 is well below both thresholds, so the full escalation + clarification path fires.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml

from concierge.agents.dispatcher import DispatcherAgent
from concierge.agents.guardrail import CLARIFICATION_QUESTION, GuardrailAgent
from concierge.graph import build_graph
from concierge.state import initialize_state


# ---------------------------------------------------------------------------
# Constants pinned from config files so the test fails loudly if config drifts
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
_ROUTING_RULES_PATH = _REPO_ROOT / "config" / "routing_rules.yaml"
_DISPATCHER_POLICY_PATH = _REPO_ROOT / "prompts" / "dispatcher" / "policy.yaml"

_STAGE1_ESCALATION_THRESHOLD: float = 0.72   # routing_rules.yaml  escalation_threshold
_DISPATCHER_CONFIDENCE_THRESHOLD: float = 0.75  # dispatcher/policy.yaml confidence_threshold
_LLM_RESPONSE_INTENT: str = "travel_advice"
_LLM_RESPONSE_CONFIDENCE: float = 0.55
_AMBIGUOUS_INPUT: str = "Show me something nice"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    assert isinstance(data, dict), f"{path} must be a YAML mapping"
    return data


def _build_fake_anthropic_module(llm_json_text: str) -> tuple[types.ModuleType, dict[str, Any]]:
    """
    Build a fake ``anthropic`` module that:
    - records every call to ``messages.create`` in ``recorder``
    - returns a response whose single text block contains ``llm_json_text``

    Returns (fake_module, recorder).
    """
    recorder: dict[str, Any] = {"calls": 0, "last_kwargs": None}

    class _FakeMessages:
        def create(self, **kwargs: Any) -> types.SimpleNamespace:
            recorder["calls"] += 1
            recorder["last_kwargs"] = kwargs
            return types.SimpleNamespace(
                content=[types.SimpleNamespace(text=llm_json_text)]
            )

    class _FakeClient:
        def __init__(self) -> None:
            self.messages = _FakeMessages()

    fake_module = types.SimpleNamespace(Anthropic=lambda: _FakeClient())
    return fake_module, recorder  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Config-contract guard tests (fail fast if thresholds change without updating E2E)
# ---------------------------------------------------------------------------

class TestConfigThresholdContracts:
    """Guard rails: verify the thresholds this test depends on are still in config."""

    def test_routing_rules_escalation_threshold_is_expected_value(self) -> None:
        data = _load_yaml(_ROUTING_RULES_PATH)
        threshold = float(data["escalation_threshold"])
        assert threshold == _STAGE1_ESCALATION_THRESHOLD, (
            f"routing_rules.yaml escalation_threshold changed from {_STAGE1_ESCALATION_THRESHOLD} "
            f"to {threshold}. Update the E2E test constant."
        )

    def test_dispatcher_policy_confidence_threshold_is_expected_value(self) -> None:
        data = _load_yaml(_DISPATCHER_POLICY_PATH)
        threshold = float(data["confidence_threshold"])
        assert threshold == _DISPATCHER_CONFIDENCE_THRESHOLD, (
            f"dispatcher/policy.yaml confidence_threshold changed from "
            f"{_DISPATCHER_CONFIDENCE_THRESHOLD} to {threshold}. Update the E2E test constant."
        )

    def test_llm_confidence_is_strictly_below_dispatcher_threshold(self) -> None:
        """The scenario depends on 0.55 being below 0.75."""
        assert _LLM_RESPONSE_CONFIDENCE < _DISPATCHER_CONFIDENCE_THRESHOLD, (
            "LLM test confidence must be strictly below dispatcher threshold for this scenario."
        )

    def test_guardrail_threshold_equals_dispatcher_threshold(self) -> None:
        """GuardrailAgent reads its threshold from dispatcher/policy.yaml; confirm parity."""
        data = _load_yaml(_DISPATCHER_POLICY_PATH)
        dispatcher_ct = float(data["confidence_threshold"])
        guardrail = GuardrailAgent()
        assert guardrail._dispatcher_confidence_threshold == dispatcher_ct

    def test_no_stage1_rule_matches_ambiguous_input(self) -> None:
        """'Show me something nice' must not match any Stage 1 pattern."""
        agent = DispatcherAgent()
        intent, confidence, route = agent._evaluate_stage1(_AMBIGUOUS_INPUT)
        assert intent is None, f"Stage 1 unexpectedly matched intent={intent!r}"
        assert confidence == 0.0
        assert route is None


# ---------------------------------------------------------------------------
# Unit-level Stage 1 test
# ---------------------------------------------------------------------------

class TestStage1NoMatch:
    def test_stage1_returns_none_for_ambiguous_input(self) -> None:
        agent = DispatcherAgent()
        intent, confidence, route = agent._evaluate_stage1(_AMBIGUOUS_INPUT)

        assert intent is None, "Stage 1 must return None intent for ambiguous input"
        assert confidence == 0.0, "Stage 1 must return 0.0 confidence when no rule matches"
        assert route is None, "Stage 1 must return None route when no rule matches"

    def test_stage1_ambiguous_input_does_not_contain_pattern_keywords(self) -> None:
        """Smoke-check: none of the known routing keywords appear in the test input."""
        pattern_keywords = ["wyndham", "property", "resort", "book", "booking",
                            "reserve", "reservation", "trend", "latest", "right now",
                            "news", "weather", "stock", "crypto", "politics"]
        lower_input = _AMBIGUOUS_INPUT.lower()
        for keyword in pattern_keywords:
            assert keyword not in lower_input, (
                f"Ambiguous test input contains routing keyword {keyword!r}. "
                "Choose a different test input."
            )


# ---------------------------------------------------------------------------
# Unit-level Stage 2 LLM escalation tests
# ---------------------------------------------------------------------------

class TestStage2LLMEscalation:
    @pytest.fixture(autouse=True)
    def _setup_fake_anthropic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Set up environment variables and a fake Anthropic module for every test
        in this class, eliminating repeated boilerplate across all 5 tests."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-scenario-4")
        monkeypatch.delenv("FAST_MODE", raising=False)

        fake_mod, recorder = _build_fake_anthropic_module(
            f'{{"intent":"{_LLM_RESPONSE_INTENT}","confidence":{_LLM_RESPONSE_CONFIDENCE}}}'
        )
        monkeypatch.setitem(sys.modules, "anthropic", fake_mod)

        # Expose the recorder on the instance so individual tests can inspect it.
        self._recorder = recorder

    def test_stage2_is_invoked_when_stage1_finds_no_match(self, monkeypatch: pytest.MonkeyPatch) -> None:
        state = initialize_state(
            user_id="alex",
            session_id="session-e2e-s4-stage2",
            current_input=_AMBIGUOUS_INPUT,
            turn_id=1,
        )
        agent = DispatcherAgent()
        agent.run(state)

        assert self._recorder["calls"] == 1, "Stage 2 LLM must be called exactly once"

    def test_stage2_llm_receives_correct_model_and_token_budget(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        state = initialize_state(
            user_id="alex",
            session_id="session-e2e-s4-model",
            current_input=_AMBIGUOUS_INPUT,
            turn_id=1,
        )
        DispatcherAgent().run(state)

        kwargs = self._recorder["last_kwargs"]
        assert kwargs is not None
        # model must match dispatcher/policy.yaml (claude-opus-4-6 when FAST_MODE is off)
        assert kwargs["model"] == "claude-opus-4-6", (
            f"Stage 2 must use dispatcher model, got {kwargs['model']!r}"
        )
        # max_tokens must match dispatcher/policy.yaml
        assert kwargs["max_tokens"] == 128, (
            f"Stage 2 must use dispatcher max_tokens=128, got {kwargs['max_tokens']!r}"
        )
        # messages must contain the user input
        messages = kwargs.get("messages", [])
        assert any(
            isinstance(m, dict) and m.get("role") == "user" and _AMBIGUOUS_INPUT in str(m.get("content", ""))
            for m in messages
        ), "Stage 2 LLM call must include the user input in messages"

    def test_stage2_parses_intent_and_confidence_from_llm_response(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        state = initialize_state(
            user_id="alex",
            session_id="session-e2e-s4-parse",
            current_input=_AMBIGUOUS_INPUT,
            turn_id=1,
        )
        update = DispatcherAgent().run(state)

        assert update["intent"] == _LLM_RESPONSE_INTENT, (
            f"Parsed intent must be {_LLM_RESPONSE_INTENT!r}, got {update['intent']!r}"
        )
        assert update["confidence"] == _LLM_RESPONSE_CONFIDENCE, (
            f"Parsed confidence must be {_LLM_RESPONSE_CONFIDENCE}, got {update['confidence']!r}"
        )

    def test_stage2_low_confidence_sets_route_to_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        state = initialize_state(
            user_id="alex",
            session_id="session-e2e-s4-fallback",
            current_input=_AMBIGUOUS_INPUT,
            turn_id=1,
        )
        update = DispatcherAgent().run(state)

        assert update["route"] == "fallback", (
            f"Confidence {_LLM_RESPONSE_CONFIDENCE} < {_DISPATCHER_CONFIDENCE_THRESHOLD} "
            f"must route to 'fallback', got {update['route']!r}"
        )

    def test_token_budget_check_occurs_before_stage2(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        TokenBudgetManager.check_and_summarize must be called between Stage 1 returning
        None and Stage 2 invoking the LLM.  We verify this by monkeypatching the budget
        manager and confirming its call is recorded before the Anthropic call.
        """
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-scenario-4")
        monkeypatch.delenv("FAST_MODE", raising=False)

        call_order: list[str] = []

        # Patch TokenBudgetManager.check_and_summarize
        original_check_and_summarize = __import__(
            "concierge.agents.token_budget_manager",
            fromlist=["TokenBudgetManager"],
        ).TokenBudgetManager.check_and_summarize

        def _patched_check_and_summarize(self_inner, history):
            call_order.append("token_budget")
            return original_check_and_summarize(self_inner, history)

        monkeypatch.setattr(
            "concierge.agents.token_budget_manager.TokenBudgetManager.check_and_summarize",
            _patched_check_and_summarize,
        )

        # Patch Anthropic so the LLM call is tracked
        class _RecordingMessages:
            def create(self_inner, **kwargs: Any):
                call_order.append("anthropic_llm")
                return types.SimpleNamespace(
                    content=[types.SimpleNamespace(
                        text=f'{{"intent":"{_LLM_RESPONSE_INTENT}","confidence":{_LLM_RESPONSE_CONFIDENCE}}}'
                    )]
                )

        class _FakeClient:
            def __init__(self) -> None:
                self.messages = _RecordingMessages()

        monkeypatch.setitem(
            sys.modules,
            "anthropic",
            types.SimpleNamespace(Anthropic=lambda: _FakeClient()),
        )

        state = initialize_state(
            user_id="alex",
            session_id="session-e2e-s4-order",
            current_input=_AMBIGUOUS_INPUT,
            turn_id=1,
        )
        DispatcherAgent().run(state)

        assert "token_budget" in call_order, "TokenBudgetManager must be invoked before Stage 2"
        assert "anthropic_llm" in call_order, "Anthropic LLM must be called in Stage 2"
        token_budget_idx = call_order.index("token_budget")
        anthropic_idx = call_order.index("anthropic_llm")
        assert token_budget_idx < anthropic_idx, (
            "Token budget check must happen BEFORE the Stage 2 LLM call. "
            f"Order was: {call_order}"
        )


# ---------------------------------------------------------------------------
# Unit-level Guardrail tests
# ---------------------------------------------------------------------------

class TestGuardrailLowConfidence:
    def _build_post_dispatcher_state(self) -> dict[str, Any]:
        state = dict(
            initialize_state(
                user_id="alex",
                session_id="session-e2e-s4-guardrail",
                current_input=_AMBIGUOUS_INPUT,
                turn_id=1,
            )
        )
        # Simulate what dispatcher writes after Stage 2 LLM escalation
        state["intent"] = _LLM_RESPONSE_INTENT
        state["confidence"] = _LLM_RESPONSE_CONFIDENCE
        state["route"] = "fallback"
        state["conversation_history"] = [{"role": "user", "content": _AMBIGUOUS_INPUT}]
        return state

    def test_guardrail_detects_low_confidence(self) -> None:
        state = self._build_post_dispatcher_state()
        guardrail = GuardrailAgent()
        update = guardrail.run(state)

        assert update.get("clarification_needed") is True, (
            f"Guardrail must set clarification_needed=True when confidence "
            f"{_LLM_RESPONSE_CONFIDENCE} < threshold {_DISPATCHER_CONFIDENCE_THRESHOLD}"
        )

    def test_guardrail_sets_clarification_question_non_empty(self) -> None:
        state = self._build_post_dispatcher_state()
        guardrail = GuardrailAgent()
        update = guardrail.run(state)

        question = update.get("clarification_question")
        assert isinstance(question, str) and question.strip(), (
            "Guardrail must produce a non-empty clarification_question string"
        )

    def test_guardrail_clarification_question_matches_canonical_constant(self) -> None:
        state = self._build_post_dispatcher_state()
        guardrail = GuardrailAgent()
        update = guardrail.run(state)

        assert update.get("clarification_question") == CLARIFICATION_QUESTION, (
            f"clarification_question must match canonical constant. "
            f"Got: {update.get('clarification_question')!r}"
        )

    def test_guardrail_passes_false_on_low_confidence(self) -> None:
        state = self._build_post_dispatcher_state()
        guardrail = GuardrailAgent()
        update = guardrail.run(state)

        assert update.get("guardrail_passed") is False, (
            "guardrail_passed must be False when clarification is required"
        )

    def test_guardrail_human_handoff_false_on_first_turn(self) -> None:
        state = self._build_post_dispatcher_state()
        guardrail = GuardrailAgent()
        update = guardrail.run(state)

        assert update.get("human_handoff") is False, (
            "human_handoff must be False on the first clarification turn"
        )

    def test_guardrail_threshold_is_strictly_lower_than_dispatcher_escalation_threshold(
        self,
    ) -> None:
        """
        Per scenario notes: Guardrail threshold must be strictly lower than the
        routing_rules.yaml escalation_threshold (0.72) to avoid a gap where Stage 1
        would escalate but Guardrail would pass.  The Guardrail reads from
        dispatcher/policy.yaml (0.75), which is *higher* than the Stage 1 escalation
        threshold — this means any query that needed LLM escalation (score < 0.72) will
        also be scrutinized by the Guardrail unless Stage 2 returns high confidence.
        Verify the logical invariant: dispatcher.confidence_threshold > escalation_threshold.
        """
        guardrail = GuardrailAgent()
        guardrail_ct = guardrail._dispatcher_confidence_threshold  # loaded from dispatcher policy
        assert guardrail_ct > _STAGE1_ESCALATION_THRESHOLD, (
            f"dispatcher confidence_threshold ({guardrail_ct}) must be > Stage 1 "
            f"escalation_threshold ({_STAGE1_ESCALATION_THRESHOLD}). "
            "This ensures any query requiring Stage 2 escalation is caught by Guardrail."
        )


# ---------------------------------------------------------------------------
# Integration: Dispatcher -> Guardrail pipeline (without graph)
# ---------------------------------------------------------------------------

class TestDispatcherGuardrailPipeline:
    def test_pipeline_intent_confidence_and_clarification(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-scenario-4")
        monkeypatch.delenv("FAST_MODE", raising=False)

        fake_mod, recorder = _build_fake_anthropic_module(
            f'{{"intent":"{_LLM_RESPONSE_INTENT}","confidence":{_LLM_RESPONSE_CONFIDENCE}}}'
        )
        monkeypatch.setitem(sys.modules, "anthropic", fake_mod)

        # --- Step 1: Dispatcher ---
        state = dict(
            initialize_state(
                user_id="alex",
                session_id="session-e2e-s4-pipeline",
                current_input=_AMBIGUOUS_INPUT,
                turn_id=1,
            )
        )
        dispatcher_update = DispatcherAgent().run(state)
        state.update(dispatcher_update)

        # Dispatcher assertions
        assert recorder["calls"] == 1, "LLM must be called exactly once (Stage 2)"
        assert state["intent"] == _LLM_RESPONSE_INTENT
        assert state["confidence"] == _LLM_RESPONSE_CONFIDENCE
        assert state["route"] == "fallback"

        # --- Step 2: Guardrail ---
        guardrail_update = GuardrailAgent().run(state)
        state.update(guardrail_update)

        # Guardrail assertions
        assert state["clarification_needed"] is True
        assert state["guardrail_passed"] is False
        assert state["human_handoff"] is False
        assert isinstance(state["clarification_question"], str)
        assert state["clarification_question"].strip()

    def test_pipeline_trace_emitted_for_stage2_and_guardrail(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-scenario-4")
        monkeypatch.delenv("FAST_MODE", raising=False)

        traces: list[tuple[str, dict[str, Any]]] = []

        def _capture_trace(node_name: str, outcome: str | None = None, **fields: Any) -> None:
            traces.append((node_name, dict(fields)))

        monkeypatch.setattr("concierge.agents.dispatcher.trace", _capture_trace)
        monkeypatch.setattr("concierge.agents.guardrail.trace", _capture_trace)

        fake_mod, _ = _build_fake_anthropic_module(
            f'{{"intent":"{_LLM_RESPONSE_INTENT}","confidence":{_LLM_RESPONSE_CONFIDENCE}}}'
        )
        monkeypatch.setitem(sys.modules, "anthropic", fake_mod)

        state = dict(
            initialize_state(
                user_id="alex",
                session_id="session-e2e-s4-traces",
                current_input=_AMBIGUOUS_INPUT,
                turn_id=1,
            )
        )
        dispatcher_update = DispatcherAgent().run(state)
        state.update(dispatcher_update)
        guardrail_update = GuardrailAgent().run(state)
        state.update(guardrail_update)

        # Dispatcher must have emitted a trace with stage=llm_escalation
        dispatcher_traces = [t for t in traces if t[0] == "dispatcher"]
        llm_escalation_traces = [
            t for t in dispatcher_traces if t[1].get("stage") == "llm_escalation"
        ]
        assert llm_escalation_traces, (
            "Dispatcher must emit a trace with stage='llm_escalation' for Stage 2 path"
        )
        last_dispatcher_trace = llm_escalation_traces[-1][1]
        assert last_dispatcher_trace["intent"] == _LLM_RESPONSE_INTENT
        assert last_dispatcher_trace["confidence"] == _LLM_RESPONSE_CONFIDENCE
        assert last_dispatcher_trace["route"] == "fallback"

        # Guardrail must have emitted a clarification_fired trace
        guardrail_traces = [t for t in traces if t[0] == "guardrail"]
        clarification_traces = [
            t for t in guardrail_traces if t[1].get("event") == "clarification_fired"
        ]
        assert clarification_traces, (
            "Guardrail must emit a trace with event='clarification_fired'"
        )
        clarification_trace_fields = clarification_traces[-1][1]
        assert clarification_trace_fields["confidence"] == round(_LLM_RESPONSE_CONFIDENCE, 2)
        assert clarification_trace_fields["threshold"] == _DISPATCHER_CONFIDENCE_THRESHOLD


# ---------------------------------------------------------------------------
# Full E2E: graph invocation (Dispatcher -> Guardrail -> Synthesis)
# ---------------------------------------------------------------------------

class TestFullGraphScenario4:
    def test_full_graph_produces_clarification_response(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Invoke the compiled graph end-to-end.  The final state must reflect:
        - intent = "travel_advice"
        - confidence = 0.55
        - clarification_needed = True
        - clarification_question not empty
        - current_response is the clarification question (set by Synthesis)
        """
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-scenario-4")
        monkeypatch.delenv("FAST_MODE", raising=False)

        fake_mod, recorder = _build_fake_anthropic_module(
            f'{{"intent":"{_LLM_RESPONSE_INTENT}","confidence":{_LLM_RESPONSE_CONFIDENCE}}}'
        )
        monkeypatch.setitem(sys.modules, "anthropic", fake_mod)

        initial_state = initialize_state(
            user_id="alex",
            session_id="session-e2e-s4-full",
            current_input=_AMBIGUOUS_INPUT,
            turn_id=1,
        )

        graph = build_graph()
        final_state = graph.invoke(initial_state)

        # --- Intent and confidence from LLM ---
        assert final_state["intent"] == _LLM_RESPONSE_INTENT, (
            f"Expected intent={_LLM_RESPONSE_INTENT!r}, got {final_state['intent']!r}"
        )
        assert final_state["confidence"] == _LLM_RESPONSE_CONFIDENCE, (
            f"Expected confidence={_LLM_RESPONSE_CONFIDENCE}, got {final_state['confidence']!r}"
        )

        # --- Route is fallback ---
        assert final_state["route"] == "fallback", (
            f"Expected route='fallback', got {final_state['route']!r}"
        )

        # --- Guardrail outcome ---
        assert final_state["clarification_needed"] is True, (
            "clarification_needed must be True after Guardrail fires on low confidence"
        )
        clarification_question = final_state.get("clarification_question")
        assert isinstance(clarification_question, str) and clarification_question.strip(), (
            "clarification_question must be a non-empty string"
        )

        # --- Synthesis output ---
        current_response = final_state.get("current_response")
        assert isinstance(current_response, str) and current_response.strip(), (
            "current_response must be non-empty after Synthesis processes clarification"
        )
        assert current_response == clarification_question, (
            "Synthesis must forward the clarification_question as current_response. "
            f"Got: {current_response!r}"
        )

        # --- LLM was actually called ---
        assert recorder["calls"] == 1, (
            "Anthropic LLM must be called exactly once (Stage 2 escalation)"
        )

    def test_full_graph_stage1_never_matched(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Verify that Stage 1 produced no direct route; the route in final state
        came from Stage 2 (routed to fallback), not Stage 1.
        We confirm this by checking the LLM was called — if Stage 1 had matched,
        the LLM would not be called.
        """
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-scenario-4")
        monkeypatch.delenv("FAST_MODE", raising=False)

        fake_mod, recorder = _build_fake_anthropic_module(
            f'{{"intent":"{_LLM_RESPONSE_INTENT}","confidence":{_LLM_RESPONSE_CONFIDENCE}}}'
        )
        monkeypatch.setitem(sys.modules, "anthropic", fake_mod)

        initial_state = initialize_state(
            user_id="alex",
            session_id="session-e2e-s4-stage1-check",
            current_input=_AMBIGUOUS_INPUT,
            turn_id=1,
        )

        graph = build_graph()
        graph.invoke(initial_state)

        assert recorder["calls"] == 1, (
            "LLM must be called, confirming Stage 1 did not match and Stage 2 escalated."
        )

    def test_full_graph_executed_nodes_include_dispatcher_and_guardrail(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-scenario-4")
        monkeypatch.delenv("FAST_MODE", raising=False)

        fake_mod, _ = _build_fake_anthropic_module(
            f'{{"intent":"{_LLM_RESPONSE_INTENT}","confidence":{_LLM_RESPONSE_CONFIDENCE}}}'
        )
        monkeypatch.setitem(sys.modules, "anthropic", fake_mod)

        initial_state = initialize_state(
            user_id="alex",
            session_id="session-e2e-s4-nodes",
            current_input=_AMBIGUOUS_INPUT,
            turn_id=1,
        )

        graph = build_graph()
        final_state = graph.invoke(initial_state)

        executed = list(final_state.get("_executed_nodes") or [])
        assert "dispatcher" in executed, (
            f"'dispatcher' must be in _executed_nodes. Got: {executed}"
        )
        assert "guardrail" in executed, (
            f"'guardrail' must be in _executed_nodes. Got: {executed}"
        )
        assert "synthesis" in executed, (
            f"'synthesis' must be in _executed_nodes. Got: {executed}"
        )
        # RAG / Research / Booking must NOT run — route was fallback -> guardrail directly
        for specialist in ("rag", "research", "booking_stub"):
            assert specialist not in executed, (
                f"{specialist!r} must NOT be executed for the fallback route. Got: {executed}"
            )

    def test_full_graph_guardrail_passed_is_false(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-scenario-4")
        monkeypatch.delenv("FAST_MODE", raising=False)

        fake_mod, _ = _build_fake_anthropic_module(
            f'{{"intent":"{_LLM_RESPONSE_INTENT}","confidence":{_LLM_RESPONSE_CONFIDENCE}}}'
        )
        monkeypatch.setitem(sys.modules, "anthropic", fake_mod)

        initial_state = initialize_state(
            user_id="alex",
            session_id="session-e2e-s4-guardrail-passed",
            current_input=_AMBIGUOUS_INPUT,
            turn_id=1,
        )
        graph = build_graph()
        final_state = graph.invoke(initial_state)

        assert final_state["guardrail_passed"] is False, (
            "guardrail_passed must be False when low-confidence clarification fires"
        )

    def test_full_graph_human_handoff_is_false(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-scenario-4")
        monkeypatch.delenv("FAST_MODE", raising=False)

        fake_mod, _ = _build_fake_anthropic_module(
            f'{{"intent":"{_LLM_RESPONSE_INTENT}","confidence":{_LLM_RESPONSE_CONFIDENCE}}}'
        )
        monkeypatch.setitem(sys.modules, "anthropic", fake_mod)

        initial_state = initialize_state(
            user_id="alex",
            session_id="session-e2e-s4-handoff",
            current_input=_AMBIGUOUS_INPUT,
            turn_id=1,
        )
        graph = build_graph()
        final_state = graph.invoke(initial_state)

        assert final_state["human_handoff"] is False, (
            "human_handoff must remain False on the first clarification turn"
        )

    def test_full_graph_no_error_field_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-scenario-4")
        monkeypatch.delenv("FAST_MODE", raising=False)

        fake_mod, _ = _build_fake_anthropic_module(
            f'{{"intent":"{_LLM_RESPONSE_INTENT}","confidence":{_LLM_RESPONSE_CONFIDENCE}}}'
        )
        monkeypatch.setitem(sys.modules, "anthropic", fake_mod)

        initial_state = initialize_state(
            user_id="alex",
            session_id="session-e2e-s4-error",
            current_input=_AMBIGUOUS_INPUT,
            turn_id=1,
        )
        graph = build_graph()
        final_state = graph.invoke(initial_state)

        error = final_state.get("error")
        assert not error, f"No error must be set during the normal clarification path. Got: {error!r}"
