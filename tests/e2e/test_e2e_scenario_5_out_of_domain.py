"""
E2E Scenario 5: Out-of-Domain Query Deflection
================================================
User asks "What's the weather in Hawaii?" and the system must:
  1. Stage-1 regex pattern match on "weather" keyword (no LLM call).
  2. Resolve intent=out_of_domain, route=fallback, confidence=0.3.
  3. Dispatcher routes directly to Guardrail (fallback path skips specialist nodes).
  4. Guardrail detects out_of_domain and emits a deflection response.
  5. guardrail_passed=False, human_handoff=False.
  6. current_response contains "specialize in travel".

Covers:
  - routing_rules.yaml weather pattern (score=0.3, below escalation_threshold=0.72)
  - Stage-1 pre-filter fires but score < escalation_threshold → returns intent + confidence
    with route=None, then _evaluate_stage2 is attempted; because ANTHROPIC_API_KEY is absent
    in the test environment the stage-2 call is a no-op and the dispatcher falls back to the
    stage-1 values (intent=out_of_domain, confidence=0.3, route=None → coerced to "fallback"
    by the graph runner).
  - Guardrail out_of_domain path → guardrail_passed=False
  - Synthesis preserves the deflection current_response set by Guardrail
  - No Anthropic API call is issued (no mock needed for the happy path)
  - Node execution order: dispatcher → guardrail → synthesis (followup skipped)
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml

from concierge.agents.dispatcher import DispatcherAgent
from concierge.agents.guardrail import (
    OUT_OF_DOMAIN_CONFIDENCE_MAX,
    OUT_OF_DOMAIN_DEFLECTION_TEMPLATE,
    GuardrailAgent,
)
from concierge.graph import compiled_graph
from concierge.state import NodeName, initialize_state

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

USER_INPUT = "What's the weather in Hawaii?"

REPO_ROOT = Path(__file__).resolve().parents[2]
ROUTING_RULES_PATH = REPO_ROOT / "config" / "routing_rules.yaml"

EXPECTED_INTENT = "out_of_domain"
EXPECTED_ROUTE = "fallback"
EXPECTED_STAGE1_SCORE = 0.3
EXPECTED_ESCALATION_THRESHOLD = 0.72

# The deflection message must mention travel specialisation.
DEFLECTION_KEYWORD = "specialize in travel"


# ---------------------------------------------------------------------------
# Helper: load routing rules directly for white-box assertions
# ---------------------------------------------------------------------------

def _load_routing_rules() -> dict[str, Any]:
    with ROUTING_RULES_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)  # type: ignore[return-value]


# ===========================================================================
# GROUP 1 — Routing rules contract
# ===========================================================================


class TestRoutingRulesContract:
    """Verify the routing_rules.yaml out-of-domain entry is structurally correct."""

    def setup_method(self) -> None:
        self.rules_data = _load_routing_rules()

    def test_escalation_threshold_is_above_out_of_domain_score(self) -> None:
        threshold = float(self.rules_data["escalation_threshold"])
        assert threshold == EXPECTED_ESCALATION_THRESHOLD, (
            f"escalation_threshold must be {EXPECTED_ESCALATION_THRESHOLD}, got {threshold}"
        )

    def test_out_of_domain_rule_exists(self) -> None:
        rules = self.rules_data["rules"]
        ood_rules = [r for r in rules if r.get("intent") == EXPECTED_INTENT]
        assert len(ood_rules) >= 1, "No out_of_domain rule found in routing_rules.yaml"

    def test_out_of_domain_score_is_below_escalation_threshold(self) -> None:
        rules = self.rules_data["rules"]
        ood_rule = next(r for r in rules if r.get("intent") == EXPECTED_INTENT)
        score = float(ood_rule["score"])
        threshold = float(self.rules_data["escalation_threshold"])
        assert score < threshold, (
            f"out_of_domain score ({score}) must be below escalation_threshold ({threshold})"
        )
        assert score == EXPECTED_STAGE1_SCORE

    def test_out_of_domain_route_is_fallback(self) -> None:
        rules = self.rules_data["rules"]
        ood_rule = next(r for r in rules if r.get("intent") == EXPECTED_INTENT)
        assert ood_rule["route"] == EXPECTED_ROUTE

    def test_out_of_domain_pattern_covers_weather(self) -> None:
        rules = self.rules_data["rules"]
        ood_rule = next(r for r in rules if r.get("intent") == EXPECTED_INTENT)
        pattern = re.compile(ood_rule["pattern"], flags=re.IGNORECASE)
        assert pattern.search(USER_INPUT) is not None, (
            f"Pattern '{ood_rule['pattern']}' did not match '{USER_INPUT}'"
        )

    def test_out_of_domain_pattern_covers_all_declared_keywords(self) -> None:
        """Regression guard: weather, stock, crypto, politics all hit the same rule."""
        rules = self.rules_data["rules"]
        ood_rule = next(r for r in rules if r.get("intent") == EXPECTED_INTENT)
        pattern = re.compile(ood_rule["pattern"], flags=re.IGNORECASE)
        for keyword in ("weather", "stock", "crypto", "politics"):
            assert pattern.search(keyword) is not None, (
                f"Pattern '{ood_rule['pattern']}' did not match keyword '{keyword}'"
            )


# ===========================================================================
# GROUP 2 — Dispatcher Stage-1 pre-filter (unit, no LLM)
# ===========================================================================


class TestDispatcherStage1PreFilter:
    """Verify DispatcherAgent._evaluate_stage1 classifies the weather query correctly."""

    def setup_method(self) -> None:
        self.agent = DispatcherAgent()

    def test_stage1_detects_weather_keyword(self) -> None:
        intent, confidence, route = self.agent._evaluate_stage1(USER_INPUT)
        assert intent == EXPECTED_INTENT, f"Expected intent={EXPECTED_INTENT!r}, got {intent!r}"

    def test_stage1_confidence_equals_rule_score(self) -> None:
        _intent, confidence, _route = self.agent._evaluate_stage1(USER_INPUT)
        assert confidence == EXPECTED_STAGE1_SCORE, (
            f"Expected confidence={EXPECTED_STAGE1_SCORE}, got {confidence}"
        )

    def test_stage1_returns_none_route_because_score_below_threshold(self) -> None:
        """
        Stage-1 returns route=None when score < escalation_threshold.
        The dispatcher then attempts Stage-2 LLM escalation; with no API key
        stage-2 is a no-op so route remains None and the graph coerces it to
        "fallback" at runtime.
        """
        _intent, _confidence, route = self.agent._evaluate_stage1(USER_INPUT)
        assert route is None, (
            "Stage-1 must NOT short-circuit to a route when score < escalation_threshold; "
            f"got route={route!r}"
        )

    def test_no_anthropic_call_when_api_key_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With ANTHROPIC_API_KEY unset, _evaluate_stage2 must return (None, None, None)."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        state = initialize_state("alex", "e2e-s5-no-key", USER_INPUT, turn_id=1)
        mock_anthropic = MagicMock()
        with patch("anthropic.Anthropic", return_value=mock_anthropic) as mock_cls:
            result = self.agent.run(state)
            mock_cls.assert_not_called()
        # Dispatcher still populates intent from Stage-1 partial match
        assert result.get("intent") == EXPECTED_INTENT

    def test_dispatcher_run_sets_intent_out_of_domain(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        state = initialize_state("alex", "e2e-s5-dispatcher", USER_INPUT, turn_id=1)
        update = DispatcherAgent().run(state)
        assert update.get("intent") == EXPECTED_INTENT

    def test_dispatcher_run_sets_confidence_0_3(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        state = initialize_state("alex", "e2e-s5-dispatcher-conf", USER_INPUT, turn_id=1)
        update = DispatcherAgent().run(state)
        assert update.get("confidence") == EXPECTED_STAGE1_SCORE


# ===========================================================================
# GROUP 3 — Guardrail unit: out_of_domain detection and deflection
# ===========================================================================


class TestGuardrailOutOfDomainUnit:
    """Unit tests for GuardrailAgent given pre-classified out_of_domain state."""

    def _make_state(
        self,
        *,
        intent: str = EXPECTED_INTENT,
        confidence: float = EXPECTED_STAGE1_SCORE,
        input_text: str = USER_INPUT,
    ) -> dict[str, Any]:
        state = initialize_state("alex", "e2e-s5-guardrail", input_text, turn_id=1)
        state["intent"] = intent
        state["confidence"] = confidence
        state["route"] = EXPECTED_ROUTE
        return state

    def test_guardrail_returns_guardrail_passed_false(self) -> None:
        state = self._make_state()
        update = GuardrailAgent().run(state)
        assert update["guardrail_passed"] is False

    def test_guardrail_does_not_set_clarification_needed(self) -> None:
        state = self._make_state()
        update = GuardrailAgent().run(state)
        assert update.get("clarification_needed") is False

    def test_guardrail_does_not_set_clarification_question(self) -> None:
        state = self._make_state()
        update = GuardrailAgent().run(state)
        assert update.get("clarification_question") is None

    def test_guardrail_does_not_trigger_human_handoff(self) -> None:
        state = self._make_state()
        update = GuardrailAgent().run(state)
        assert update.get("human_handoff") is False

    def test_guardrail_deflection_contains_travel_specialization_phrase(self) -> None:
        state = self._make_state()
        update = GuardrailAgent().run(state)
        response = str(update.get("current_response") or "")
        assert DEFLECTION_KEYWORD in response, (
            f"Expected deflection phrase {DEFLECTION_KEYWORD!r} in response; got: {response!r}"
        )

    def test_guardrail_deflection_references_extracted_location(self) -> None:
        """Hawaii should be extracted and echoed in the deflection message."""
        state = self._make_state()
        update = GuardrailAgent().run(state)
        response = str(update.get("current_response") or "")
        assert "Hawaii" in response, (
            f"Expected 'Hawaii' to appear in deflection; got: {response!r}"
        )

    def test_guardrail_deflection_matches_template_structure(self) -> None:
        """Verify the response matches the expected template prefix."""
        expected_prefix = "I specialize in travel planning and concierge services."
        state = self._make_state()
        update = GuardrailAgent().run(state)
        response = str(update.get("current_response") or "")
        assert response.startswith(expected_prefix), (
            f"Response must start with template prefix.\nExpected prefix: {expected_prefix!r}\n"
            f"Got: {response!r}"
        )

    def test_guardrail_out_of_domain_confidence_constant_is_0_4(self) -> None:
        """Guard against accidental drift of the module-level constant."""
        assert OUT_OF_DOMAIN_CONFIDENCE_MAX == 0.4

    def test_stage1_score_is_below_out_of_domain_confidence_max(self) -> None:
        """0.3 < 0.4 ensures the confidence branch also triggers deflection."""
        assert EXPECTED_STAGE1_SCORE < OUT_OF_DOMAIN_CONFIDENCE_MAX

    def test_guardrail_emits_trace_event_out_of_domain(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from concierge.agents import guardrail as guardrail_module

        trace_calls: list[tuple[str, dict[str, Any]]] = []
        monkeypatch.setattr(
            guardrail_module,
            "trace",
            lambda node, **kw: trace_calls.append((node, kw)),
        )
        state = self._make_state()
        GuardrailAgent().run(state)

        assert len(trace_calls) == 1, f"Expected exactly one trace call, got: {trace_calls}"
        node, fields = trace_calls[0]
        assert node == "guardrail"
        assert fields.get("event") == "out_of_domain"
        assert fields.get("confidence") == round(EXPECTED_STAGE1_SCORE, 2)


# ===========================================================================
# GROUP 4 — Full E2E graph invocation (no LLM, no API key)
# ===========================================================================


class TestE2EGraphScenario5:
    """
    Full pipeline invocation: initialize_state → compiled_graph.invoke.
    No ANTHROPIC_API_KEY is set so Stage-2 is bypassed, giving a pure
    pattern-match path end-to-end.
    """

    @pytest.fixture(autouse=True)
    def _no_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    def _run_graph(self, session_suffix: str = "main") -> dict[str, Any]:
        state = initialize_state(
            "alex",
            f"e2e-s5-graph-{session_suffix}",
            USER_INPUT,
            turn_id=1,
        )
        result: dict[str, Any] = compiled_graph.invoke(state)
        return result

    # --- intent & route ---

    def test_result_intent_is_out_of_domain(self) -> None:
        result = self._run_graph("intent")
        assert result["intent"] == EXPECTED_INTENT, (
            f"intent must be {EXPECTED_INTENT!r}, got {result['intent']!r}"
        )

    def test_result_route_is_fallback(self) -> None:
        """
        After Stage-1 the dispatcher stores route=None for below-threshold scores
        and Stage-2 is skipped (no API key).  The _FallbackCompiledGraph runner
        coerces the missing route to "fallback" when deciding which node sequence
        to execute, but the state field itself remains None.  We therefore assert
        that the route value in the final state is either "fallback" or None (the
        system-level invariant is that the fallback path was taken, which is
        verified separately by the node execution order test).
        """
        result = self._run_graph("route")
        route_value = result.get("route")
        assert route_value in (EXPECTED_ROUTE, None), (
            f"route must be {EXPECTED_ROUTE!r} or None on fallback path, got {route_value!r}"
        )

    def test_result_confidence_is_0_3(self) -> None:
        result = self._run_graph("confidence")
        assert result["confidence"] == EXPECTED_STAGE1_SCORE

    # --- guardrail outcome ---

    def test_guardrail_passed_is_false(self) -> None:
        result = self._run_graph("gp")
        assert result["guardrail_passed"] is False, (
            "guardrail_passed must be False for out_of_domain deflection"
        )

    def test_human_handoff_is_false(self) -> None:
        """Deflection does NOT escalate; human_handoff must remain False."""
        result = self._run_graph("hh")
        assert result["human_handoff"] is False, (
            "human_handoff must be False — out-of-domain deflects, it does not escalate"
        )

    def test_clarification_needed_is_false(self) -> None:
        result = self._run_graph("cn")
        assert result.get("clarification_needed") is False

    def test_clarification_question_is_none(self) -> None:
        result = self._run_graph("cq")
        assert result.get("clarification_question") is None

    # --- response content ---

    def test_current_response_contains_specialize_in_travel(self) -> None:
        result = self._run_graph("resp")
        response = str(result.get("current_response") or "")
        assert DEFLECTION_KEYWORD in response, (
            f"Response must contain {DEFLECTION_KEYWORD!r}.\nGot: {response!r}"
        )

    def test_current_response_is_non_empty(self) -> None:
        result = self._run_graph("nonempty")
        assert str(result.get("current_response") or "").strip(), (
            "current_response must not be empty"
        )

    def test_current_response_contains_hawaii(self) -> None:
        result = self._run_graph("hawaii")
        response = str(result.get("current_response") or "")
        assert "Hawaii" in response, (
            f"Deflection response must mention the extracted location 'Hawaii'.\nGot: {response!r}"
        )

    # --- node execution order ---

    def test_executed_nodes_order_is_dispatcher_guardrail_synthesis(self) -> None:
        result = self._run_graph("nodes")
        executed = result.get("_executed_nodes")
        assert executed == [
            NodeName.DISPATCHER,
            NodeName.GUARDRAIL,
            NodeName.SYNTHESIS,
        ], (
            f"Expected fallback path: dispatcher → guardrail → synthesis.\n"
            f"Got: {executed}"
        )

    def test_followup_node_not_executed(self) -> None:
        result = self._run_graph("nofollowup")
        executed = result.get("_executed_nodes") or []
        assert NodeName.FOLLOWUP not in executed, (
            "followup node must not run on out-of-domain deflection path"
        )

    def test_rag_node_not_executed(self) -> None:
        result = self._run_graph("norag")
        executed = result.get("_executed_nodes") or []
        assert NodeName.RAG not in executed, (
            "rag node must not run on fallback route"
        )

    def test_research_node_not_executed(self) -> None:
        result = self._run_graph("noresearch")
        executed = result.get("_executed_nodes") or []
        assert NodeName.RESEARCH not in executed, (
            "research node must not run on fallback route"
        )

    def test_booking_node_not_executed(self) -> None:
        result = self._run_graph("nobooking")
        executed = result.get("_executed_nodes") or []
        assert NodeName.BOOKING not in executed, (
            "booking node must not run on fallback route"
        )

    # --- no LLM call guard ---

    def test_no_anthropic_client_instantiated(self) -> None:
        """
        Verify the full pipeline does not touch the Anthropic SDK.
        We patch anthropic.Anthropic and assert it is never constructed.
        """
        mock_client = MagicMock()
        with patch("anthropic.Anthropic", return_value=mock_client) as mock_cls:
            state = initialize_state("alex", "e2e-s5-nollm", USER_INPUT, turn_id=1)
            compiled_graph.invoke(state)
            mock_cls.assert_not_called(), (
                "anthropic.Anthropic must NOT be instantiated — pattern match bypasses LLM"
            )


# ===========================================================================
# GROUP 5 — Parametric coverage: all out-of-domain trigger words
# ===========================================================================


WEATHER_VARIANTS = [
    "What's the weather in Hawaii?",
    "weather forecast for Paris",
    "What is the weather like in Bali?",
]

STOCK_VARIANTS = [
    "What's the stock price of Apple?",
    "stock market performance today",
]

CRYPTO_VARIANTS = [
    "How is crypto performing today?",
    # Note: queries mixing "crypto" with high-score trend keywords (news, latest, right now)
    # will be won by the trend_research rule (score 0.82 > 0.3).  Use phrases that only
    # trigger the out_of_domain pattern.
    "bitcoin and crypto prices",
]

POLITICS_VARIANTS = [
    "What is happening in US politics?",
    # Similarly, "latest … news" activates trend_research (0.82) and wins.
    # Use a phrase that only hits the politics keyword.
    "US politics and election updates",
]


@pytest.mark.parametrize(
    "query",
    WEATHER_VARIANTS + STOCK_VARIANTS + CRYPTO_VARIANTS + POLITICS_VARIANTS,
)
def test_stage1_pattern_matches_all_out_of_domain_variants(query: str) -> None:
    """Stage-1 must classify every out-of-domain variant as out_of_domain."""
    agent = DispatcherAgent()
    intent, confidence, _route = agent._evaluate_stage1(query)
    assert intent == EXPECTED_INTENT, (
        f"Expected intent={EXPECTED_INTENT!r} for query {query!r}; got {intent!r}"
    )
    assert confidence == EXPECTED_STAGE1_SCORE


@pytest.mark.parametrize("query", WEATHER_VARIANTS)
def test_guardrail_deflects_all_weather_variants(
    query: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Guardrail must deflect every weather query variant."""
    from concierge.agents import guardrail as guardrail_module

    monkeypatch.setattr(guardrail_module, "trace", lambda *a, **kw: None)
    state = initialize_state("alex", "e2e-s5-param", query, turn_id=1)
    state["intent"] = EXPECTED_INTENT
    state["confidence"] = EXPECTED_STAGE1_SCORE
    state["route"] = EXPECTED_ROUTE

    update = GuardrailAgent().run(state)
    assert update["guardrail_passed"] is False
    assert update.get("human_handoff") is False
    assert DEFLECTION_KEYWORD in str(update.get("current_response") or "")


# ===========================================================================
# GROUP 6 — Negative: in-domain queries must NOT trigger out-of-domain path
# ===========================================================================


IN_DOMAIN_QUERIES = [
    "Book a room at Wyndham Grand",
    "What are the best resorts in Hawaii?",
    "Latest travel trends for 2026",
    "Reserve a suite for next weekend",
]


@pytest.mark.parametrize("query", IN_DOMAIN_QUERIES)
def test_in_domain_queries_do_not_produce_out_of_domain_intent(query: str) -> None:
    """In-domain queries must not be misclassified as out_of_domain by Stage-1."""
    agent = DispatcherAgent()
    intent, _confidence, _route = agent._evaluate_stage1(query)
    assert intent != EXPECTED_INTENT, (
        f"In-domain query {query!r} was incorrectly classified as out_of_domain"
    )
