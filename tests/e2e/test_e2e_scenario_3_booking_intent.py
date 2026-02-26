"""
E2E Test — Scenario 3: Booking Intent (booking_stub path)

Scenario:
    User asks "Book the Wyndham in Bali for March"
    → Dispatcher routes to BOOKING_STUB via Stage-1 pattern match
    → Booking Agent returns a deterministic integration placeholder
    → Response Synthesis extracts and formats the booking payload

Coverage:
    - routing_rules.yaml booking pattern match and confidence score
    - DispatcherAgent Stage-1 pre-filter resolves route="booking_stub"
    - Stage-2 LLM is NOT called (confidence 0.95 >= escalation_threshold 0.72)
    - BookingAgent returns a BookingAgentResponse with integration_point field
    - ResponseSynthesisAgent extracts message, integration_point, required_env_vars
    - Final current_response contains "Booking is not available"
    - Final current_response contains BedrockBookingAPI integration_point
    - No error state set; human_handoff remains False
    - Executed node sequence: dispatcher → booking_stub → guardrail → synthesis
"""
from __future__ import annotations

import re
from typing import Any

import pytest

from concierge.agents.booking_agent import BookingAgent, BookingAgentResponse
from concierge.agents.dispatcher import DispatcherAgent
from concierge.agents.response_synthesis import ResponseSynthesisAgent
from concierge.graph import compiled_graph
from concierge.state import NodeName, initialize_state


# ---------------------------------------------------------------------------
# Constants mirrored from production code — kept here so a drift in source
# immediately breaks these assertions rather than silently passing.
# ---------------------------------------------------------------------------

BOOKING_MESSAGE = (
    "Booking is not available in this version. "
    "A human travel specialist can assist - shall I connect you?"
)
BOOKING_INTEGRATION_POINT = "Replace with BedrockBookingAPI(region=X, api_key=...)"
BOOKING_REQUIRED_ENV_VARS = ["BOOKING_API_KEY", "BOOKING_REGION"]

# Canonical user utterance for Scenario 3
SCENARIO_INPUT = "Book the Wyndham in Bali for March"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_state(current_input: str = SCENARIO_INPUT, **overrides: Any) -> dict[str, Any]:
    """Return a mutable ConciergeState dict seeded for this scenario."""
    state = dict(
        initialize_state(
            user_id="alex",
            session_id="e2e-scenario-3-booking",
            current_input=current_input,
            turn_id=1,
        )
    )
    state.update(overrides)
    return state


# ===========================================================================
# Part 1 — Dispatcher: pattern matching and routing
# ===========================================================================


class TestDispatcherBookingPatternMatch:
    """Dispatcher Stage-1 must pattern-match booking keywords without LLM."""

    def test_booking_rule_exists_in_routing_config(self) -> None:
        """routing_rules.yaml must contain a booking_intent rule pointing to booking_stub."""
        agent = DispatcherAgent()
        booking_rules = [r for r in agent.routing_rules if r.route == "booking_stub"]
        assert booking_rules, "No rule with route='booking_stub' found in routing_rules.yaml"

    def test_booking_rule_confidence_is_0_95(self) -> None:
        """Booking rule confidence must be 0.95 as documented in routing_rules.yaml."""
        agent = DispatcherAgent()
        booking_rule = next(r for r in agent.routing_rules if r.route == "booking_stub")
        assert booking_rule.confidence == pytest.approx(0.95), (
            f"Expected confidence 0.95, got {booking_rule.confidence}"
        )

    def test_booking_rule_intent_is_booking_intent(self) -> None:
        """The matched intent name must be 'booking_intent'."""
        agent = DispatcherAgent()
        booking_rule = next(r for r in agent.routing_rules if r.route == "booking_stub")
        assert booking_rule.intent == "booking_intent"

    @pytest.mark.parametrize(
        "user_input",
        [
            "Book the Wyndham in Bali for March",
            "I want to book a hotel in Phuket",
            "Can you help with booking?",
            "I'd like to make a reservation at the Marriott",
            "reserve a room for two nights",
            "Reservation for March 15th please",
            "BOOKING confirmation needed",   # case-insensitive
            "Reserve suite in Bali",
        ],
    )
    def test_booking_keyword_matches_pattern(self, user_input: str) -> None:
        """All booking-intent keyword variants must match the routing rule pattern."""
        agent = DispatcherAgent()
        booking_rule = next(r for r in agent.routing_rules if r.route == "booking_stub")
        assert booking_rule.pattern.search(user_input) is not None, (
            f"Pattern did not match: '{user_input}'"
        )

    def test_non_booking_input_does_not_match_booking_pattern(self) -> None:
        """Unrelated queries must not match the booking pattern."""
        agent = DispatcherAgent()
        booking_rule = next(r for r in agent.routing_rules if r.route == "booking_stub")
        assert booking_rule.pattern.search("What is the weather in Bali?") is None

    def test_dispatcher_routes_scenario_3_input_to_booking_stub(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Full DispatcherAgent.run() must resolve route='booking_stub' for scenario input."""
        stage2_called = []

        def _fake_stage2(
            self: DispatcherAgent, current_input: str
        ) -> tuple[str | None, float | None, str | None]:
            stage2_called.append(current_input)
            return None, None, None

        monkeypatch.setattr(DispatcherAgent, "_evaluate_stage2", _fake_stage2)
        monkeypatch.setattr("concierge.agents.dispatcher.trace", lambda *a, **kw: None)

        state = _make_state()
        update = DispatcherAgent().run(state)

        assert update["intent"] == "booking_intent"
        assert update["confidence"] == pytest.approx(0.95)
        assert update["route"] == "booking_stub"
        # Stage-2 LLM must NOT be called because 0.95 >= escalation_threshold 0.72
        assert stage2_called == [], (
            "Stage-2 LLM was invoked unexpectedly; Stage-1 should have resolved the route"
        )

    def test_dispatcher_emits_pre_filter_trace_for_booking(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Dispatcher must emit a 'pre_filter' stage trace for Stage-1 booking resolution."""
        traces: list[tuple[str, dict[str, object]]] = []

        def _capture(node_name: str, outcome: str | None = None, **fields: object) -> None:
            traces.append((node_name, dict(fields)))

        monkeypatch.setattr("concierge.agents.dispatcher.trace", _capture)
        monkeypatch.setattr(
            DispatcherAgent,
            "_evaluate_stage2",
            lambda self, _: (None, None, None),
        )

        state = _make_state()
        DispatcherAgent().run(state)

        dispatcher_traces = [f for node, f in traces if node == "dispatcher"]
        assert len(dispatcher_traces) == 1
        t = dispatcher_traces[0]
        assert t["stage"] == "pre_filter"
        assert t["intent"] == "booking_intent"
        assert t["route"] == "booking_stub"
        assert t["confidence"] == pytest.approx(0.95)

    def test_dispatcher_resets_stale_turn_state_before_routing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Dispatcher must wipe stale per-turn fields even for the booking path."""
        monkeypatch.setattr("concierge.agents.dispatcher.trace", lambda *a, **kw: None)
        monkeypatch.setattr(
            DispatcherAgent,
            "_evaluate_stage2",
            lambda self, _: (None, None, None),
        )

        state = _make_state(
            rag_results=[{"id": "stale"}],
            research_results=[{"id": "stale-web"}],
            source_attribution=["[RAG] stale"],
            clarification_needed=True,
            clarification_question="stale?",
            human_handoff=True,
            error="stale error",
        )

        update = DispatcherAgent().run(state)

        assert update["rag_results"] is None
        assert update["research_results"] is None
        assert update["source_attribution"] == []
        assert update["clarification_needed"] is False
        assert update["clarification_question"] is None
        assert update["human_handoff"] is False
        assert update["error"] is None

    def test_dispatcher_appends_user_turn_to_conversation_history(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Dispatcher must append the current user input to conversation_history."""
        monkeypatch.setattr("concierge.agents.dispatcher.trace", lambda *a, **kw: None)
        monkeypatch.setattr(
            DispatcherAgent,
            "_evaluate_stage2",
            lambda self, _: (None, None, None),
        )

        state = _make_state()
        state["conversation_history"] = [{"role": "assistant", "content": "Welcome!"}]

        update = DispatcherAgent().run(state)

        history = update["conversation_history"]
        assert history[-1] == {"role": "user", "content": SCENARIO_INPUT}


# ===========================================================================
# Part 2 — BookingAgent: stub response contract
# ===========================================================================


class TestBookingAgentStubContract:
    """BookingAgent must return a deterministic stub response without external calls."""

    def test_booking_agent_response_model_has_required_fields(self) -> None:
        """BookingAgentResponse Pydantic model must expose all required fields."""
        fields = set(BookingAgentResponse.model_fields.keys())
        assert fields == {"status", "message", "integration_point", "required_env_vars"}

    def test_booking_agent_run_returns_unavailable_status(self) -> None:
        """BookingAgent.run() must set status='unavailable'."""
        state = _make_state()
        update = BookingAgent().run(state)
        assert update["current_response"].status == "unavailable"

    def test_booking_agent_message_contains_not_available(self) -> None:
        """Booking message must include the canonical unavailability phrase."""
        state = _make_state()
        update = BookingAgent().run(state)
        assert "Booking is not available" in update["current_response"].message

    def test_booking_agent_message_offers_human_specialist(self) -> None:
        """Booking stub must offer a human specialist connection."""
        state = _make_state()
        update = BookingAgent().run(state)
        assert "human travel specialist" in update["current_response"].message

    def test_booking_agent_integration_point_references_bedrock_api(self) -> None:
        """integration_point must reference BedrockBookingAPI as the contract."""
        state = _make_state()
        update = BookingAgent().run(state)
        assert "BedrockBookingAPI" in update["current_response"].integration_point

    def test_booking_agent_integration_point_exact_value(self) -> None:
        """integration_point must match the exact stub contract string."""
        state = _make_state()
        update = BookingAgent().run(state)
        assert update["current_response"].integration_point == BOOKING_INTEGRATION_POINT

    def test_booking_agent_required_env_vars_contract(self) -> None:
        """required_env_vars must list BOOKING_API_KEY and BOOKING_REGION."""
        state = _make_state()
        update = BookingAgent().run(state)
        assert update["current_response"].required_env_vars == BOOKING_REQUIRED_ENV_VARS

    def test_booking_agent_returns_full_message_verbatim(self) -> None:
        """BookingAgent message must match the exact stub string."""
        state = _make_state()
        update = BookingAgent().run(state)
        assert update["current_response"].message == BOOKING_MESSAGE

    def test_booking_agent_emits_trace_with_correct_event(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """BookingAgent must emit exactly one trace event='unavailable'."""
        import concierge.agents.booking_agent as booking_module

        trace_calls: list[tuple[str, dict[str, object]]] = []
        monkeypatch.setattr(
            booking_module,
            "trace",
            lambda node_name, outcome=None, **fields: trace_calls.append(
                (node_name, dict(fields))
            ),
        )

        state = _make_state()
        BookingAgent().run(state)

        assert trace_calls == [
            ("booking_agent", {"event": "unavailable", "message": "stub_integration_contract"})
        ]

    def test_booking_agent_is_deterministic_across_invocations(self) -> None:
        """Stub must return identical output regardless of input content variation."""
        state_a = _make_state(current_input="Book the Wyndham in Bali for March")
        state_b = _make_state(current_input="reserve a room at Marriott next week")
        state_c = _make_state(current_input="reservation for two")

        resp_a = BookingAgent().run(state_a)["current_response"]
        resp_b = BookingAgent().run(state_b)["current_response"]
        resp_c = BookingAgent().run(state_c)["current_response"]

        assert resp_a.message == resp_b.message == resp_c.message
        assert resp_a.integration_point == resp_b.integration_point == resp_c.integration_point
        assert resp_a.required_env_vars == resp_b.required_env_vars == resp_c.required_env_vars

    def test_booking_agent_response_is_pydantic_model_instance(self) -> None:
        """current_response must be a BookingAgentResponse Pydantic model instance."""
        state = _make_state()
        update = BookingAgent().run(state)
        assert isinstance(update["current_response"], BookingAgentResponse)

    def test_booking_agent_does_not_set_error_field(self) -> None:
        """BookingAgent.run() must not set an error field in its update."""
        state = _make_state()
        update = BookingAgent().run(state)
        # BookingAgent returns only {'current_response': ...}
        assert "error" not in update or update.get("error") is None

    def test_booking_agent_does_not_set_human_handoff(self) -> None:
        """BookingAgent.run() must not set human_handoff in its direct update."""
        state = _make_state()
        update = BookingAgent().run(state)
        assert update.get("human_handoff") is None or update.get("human_handoff") is False


# ===========================================================================
# Part 3 — ResponseSynthesisAgent: booking payload extraction
# ===========================================================================


class TestResponseSynthesisBookingPayloadExtraction:
    """ResponseSynthesisAgent must extract and format the BookingAgentResponse payload."""

    def _booking_response_object(self) -> BookingAgentResponse:
        return BookingAgentResponse(
            status="unavailable",
            message=BOOKING_MESSAGE,
            integration_point=BOOKING_INTEGRATION_POINT,
            required_env_vars=BOOKING_REQUIRED_ENV_VARS,
        )

    def test_synthesis_extracts_message_from_pydantic_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Synthesis must read .message from BookingAgentResponse attribute access."""
        import concierge.agents.response_synthesis as module

        monkeypatch.setattr(module, "trace", lambda *a, **kw: None)

        state = _make_state()
        state["route"] = "booking_stub"
        state["current_response"] = self._booking_response_object()

        update = ResponseSynthesisAgent().run(state)

        assert "Booking is not available" in update["current_response"]

    def test_synthesis_appends_integration_point_to_response(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Synthesis must append 'Integration point: ...' to the booking response."""
        import concierge.agents.response_synthesis as module

        monkeypatch.setattr(module, "trace", lambda *a, **kw: None)

        state = _make_state()
        state["route"] = "booking_stub"
        state["current_response"] = self._booking_response_object()

        update = ResponseSynthesisAgent().run(state)

        assert "Integration point: Replace with BedrockBookingAPI" in update["current_response"]

    def test_synthesis_appends_required_env_vars_to_response(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Synthesis must append 'Required env vars: ...' to the booking response."""
        import concierge.agents.response_synthesis as module

        monkeypatch.setattr(module, "trace", lambda *a, **kw: None)

        state = _make_state()
        state["route"] = "booking_stub"
        state["current_response"] = self._booking_response_object()

        update = ResponseSynthesisAgent().run(state)

        assert "Required env vars: BOOKING_API_KEY, BOOKING_REGION." in update["current_response"]

    def test_synthesis_extracts_booking_payload_from_dict_representation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Synthesis must also handle current_response as a plain dict (dict path)."""
        import concierge.agents.response_synthesis as module

        monkeypatch.setattr(module, "trace", lambda *a, **kw: None)

        state = _make_state()
        state["route"] = "booking_stub"
        state["current_response"] = {
            "status": "unavailable",
            "message": BOOKING_MESSAGE,
            "integration_point": BOOKING_INTEGRATION_POINT,
            "required_env_vars": BOOKING_REQUIRED_ENV_VARS,
        }

        update = ResponseSynthesisAgent().run(state)

        assert "Booking is not available" in update["current_response"]
        assert "BedrockBookingAPI" in update["current_response"]

    def test_synthesis_skips_booking_path_when_route_is_not_booking_stub(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Synthesis must NOT use booking path when route is 'rag'."""
        import concierge.agents.response_synthesis as module

        monkeypatch.setattr(module, "trace", lambda *a, **kw: None)

        state = _make_state()
        state["route"] = "rag"
        state["current_response"] = self._booking_response_object()
        # No rag_results or research_results — synthesis returns existing response
        state["rag_results"] = None
        state["research_results"] = None

        update = ResponseSynthesisAgent().run(state)

        # When route != booking_stub, synthesis falls through to non-booking path.
        # With no rag/research results and a non-string current_response it returns {}.
        # The key assertion: "Integration point:" must NOT be injected.
        current = update.get("current_response", "")
        assert "Integration point:" not in str(current)

    def test_synthesis_returns_empty_source_attribution_for_booking(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Booking synthesis must return an empty source_attribution list."""
        import concierge.agents.response_synthesis as module

        monkeypatch.setattr(module, "trace", lambda *a, **kw: None)

        state = _make_state()
        state["route"] = "booking_stub"
        state["current_response"] = self._booking_response_object()

        update = ResponseSynthesisAgent().run(state)

        assert update.get("source_attribution") == []

    def test_synthesis_emits_trace_on_booking_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Synthesis must emit a 'synthesis_complete' trace with sources_cited=0."""
        import concierge.agents.response_synthesis as module

        trace_calls: list[tuple[str, dict[str, Any]]] = []
        monkeypatch.setattr(
            module,
            "trace",
            lambda node_name, outcome=None, **fields: trace_calls.append(
                (node_name, dict(fields))
            ),
        )

        state = _make_state()
        state["route"] = "booking_stub"
        state["current_response"] = self._booking_response_object()

        ResponseSynthesisAgent().run(state)

        assert ("response_synthesis", {"event": "synthesis_complete", "sources_cited": 0}) in (
            trace_calls
        )


# ===========================================================================
# Part 4 — Full graph E2E: Scenario 3 end-to-end
# ===========================================================================


class TestE2EScenario3BookingIntent:
    """
    Full compiled graph invocation for Scenario 3.

    Input:  "Book the Wyndham in Bali for March"
    Route:  dispatcher → booking_stub → guardrail → synthesis
    Output: current_response contains unavailability message + integration contract
    """

    def test_e2e_graph_routes_booking_input_to_booking_stub_node(self) -> None:
        """Graph must set route='booking_stub' in state for the scenario input."""
        state = _make_state()
        result = compiled_graph.invoke(state)
        assert result["route"] == "booking_stub"

    def test_e2e_graph_sets_intent_to_booking_intent(self) -> None:
        """Graph must resolve intent='booking_intent' for the scenario input."""
        state = _make_state()
        result = compiled_graph.invoke(state)
        assert result["intent"] == "booking_intent"

    def test_e2e_graph_confidence_is_0_95(self) -> None:
        """Graph must record confidence=0.95 from the Stage-1 booking rule."""
        state = _make_state()
        result = compiled_graph.invoke(state)
        assert result["confidence"] == pytest.approx(0.95)

    def test_e2e_current_response_contains_booking_unavailable_message(self) -> None:
        """current_response must contain the canonical unavailability message."""
        state = _make_state()
        result = compiled_graph.invoke(state)
        assert "Booking is not available" in result["current_response"]

    def test_e2e_current_response_contains_bedrock_booking_api(self) -> None:
        """current_response must reference BedrockBookingAPI as the integration_point."""
        state = _make_state()
        result = compiled_graph.invoke(state)
        assert "BedrockBookingAPI" in result["current_response"]

    def test_e2e_current_response_contains_integration_point_label(self) -> None:
        """current_response must include the 'Integration point:' label injected by synthesis."""
        state = _make_state()
        result = compiled_graph.invoke(state)
        assert "Integration point:" in result["current_response"]

    def test_e2e_current_response_contains_exact_integration_point(self) -> None:
        """current_response must contain the exact integration_point contract string."""
        state = _make_state()
        result = compiled_graph.invoke(state)
        assert (
            f"Integration point: {BOOKING_INTEGRATION_POINT}."
            in result["current_response"]
        )

    def test_e2e_current_response_contains_required_env_vars(self) -> None:
        """current_response must list the required env vars from the booking contract."""
        state = _make_state()
        result = compiled_graph.invoke(state)
        assert "Required env vars: BOOKING_API_KEY, BOOKING_REGION." in result["current_response"]

    def test_e2e_no_error_state_set(self) -> None:
        """error field must be None — no exception should occur on the booking path."""
        state = _make_state()
        result = compiled_graph.invoke(state)
        assert result.get("error") is None

    def test_e2e_human_handoff_remains_false(self) -> None:
        """human_handoff must remain False — booking stub is graceful, not an escalation."""
        state = _make_state()
        result = compiled_graph.invoke(state)
        assert result["human_handoff"] is False

    def test_e2e_guardrail_passed_is_true(self) -> None:
        """guardrail_passed must be True — confidence 0.95 is above dispatcher threshold 0.75."""
        state = _make_state()
        result = compiled_graph.invoke(state)
        assert result["guardrail_passed"] is True

    def test_e2e_clarification_not_triggered(self) -> None:
        """clarification_needed must be False — high confidence booking needs no clarification."""
        state = _make_state()
        result = compiled_graph.invoke(state)
        assert result["clarification_needed"] is False

    def test_e2e_node_execution_sequence_is_correct(self) -> None:
        """Executed node sequence must be: dispatcher → booking_stub → guardrail → synthesis."""
        state = _make_state()
        result = compiled_graph.invoke(state)
        assert result.get("_executed_nodes") == [
            NodeName.DISPATCHER,    # "dispatcher"
            NodeName.BOOKING,       # "booking_stub"
            NodeName.GUARDRAIL,     # "guardrail"
            NodeName.SYNTHESIS,     # "synthesis"
        ]

    def test_e2e_followup_node_not_executed_on_booking_path(self) -> None:
        """FollowUp node must NOT execute — it is conditioned on the research route only."""
        state = _make_state()
        result = compiled_graph.invoke(state)
        assert NodeName.FOLLOWUP not in (result.get("_executed_nodes") or [])

    def test_e2e_rag_node_not_executed_on_booking_path(self) -> None:
        """RAG node must NOT execute — dispatcher routes directly to booking_stub."""
        state = _make_state()
        result = compiled_graph.invoke(state)
        assert NodeName.RAG not in (result.get("_executed_nodes") or [])

    def test_e2e_research_node_not_executed_on_booking_path(self) -> None:
        """Research node must NOT execute — dispatcher routes directly to booking_stub."""
        state = _make_state()
        result = compiled_graph.invoke(state)
        assert NodeName.RESEARCH not in (result.get("_executed_nodes") or [])

    def test_e2e_source_attribution_is_empty_for_booking(self) -> None:
        """source_attribution must be empty — no KB or web results on the booking path."""
        state = _make_state()
        result = compiled_graph.invoke(state)
        assert result.get("source_attribution") == []

    def test_e2e_conversation_history_has_user_turn_appended(self) -> None:
        """Dispatcher must have appended the user input to conversation_history."""
        state = _make_state()
        result = compiled_graph.invoke(state)
        history = result.get("conversation_history", [])
        user_turns = [m for m in history if m.get("role") == "user"]
        assert any(m["content"] == SCENARIO_INPUT for m in user_turns)

    @pytest.mark.parametrize(
        "booking_input",
        [
            "Book the Wyndham in Bali for March",     # canonical scenario
            "reserve a suite at the Grand Hyatt",     # 'reserve' keyword
            "I need to make a reservation",           # 'reservation' keyword
            "booking for two at Hotel Mulia",         # 'booking' keyword
        ],
    )
    def test_e2e_all_booking_keyword_variants_produce_unavailable_response(
        self, booking_input: str
    ) -> None:
        """All booking keyword variants must produce a consistent unavailability response."""
        state = _make_state(current_input=booking_input)
        result = compiled_graph.invoke(state)

        assert result["route"] == "booking_stub", (
            f"Expected route='booking_stub' for input: '{booking_input}', "
            f"got route='{result['route']}'"
        )
        assert "Booking is not available" in result["current_response"], (
            f"Expected unavailability message for input: '{booking_input}'"
        )
        assert result.get("error") is None
        assert result["human_handoff"] is False

    def test_e2e_response_is_a_string_after_synthesis(self) -> None:
        """After synthesis, current_response must be a plain str, not a Pydantic model."""
        state = _make_state()
        result = compiled_graph.invoke(state)
        assert isinstance(result["current_response"], str), (
            f"current_response is {type(result['current_response']).__name__}, expected str"
        )

    def test_e2e_response_does_not_contain_raw_pydantic_repr(self) -> None:
        """current_response must not leak Pydantic model repr (e.g. 'status=')."""
        state = _make_state()
        result = compiled_graph.invoke(state)
        # Pydantic model __repr__ includes "status=" in the output
        assert "status='unavailable'" not in result["current_response"]
        assert "required_env_vars=" not in result["current_response"]


# ===========================================================================
# Part 5 — Regression: route pre-set skips dispatcher re-evaluation
# ===========================================================================


class TestDispatcherSkipsWhenRouteAlreadySet:
    """If route is already set in state, dispatcher must not re-evaluate."""

    def test_dispatcher_is_no_op_when_route_pre_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """DispatcherAgent.run() must return {} when state already has a non-empty route."""
        stage1_called = []

        monkeypatch.setattr(
            DispatcherAgent,
            "_evaluate_stage1",
            lambda self, text: stage1_called.append(text) or (None, 0.0, None),
        )

        state = _make_state()
        state["route"] = "booking_stub"   # pre-set to simulate graph re-entry

        update = DispatcherAgent().run(state)

        assert update == {}, (
            "Dispatcher must return empty dict when route is already set"
        )
        assert stage1_called == [], "Stage-1 must not be called when route is pre-set"

    def test_graph_with_pre_set_booking_stub_route_skips_dispatcher_routing(self) -> None:
        """Graph invocation with route='booking_stub' pre-set must bypass Stage-1 evaluation."""
        state = _make_state()
        state["route"] = "booking_stub"

        result = compiled_graph.invoke(state)

        assert result["route"] == "booking_stub"
        assert "Booking is not available" in result["current_response"]
        assert result.get("error") is None
