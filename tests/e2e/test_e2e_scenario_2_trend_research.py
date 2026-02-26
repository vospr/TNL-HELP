"""
E2E Test -- Scenario 2: Trend Research (web search path)

Flow:
    User: "What are the latest beach destinations?"
    Dispatcher  -> route="research"  (Stage-1 pre-filter, score=0.82 > threshold=0.72,
                                       route resolved without LLM escalation)
    ResearchAgent -> search_duckduckgo() -> 5 mocked results
    Guardrail    -> guardrail_passed=True (high confidence)
    ResponseSynthesis -> [Web]-tagged response + source_attribution
    FollowUp     -> proactive_suggestion appended (research-only edge)

Assertions verified:
     1.  Dispatcher sets route="research" via Stage-1 pre-filter
     2.  ResearchAgent receives a scoped query containing the original user input
     3.  search_duckduckgo called with max_results=5
     4.  research_results populated with all 5 mocked DuckDuckGo records
     5.  Parsed shape: href->link, body->snippet field renaming applied
     6.  source_attribution entries prefixed [Web] contain mocked URLs
     7.  degradation_label is None (success path)
     8.  guardrail_passed is True
     9.  FollowUp node fires for research route only
    10.  _executed_nodes == [dispatcher, research, guardrail, synthesis, followup]
    11.  proactive_suggestion == FOLLOWUP_SUGGESTION constant
    12.  FOLLOWUP_SUGGESTION is appended to current_response
    13.  conversation_history grows by one user message per turn
    14.  Smoke: RAG and booking routes do NOT activate FollowUp

Mocking strategy:
    search_duckduckgo is patched on the research_agent module object so
    ResearchAgent.run() picks up the stub without import-order dependency.
    ANTHROPIC_API_KEY is deleted so Stage-2 LLM escalation and LLM ranking
    are suppressed -- no Anthropic SDK call occurs in this scenario.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml

from concierge.agents.dispatcher import DispatcherAgent
from concierge.agents.followup import FOLLOWUP_SUGGESTION, FollowUpAgent
from concierge.agents.research_agent import ResearchAgent
from concierge.graph import compiled_graph
from concierge.state import NodeName, initialize_state

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

USER_QUERY = "What are the latest beach destinations?"

REPO_ROOT = Path(__file__).resolve().parents[2]
ROUTING_RULES_PATH = REPO_ROOT / "config" / "routing_rules.yaml"

EXPECTED_INTENT = "trend_research"
EXPECTED_ROUTE = "research"
EXPECTED_STAGE1_SCORE = 0.82
EXPECTED_ESCALATION_THRESHOLD = 0.72

# Five deterministic DuckDuckGo mock results (title / href / body shape).
MOCK_DDG_RESULTS: list[dict[str, str]] = [
    {
        "title": "Top 10 Trending Beach Destinations 2026",
        "href": "https://travelblog.example.com/trending-beaches-2026",
        "body": "Maldives and Bali top the list for beach lovers this year.",
    },
    {
        "title": "Hidden Beach Gems: Where to Go Right Now",
        "href": "https://wanderlust.example.com/hidden-beaches",
        "body": "Palawan, El Nido, and Siargao are drawing crowds in 2026.",
    },
    {
        "title": "Budget Beach Travel: Trending Spots Under 100 Per Day",
        "href": "https://budgettravel.example.com/beach-budget",
        "body": "Vietnam and Thailand offer world-class beaches at low cost.",
    },
    {
        "title": "Luxury Beach Resorts Trending Among High Earners",
        "href": "https://luxurytravel.example.com/luxury-beaches",
        "body": "Maldives overwater bungalows sold out through Q3 2026.",
    },
    {
        "title": "Eco-Friendly Beach Destinations Growing in Popularity",
        "href": "https://ecotravel.example.com/eco-beaches",
        "body": "Costa Rica and Seychelles lead sustainable beach tourism.",
    },
]

# Shape produced by ResearchAgent._parse_results(): href->link, body->snippet.
EXPECTED_PARSED_RESULTS: list[dict[str, str]] = [
    {"title": r["title"], "link": r["href"], "snippet": r["body"]}
    for r in MOCK_DDG_RESULTS
]

EXPECTED_URLS: list[str] = [r["href"] for r in MOCK_DDG_RESULTS]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _load_routing_rules() -> dict[str, Any]:
    with ROUTING_RULES_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)  # type: ignore[return-value]


def _patch_duckduckgo(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """
    Replace search_duckduckgo on the research_agent module with a stub that
    returns MOCK_DDG_RESULTS and records every call in the returned list.
    """
    from concierge.agents import research_agent as _mod

    call_log: list[dict[str, Any]] = []

    def _fake(query: str, max_results: int = 5) -> list[dict[str, str]]:
        call_log.append({"query": query, "max_results": max_results})
        return list(MOCK_DDG_RESULTS)

    monkeypatch.setattr(_mod, "search_duckduckgo", _fake)
    return call_log  # type: ignore[return-value]


def _fresh_state(
    session_suffix: str = "main",
    turn_id: int = 1,
    current_input: str = USER_QUERY,
) -> dict[str, Any]:
    """Return a fresh ConciergeState dict for the trend-research scenario."""
    return dict(
        initialize_state(
            user_id="alex",
            session_id=f"e2e-s2-{session_suffix}",
            current_input=current_input,
            turn_id=turn_id,
        )
    )


# ===========================================================================
# GROUP 1 -- Routing rules contract (white-box, no graph invocation)
# ===========================================================================


class TestScenario2RoutingRulesContract:
    """Verify routing_rules.yaml covers the trend_research intent correctly."""

    def setup_method(self) -> None:
        self.rules_data = _load_routing_rules()

    def test_trend_research_rule_exists(self) -> None:
        rules = self.rules_data["rules"]
        matches = [r for r in rules if r.get("intent") == EXPECTED_INTENT]
        assert len(matches) >= 1, "No trend_research rule found in routing_rules.yaml"

    def test_trend_research_score_exceeds_escalation_threshold(self) -> None:
        """score=0.82 > threshold=0.72 so Stage-1 short-circuits without LLM."""
        rules = self.rules_data["rules"]
        rule = next(r for r in rules if r.get("intent") == EXPECTED_INTENT)
        score = float(rule["score"])
        threshold = float(self.rules_data["escalation_threshold"])
        assert score > threshold, (
            f"trend_research score ({score}) must exceed escalation_threshold ({threshold})"
        )

    def test_trend_research_score_is_0_82(self) -> None:
        rules = self.rules_data["rules"]
        rule = next(r for r in rules if r.get("intent") == EXPECTED_INTENT)
        assert float(rule["score"]) == pytest.approx(EXPECTED_STAGE1_SCORE)

    def test_trend_research_route_is_research(self) -> None:
        rules = self.rules_data["rules"]
        rule = next(r for r in rules if r.get("intent") == EXPECTED_INTENT)
        assert rule["route"] == EXPECTED_ROUTE

    def test_trend_pattern_matches_user_query(self) -> None:
        rules = self.rules_data["rules"]
        rule = next(r for r in rules if r.get("intent") == EXPECTED_INTENT)
        pat = re.compile(rule["pattern"], flags=re.IGNORECASE)
        assert pat.search(USER_QUERY) is not None, (
            f"Pattern must match USER_QUERY {USER_QUERY!r}"
        )

    def test_trend_pattern_covers_all_declared_keywords(self) -> None:
        """Regression guard: trend, latest, right now, news must all match."""
        rules = self.rules_data["rules"]
        rule = next(r for r in rules if r.get("intent") == EXPECTED_INTENT)
        pat = re.compile(rule["pattern"], flags=re.IGNORECASE)
        for kw in ("trend", "latest", "right now", "news"):
            assert pat.search(kw) is not None, (
                f"Pattern must match keyword {kw!r}"
            )


# ===========================================================================
# GROUP 2 -- Dispatcher Stage-1 pre-filter (unit, no LLM call)
# ===========================================================================


class TestScenario2DispatcherStage1:
    """DispatcherAgent._evaluate_stage1 must classify the trend query correctly."""

    def setup_method(self) -> None:
        self.agent = DispatcherAgent()

    def test_stage1_returns_trend_research_intent(self) -> None:
        intent, _conf, _route = self.agent._evaluate_stage1(USER_QUERY)
        assert intent == EXPECTED_INTENT, f"Expected {EXPECTED_INTENT!r}, got {intent!r}"

    def test_stage1_confidence_equals_rule_score(self) -> None:
        _intent, conf, _route = self.agent._evaluate_stage1(USER_QUERY)
        assert conf == pytest.approx(EXPECTED_STAGE1_SCORE)

    def test_stage1_resolves_route_because_score_exceeds_threshold(self) -> None:
        """score=0.82 > escalation_threshold=0.72 so Stage-1 returns route directly."""
        _intent, _conf, route = self.agent._evaluate_stage1(USER_QUERY)
        assert route == EXPECTED_ROUTE, (
            f"Stage-1 must resolve route={EXPECTED_ROUTE!r}; got {route!r}"
        )

    def test_stage2_not_called_when_stage1_resolves_route(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stage2_calls: list[str] = []

        def _spy(self_inner: DispatcherAgent, q: str) -> tuple[None, None, None]:
            stage2_calls.append(q)
            return None, None, None

        monkeypatch.setattr(DispatcherAgent, "_evaluate_stage2", _spy)
        DispatcherAgent().run(_fresh_state("dispatcher-stage1-spy"))
        assert stage2_calls == [], "Stage-2 must NOT be called when Stage-1 resolves route"

    def test_dispatcher_run_sets_route_to_research(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        update = DispatcherAgent().run(_fresh_state("dispatcher-route"))
        assert update.get("route") == EXPECTED_ROUTE

    def test_dispatcher_run_sets_intent_to_trend_research(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        update = DispatcherAgent().run(_fresh_state("dispatcher-intent"))
        assert update.get("intent") == EXPECTED_INTENT

    def test_dispatcher_appends_user_message_to_conversation_history(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        state = _fresh_state("dispatcher-history")
        state["conversation_history"] = [{"role": "assistant", "content": "Welcome!"}]
        update = DispatcherAgent().run(state)
        history = update.get("conversation_history", [])
        assert {"role": "user", "content": USER_QUERY} in history

    def test_dispatcher_resets_stale_per_turn_fields(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        state = _fresh_state("dispatcher-reset")
        state["rag_results"] = [{"id": "stale"}]
        state["research_results"] = [{"id": "stale-web"}]
        state["source_attribution"] = ["[RAG] stale"]
        state["degradation_label"] = "STALE"
        state["clarification_needed"] = True
        state["human_handoff"] = True
        state["error"] = "old error"
        update = DispatcherAgent().run(state)
        assert update.get("rag_results") is None
        assert update.get("research_results") is None
        assert update.get("source_attribution") == []
        assert update.get("degradation_label") is None
        assert update.get("clarification_needed") is False
        assert update.get("human_handoff") is False
        assert update.get("error") is None


# ===========================================================================
# GROUP 3 -- ResearchAgent unit (mocked DuckDuckGo)
# ===========================================================================


class TestScenario2ResearchAgent:
    """ResearchAgent.run() with search_duckduckgo stubbed out."""

    def test_search_called_once_with_query_containing_user_input(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        call_log = _patch_duckduckgo(monkeypatch)
        ResearchAgent().run(_fresh_state("ra-scoped"))
        assert len(call_log) == 1, f"Expected 1 DDG call, got {len(call_log)}"
        assert USER_QUERY in call_log[0]["query"], (
            f"Scoped query must contain user input. got={call_log[0]['query']!r}"
        )

    def test_search_called_with_max_results_5(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        call_log = _patch_duckduckgo(monkeypatch)
        ResearchAgent().run(_fresh_state("ra-maxresults"))
        assert call_log[0]["max_results"] == 5

    def test_research_results_has_5_entries(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_duckduckgo(monkeypatch)
        update = ResearchAgent().run(_fresh_state("ra-count"))
        assert len(update.get("research_results", [])) == 5

    def test_research_results_parsed_shape_href_to_link_body_to_snippet(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """_parse_results must rename href->link and body->snippet."""
        _patch_duckduckgo(monkeypatch)
        update = ResearchAgent().run(_fresh_state("ra-shape"))
        assert update.get("research_results") == EXPECTED_PARSED_RESULTS, (
            f"Parsed results mismatch.\n"
            f"  got={update.get('research_results')}\n"
            f"  expected={EXPECTED_PARSED_RESULTS}"
        )

    def test_degradation_label_is_none_on_successful_search(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_duckduckgo(monkeypatch)
        update = ResearchAgent().run(_fresh_state("ra-degrad"))
        assert update.get("degradation_label") is None

    def test_research_agent_emits_search_complete_trace(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_duckduckgo(monkeypatch)
        from concierge.agents import research_agent as _mod
        calls: list[tuple[str, dict[str, Any]]] = []
        monkeypatch.setattr(_mod, "trace", lambda node, **kw: calls.append((node, kw)))
        ResearchAgent().run(_fresh_state("ra-trace"))
        assert calls == [("research_agent", {"event": "search_complete", "results_count": 5})]

    def test_all_expected_urls_present_as_link_fields(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_duckduckgo(monkeypatch)
        update = ResearchAgent().run(_fresh_state("ra-urls"))
        links = {r["link"] for r in update.get("research_results", [])}
        for url in EXPECTED_URLS:
            assert url in links, f"URL {url!r} missing from research_results links"


# ===========================================================================
# GROUP 4 -- FollowUpAgent unit
# ===========================================================================


class TestScenario2FollowUpAgent:
    """FollowUpAgent.run() in isolation."""

    def _research_state(
        self,
        suffix: str,
        current_response: str = "Based on latest travel trends [Web], consider Maldives.",
    ) -> dict[str, Any]:
        state = _fresh_state(f"fu-{suffix}")
        state["route"] = "research"
        state["guardrail_passed"] = True
        state["human_handoff"] = False
        state["current_response"] = current_response
        return state

    def test_proactive_suggestion_equals_followup_constant(self) -> None:
        update = FollowUpAgent().run(self._research_state("suggestion"))
        assert update.get("proactive_suggestion") == FOLLOWUP_SUGGESTION

    def test_suggestion_appended_to_existing_synthesis_response(self) -> None:
        synthesis = "Based on latest travel trends [Web], consider Maldives."
        update = FollowUpAgent().run(self._research_state("append", synthesis))
        combined = update.get("current_response", "")
        assert synthesis in combined
        assert FOLLOWUP_SUGGESTION in combined

    def test_followup_emits_suggestion_generated_trace(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from concierge.agents import followup as _mod
        calls: list[tuple[str, dict[str, Any]]] = []
        monkeypatch.setattr(_mod, "trace", lambda node, **kw: calls.append((node, kw)))
        FollowUpAgent().run(self._research_state("trace"))
        assert calls == [("followup", {"event": "suggestion_generated"})]

    def test_returns_empty_dict_when_human_handoff_true(self) -> None:
        state = self._research_state("handoff")
        state["human_handoff"] = True
        assert FollowUpAgent().run(state) == {}

    def test_returns_empty_dict_when_guardrail_passed_false(self) -> None:
        state = self._research_state("guardrail")
        state["guardrail_passed"] = False
        assert FollowUpAgent().run(state) == {}

    @pytest.mark.parametrize(
        "non_research_route",
        ["rag", "booking_stub", "fallback", None, ""],
    )
    def test_returns_empty_dict_for_all_non_research_routes(
        self, non_research_route: Any
    ) -> None:
        state = self._research_state(f"skip-{non_research_route}")
        state["route"] = non_research_route
        result = FollowUpAgent().run(state)
        assert result == {}, f"Expected {{}} for route={non_research_route!r}, got {result!r}"


# ===========================================================================
# GROUP 5 -- Full E2E graph invocation (primary happy-path)
# ===========================================================================


class TestE2EScenario2FullGraph:
    """
    initialize_state -> compiled_graph.invoke for the research happy-path.
    No ANTHROPIC_API_KEY in env; RESEARCH_AGENT_LLM_RANKING unset.
    Stage-1 resolves the route so no LLM call is made.
    """

    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._log = _patch_duckduckgo(monkeypatch)
        monkeypatch.delenv("RESEARCH_AGENT_LLM_RANKING", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    def _run(self, suffix: str = "main") -> dict[str, Any]:
        return compiled_graph.invoke(_fresh_state(suffix))  # type: ignore[return-value]

    # --- 1. Route and intent ---

    def test_route_is_research(self) -> None:
        assert self._run("route").get("route") == EXPECTED_ROUTE

    def test_intent_is_trend_research(self) -> None:
        assert self._run("intent").get("intent") == EXPECTED_INTENT

    def test_confidence_exceeds_escalation_threshold(self) -> None:
        conf = self._run("conf").get("confidence", 0.0)
        assert isinstance(conf, float)
        assert conf >= EXPECTED_ESCALATION_THRESHOLD

    # --- 2. ResearchAgent invoked with correct query ---

    def test_duckduckgo_called_exactly_once(self) -> None:
        self._run("ddg-once")
        assert len(self._log) == 1

    def test_scoped_query_contains_user_input(self) -> None:
        self._run("ddg-query")
        assert USER_QUERY in self._log[0]["query"]

    def test_duckduckgo_called_with_max_results_5(self) -> None:
        self._run("ddg-max")
        assert self._log[0]["max_results"] == 5

    # --- 3. research_results populated ---

    def test_research_results_is_a_list(self) -> None:
        assert isinstance(self._run("rr-type").get("research_results"), list)

    def test_research_results_has_5_entries(self) -> None:
        assert len(self._run("rr-count").get("research_results", [])) == 5

    def test_research_results_matches_parsed_expected(self) -> None:
        assert self._run("rr-shape").get("research_results") == EXPECTED_PARSED_RESULTS

    # --- 4. source_attribution contains URLs ---

    def test_source_attribution_is_non_empty(self) -> None:
        assert self._run("sa-nonempty").get("source_attribution")

    def test_source_attribution_has_web_tagged_entries(self) -> None:
        attr: list[str] = self._run("sa-tag").get("source_attribution", [])
        web = [e for e in attr if e.startswith("[Web]")]
        assert len(web) >= 1, f"No [Web] entries in source_attribution={attr}"

    def test_source_attribution_web_entries_contain_mocked_urls(self) -> None:
        attr: list[str] = self._run("sa-url").get("source_attribution", [])
        found = [url for entry in attr for url in EXPECTED_URLS if url in entry]
        assert len(found) >= 1, (
            f"No mocked URL found in source_attribution. attr={attr}"
        )

    def test_web_attribution_entries_all_contain_https(self) -> None:
        attr: list[str] = self._run("sa-https").get("source_attribution", [])
        for entry in (e for e in attr if e.startswith("[Web]")):
            assert "https://" in entry, f"[Web] entry missing URL: {entry!r}"

    # --- 5. degradation_label absent ---

    def test_degradation_label_is_none(self) -> None:
        assert self._run("degrad").get("degradation_label") is None

    # --- 6. Guardrail passes ---

    def test_guardrail_passed_is_true(self) -> None:
        assert self._run("gp").get("guardrail_passed") is True

    def test_human_handoff_is_false(self) -> None:
        assert self._run("hh").get("human_handoff") is False

    def test_clarification_not_needed(self) -> None:
        assert self._run("cn").get("clarification_needed") is False

    # --- 7-8. FollowUp node activates ---

    def test_followup_in_executed_nodes(self) -> None:
        executed = self._run("nodes-fu").get("_executed_nodes", [])
        assert NodeName.FOLLOWUP in executed, (
            f"FollowUp must be in _executed_nodes; got {executed}"
        )

    def test_executed_nodes_exact_order(self) -> None:
        executed = self._run("nodes-order").get("_executed_nodes")
        assert executed == [
            NodeName.DISPATCHER,
            NodeName.RESEARCH,
            NodeName.GUARDRAIL,
            NodeName.SYNTHESIS,
            NodeName.FOLLOWUP,
        ], f"Unexpected node execution order: {executed}"

    # --- 9. proactive_suggestion ---

    def test_proactive_suggestion_is_non_empty(self) -> None:
        assert self._run("ps-nonempty").get("proactive_suggestion")

    def test_proactive_suggestion_equals_followup_constant(self) -> None:
        result = self._run("ps-value")
        assert result.get("proactive_suggestion") == FOLLOWUP_SUGGESTION

    def test_followup_suggestion_appended_to_current_response(self) -> None:
        resp = self._run("ps-resp").get("current_response", "")
        assert FOLLOWUP_SUGGESTION in resp, (
            f"FOLLOWUP_SUGGESTION not found in current_response={resp!r}"
        )

    # --- 10. Response content ---

    def test_current_response_contains_web_tag(self) -> None:
        assert "[Web]" in str(self._run("resp-web").get("current_response") or "")

    def test_current_response_is_non_empty(self) -> None:
        assert str(self._run("resp-nonempty").get("current_response") or "").strip()

    # --- 11. conversation_history grows ---

    def test_conversation_history_contains_user_message(self) -> None:
        history: list[dict[str, Any]] = self._run("hist").get("conversation_history", [])
        assert len(history) >= 1
        assert {"role": "user", "content": USER_QUERY} in history

    def test_last_user_entry_in_history_is_current_query(self) -> None:
        history: list[dict[str, Any]] = self._run("hist-last").get("conversation_history", [])
        user_entries = [h for h in history if h.get("role") == "user"]
        assert user_entries[-1]["content"] == USER_QUERY

    # --- 12. No Anthropic SDK call ---

    def test_no_anthropic_client_constructed_on_stage1_path(self) -> None:
        mock_client = MagicMock()
        with patch("anthropic.Anthropic", return_value=mock_client) as mock_cls:
            compiled_graph.invoke(_fresh_state("no-llm"))
            mock_cls.assert_not_called()


# ===========================================================================
# GROUP 6 -- Conditional edge: research vs. other routes
# ===========================================================================


class TestScenario2ConditionalEdge:
    """should_run_followup() conditional edge must fire only for research route."""

    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_duckduckgo(monkeypatch)
        monkeypatch.delenv("RESEARCH_AGENT_LLM_RANKING", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    def test_followup_runs_for_research_route(self) -> None:
        result = compiled_graph.invoke(_fresh_state("cond-research"))
        assert NodeName.FOLLOWUP in result.get("_executed_nodes", [])
        assert result.get("proactive_suggestion") == FOLLOWUP_SUGGESTION

    def test_followup_absent_for_rag_route(self) -> None:
        state = _fresh_state("cond-rag", current_input="Tell me about Wyndham resorts")
        result = compiled_graph.invoke(state)
        executed = result.get("_executed_nodes", [])
        assert NodeName.FOLLOWUP not in executed, (
            f"FollowUp must not run for rag route; nodes={executed}"
        )
        assert result.get("proactive_suggestion") is None

    def test_followup_absent_for_booking_route(self) -> None:
        state = _fresh_state("cond-booking", current_input="I want to book a room")
        result = compiled_graph.invoke(state)
        executed = result.get("_executed_nodes", [])
        assert NodeName.FOLLOWUP not in executed, (
            f"FollowUp must not run for booking route; nodes={executed}"
        )
        assert result.get("proactive_suggestion") is None

    def test_rag_node_not_executed_on_research_path(self) -> None:
        result = compiled_graph.invoke(_fresh_state("cond-norag"))
        assert NodeName.RAG not in result.get("_executed_nodes", [])

    def test_research_node_not_executed_on_rag_path(self) -> None:
        state = _fresh_state("cond-noresearch", current_input="Tell me about Wyndham resorts")
        result = compiled_graph.invoke(state)
        assert NodeName.RESEARCH not in result.get("_executed_nodes", [])


# ===========================================================================
# GROUP 7 -- Multi-turn conversation history growth
# ===========================================================================


class TestScenario2MultiTurnHistory:
    """conversation_history must grow monotonically across successive turns."""

    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_duckduckgo(monkeypatch)
        monkeypatch.delenv("RESEARCH_AGENT_LLM_RANKING", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    def test_history_has_exactly_one_entry_after_turn_1(self) -> None:
        result = compiled_graph.invoke(_fresh_state("mt-t1", turn_id=1))
        history = result.get("conversation_history", [])
        assert len(history) == 1
        assert history[0] == {"role": "user", "content": USER_QUERY}

    def test_history_has_two_user_entries_after_turn_2(self) -> None:
        result_t1 = compiled_graph.invoke(_fresh_state("mt-t1b", turn_id=1))
        history_t1 = result_t1.get("conversation_history", [])

        follow_up = "Which of those are family-friendly?"
        state_t2 = _fresh_state("mt-t2", turn_id=2, current_input=follow_up)
        state_t2["conversation_history"] = list(history_t1)
        result_t2 = compiled_graph.invoke(state_t2)

        history_t2 = result_t2.get("conversation_history", [])
        assert len(history_t2) >= 2
        user_msgs = [h["content"] for h in history_t2 if h.get("role") == "user"]
        assert USER_QUERY in user_msgs
        assert follow_up in user_msgs

    def test_turn_id_preserved_unchanged_through_graph(self) -> None:
        state = _fresh_state("mt-turnid", turn_id=7)
        result = compiled_graph.invoke(state)
        assert result.get("turn_id") == 7


# ===========================================================================
# GROUP 8 -- Parametric: first 3 mocked URLs appear in source_attribution
# ===========================================================================


@pytest.mark.parametrize(
    "expected_url",
    EXPECTED_URLS[:3],
    ids=[f"url_{i}" for i in range(3)],
)
def test_source_attribution_contains_expected_url(
    expected_url: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    ResponseSynthesisAgent caps source_attribution to the first 3 research results.
    Each of those URLs must appear verbatim inside an attribution entry.
    """
    _patch_duckduckgo(monkeypatch)
    monkeypatch.delenv("RESEARCH_AGENT_LLM_RANKING", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    result = compiled_graph.invoke(_fresh_state(f"sa-par-{expected_url[-20:]}"))
    attribution: list[str] = result.get("source_attribution", [])
    assert any(expected_url in entry for entry in attribution), (
        f"URL {expected_url!r} not found in source_attribution={attribution}"
    )

