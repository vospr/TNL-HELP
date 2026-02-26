"""
E2E Test -- Scenario 8: Optional LLM Ranking for Research Results.

Part A -- Ranking Success
    RESEARCH_AGENT_LLM_RANKING=1, DuckDuckGo returns 5 canned results,
    LLM is mocked to return ranked indices [4, 0, 2].
    Asserts: research_results reordered (top-3 first), degradation_label=None,
             no error key, all five results preserved.

Part B -- Ranking Failure (Graceful Downgrade)
    Same env flag and 5 DDG results.  LLM call raises a simulated
    APIConnectionError.
    Asserts: research_results stay in original DDG parse order, no exception
             propagated, no error state, WEB_SEARCH_UNAVAILABLE sentinel absent,
             ranking_skipped trace emitted with reason=llm_api_error.

Group C -- _apply_ranking() contract (unit-level, no network)
    Invalid JSON, non-list JSON, partial indices, out-of-bounds, negative,
    duplicate, mixed-type array items, empty inputs -- all fall back silently.

Group D -- Feature-flag / env-var gate
    Flag absent, flag="0", flag="true" (truthy but != "1"), blank API key
    -- all bypass the LLM.  Positive control: flag="1" + valid key calls LLM.

Group E -- Full compiled_graph.invoke (graph-level E2E)
    Success path: top result matches LLM first choice, all five present.
    Failure path: original DDG order preserved, no error key, graph completes.
    Node sequence: dispatcher -> research -> guardrail -> synthesis -> followup.
"""
from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from concierge.state import initialize_state


# ===========================================================================
# Shared test data
# ===========================================================================

# Five raw DuckDuckGo-format records (href / body keys as returned by DDGS).
_FIVE_RAW: list[dict[str, str]] = [
    {"title": "Result 0 Budget hostels",   "href": "https://ex.com/0", "body": "Affordable stays."},
    {"title": "Result 1 Mid-range hotels", "href": "https://ex.com/1", "body": "Mid-range comfort."},
    {"title": "Result 2 Boutique villas",  "href": "https://ex.com/2", "body": "Unique boutique."},
    {"title": "Result 3 Business class",   "href": "https://ex.com/3", "body": "Corporate stays."},
    {"title": "Result 4 Luxury resorts",   "href": "https://ex.com/4", "body": "Five-star luxury."},
]

# After ResearchAgent._parse_results the "href" key becomes "link".
_FIVE_PARSED: list[dict[str, str]] = [
    {"title": r["title"], "link": r["href"], "snippet": r["body"]} for r in _FIVE_RAW
]

# LLM preference order: index 4 first, then 0, then 2.
_LLM_INDICES: list[int] = [4, 0, 2]

# Expected list after _apply_ranking([4, 0, 2]):
#   top-3 in LLM order, remaining [1, 3] in original relative order.
_EXPECTED: list[dict[str, str]] = [
    _FIVE_PARSED[4],
    _FIVE_PARSED[0],
    _FIVE_PARSED[2],
    _FIVE_PARSED[1],
    _FIVE_PARSED[3],
]


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def _make_state(session_id: str) -> dict[str, Any]:
    return dict(initialize_state(
        user_id="alex",
        session_id=session_id,
        current_input="Best luxury resorts in Southeast Asia",
        turn_id=1,
    ))


def _fake_ddg(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """DuckDuckGo stub -- always returns the five canned raw records."""
    return list(_FIVE_RAW)


def _success_anthropic(indices: list[int]) -> SimpleNamespace:
    """Fake anthropic module whose messages.create() returns indices as JSON."""

    class _Msg:
        def create(self, **kw: object) -> SimpleNamespace:
            return SimpleNamespace(content=[SimpleNamespace(text=json.dumps(indices))])

    class _Client:
        def __init__(self) -> None:
            self.messages = _Msg()

    return SimpleNamespace(Anthropic=lambda: _Client())


class _FakeAPIError(Exception):
    """Stand-in for anthropic.APIConnectionError -- no SDK import required."""


def _failing_anthropic() -> SimpleNamespace:
    """Fake anthropic module whose messages.create() raises _FakeAPIError."""

    class _Msg:
        def create(self, **kw: object) -> SimpleNamespace:
            raise _FakeAPIError("simulated APIConnectionError")

    class _Client:
        def __init__(self) -> None:
            self.messages = _Msg()
            self.APIConnectionError = _FakeAPIError

    return SimpleNamespace(
        Anthropic=lambda: _Client(),
        APIConnectionError=_FakeAPIError,
    )


# ===========================================================================
# Part A -- LLM Ranking Success
# ===========================================================================


class TestLLMRankingSuccess:
    """
    RESEARCH_AGENT_LLM_RANKING=1, five DDG results, LLM returns [4, 0, 2].
    Full success path: result list reordered, no data lost, labels clean.
    """

    @pytest.fixture(autouse=True)
    def _env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RESEARCH_AGENT_LLM_RANKING", "1")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-8a")
        monkeypatch.delenv("FAST_MODE", raising=False)

    # --- core ordering ---

    def test_results_reordered_by_llm_confidence(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from concierge.agents import research_agent as m

        monkeypatch.setattr(m, "search_duckduckgo", _fake_ddg)
        monkeypatch.setitem(sys.modules, "anthropic", _success_anthropic(_LLM_INDICES))

        results = m.ResearchAgent().run(_make_state("8a-reorder"))["research_results"]

        assert isinstance(results, list)
        assert results == _EXPECTED, (
            "Results must appear in LLM-specified order "
            "(top-3 first, remainder in original order)"
        )

    def test_position_0_is_llm_first_choice_index_4(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from concierge.agents import research_agent as m

        monkeypatch.setattr(m, "search_duckduckgo", _fake_ddg)
        monkeypatch.setitem(sys.modules, "anthropic", _success_anthropic(_LLM_INDICES))

        results = m.ResearchAgent().run(_make_state("8a-pos0"))["research_results"]

        assert results[0] == _FIVE_PARSED[4], "Position 0 must be original index 4 (LLM #1)"
        assert results[1] == _FIVE_PARSED[0], "Position 1 must be original index 0 (LLM #2)"
        assert results[2] == _FIVE_PARSED[2], "Position 2 must be original index 2 (LLM #3)"

    def test_top_3_titles_match_llm_preference(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from concierge.agents import research_agent as m

        monkeypatch.setattr(m, "search_duckduckgo", _fake_ddg)
        monkeypatch.setitem(sys.modules, "anthropic", _success_anthropic(_LLM_INDICES))

        top3 = [
            r["title"]
            for r in m.ResearchAgent().run(_make_state("8a-top3"))["research_results"][:3]
        ]

        assert "Result 4 Luxury resorts" in top3
        assert "Result 0 Budget hostels" in top3
        assert "Result 2 Boutique villas" in top3

    def test_all_five_results_preserved_no_data_dropped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from concierge.agents import research_agent as m

        monkeypatch.setattr(m, "search_duckduckgo", _fake_ddg)
        monkeypatch.setitem(sys.modules, "anthropic", _success_anthropic(_LLM_INDICES))

        results = m.ResearchAgent().run(_make_state("8a-count"))["research_results"]

        assert len(results) == 5, "All five results must survive ranking"
        assert {r["link"] for r in results} == {r["link"] for r in _FIVE_PARSED}, (
            "No result may be lost or duplicated"
        )

    # --- state fields ---

    def test_degradation_label_is_none_on_success(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from concierge.agents import research_agent as m

        monkeypatch.setattr(m, "search_duckduckgo", _fake_ddg)
        monkeypatch.setitem(sys.modules, "anthropic", _success_anthropic(_LLM_INDICES))

        assert m.ResearchAgent().run(_make_state("8a-label"))["degradation_label"] is None

    def test_no_error_key_on_success(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from concierge.agents import research_agent as m

        monkeypatch.setattr(m, "search_duckduckgo", _fake_ddg)
        monkeypatch.setitem(sys.modules, "anthropic", _success_anthropic(_LLM_INDICES))

        update = m.ResearchAgent().run(_make_state("8a-noerr"))
        assert not update.get("error"), "error key must not be set on successful ranking"


# ===========================================================================
# Part B -- Ranking Failure: Graceful Downgrade
# ===========================================================================


class TestLLMRankingFailureFallback:
    """
    RESEARCH_AGENT_LLM_RANKING=1, LLM raises APIConnectionError.
    Verify: original DDG order preserved, no exception, no error state,
    ranking_skipped trace emitted.
    """

    @pytest.fixture(autouse=True)
    def _env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RESEARCH_AGENT_LLM_RANKING", "1")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-8b")
        monkeypatch.delenv("FAST_MODE", raising=False)

    def test_results_in_original_ddg_order_on_llm_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from concierge.agents import research_agent as m

        monkeypatch.setattr(m, "search_duckduckgo", _fake_ddg)
        monkeypatch.setitem(sys.modules, "anthropic", _failing_anthropic())

        results = m.ResearchAgent().run(_make_state("8b-order"))["research_results"]

        assert isinstance(results, list)
        assert len(results) == 5
        assert results == _FIVE_PARSED, (
            "On LLM failure results must remain in original DuckDuckGo parse order"
        )

    def test_no_exception_propagated_to_caller(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from concierge.agents import research_agent as m

        monkeypatch.setattr(m, "search_duckduckgo", _fake_ddg)
        monkeypatch.setitem(sys.modules, "anthropic", _failing_anthropic())

        # Must complete without raising.
        update = m.ResearchAgent().run(_make_state("8b-noexc"))
        assert update is not None

    def test_error_state_not_set_on_ranking_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from concierge.agents import research_agent as m

        monkeypatch.setattr(m, "search_duckduckgo", _fake_ddg)
        monkeypatch.setitem(sys.modules, "anthropic", _failing_anthropic())

        update = m.ResearchAgent().run(_make_state("8b-errkey"))
        assert not update.get("error"), "LLM ranking failure must not propagate an error"

    def test_degradation_label_not_web_search_unavailable_sentinel(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ranking failure is a softer degradation than search failure.
        The WEB_SEARCH_UNAVAILABLE sentinel must NOT be set."""
        from concierge.agents import research_agent as m

        monkeypatch.setattr(m, "search_duckduckgo", _fake_ddg)
        monkeypatch.setitem(sys.modules, "anthropic", _failing_anthropic())

        update = m.ResearchAgent().run(_make_state("8b-deglabel"))
        assert update.get("degradation_label") != m.WEB_SEARCH_UNAVAILABLE_LABEL

    def test_ranking_skipped_trace_emitted_with_llm_api_error_reason(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When the LLM call raises, the agent must emit event=ranking_skipped
        with reason=llm_api_error for operator observability."""
        from concierge.agents import research_agent as m

        trace_calls: list[tuple[str, dict[str, object]]] = []
        monkeypatch.setattr(m, "search_duckduckgo", _fake_ddg)
        monkeypatch.setitem(sys.modules, "anthropic", _failing_anthropic())
        monkeypatch.setattr(
            m, "trace",
            lambda node, outcome=None, **fields: trace_calls.append((node, dict(fields))),
        )

        m.ResearchAgent().run(_make_state("8b-trace"))

        skipped = [
            (n, f) for n, f in trace_calls if f.get("event") == "ranking_skipped"
        ]
        assert skipped, "ranking_skipped trace must be emitted when LLM call raises"
        node, fields = skipped[0]
        assert node == "research_agent"
        assert fields.get("reason") == "llm_api_error"

    def test_search_complete_trace_still_fires_after_ranking_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """search_complete trace must fire even when ranking is skipped,
        ensuring result-count telemetry is always available."""
        from concierge.agents import research_agent as m

        trace_calls: list[tuple[str, dict[str, object]]] = []
        monkeypatch.setattr(m, "search_duckduckgo", _fake_ddg)
        monkeypatch.setitem(sys.modules, "anthropic", _failing_anthropic())
        monkeypatch.setattr(
            m, "trace",
            lambda node, outcome=None, **fields: trace_calls.append((node, dict(fields))),
        )

        m.ResearchAgent().run(_make_state("8b-trace-count"))

        complete = [
            (n, f) for n, f in trace_calls if f.get("event") == "search_complete"
        ]
        assert complete, "search_complete trace must still fire after ranking failure"
        _, fields = complete[0]
        assert fields.get("results_count") == 5


# ===========================================================================
# Group C -- _apply_ranking() low-level contract
# ===========================================================================


class TestApplyRankingContract:
    """
    Exercise ResearchAgent._apply_ranking() directly without the full agent
    pipeline.  Every edge case must degrade silently without raising.
    """

    @pytest.fixture(autouse=True)
    def _agent(self) -> None:
        from concierge.agents.research_agent import ResearchAgent
        self.agent = ResearchAgent()

    # --- happy-path ordering ---

    def test_full_valid_index_list_reorders_all(self) -> None:
        ranked = self.agent._apply_ranking(
            json.dumps([4, 0, 2, 1, 3]), list(_FIVE_PARSED)
        )
        assert ranked == [
            _FIVE_PARSED[4],
            _FIVE_PARSED[0],
            _FIVE_PARSED[2],
            _FIVE_PARSED[1],
            _FIVE_PARSED[3],
        ]

    def test_partial_indices_ranked_head_then_unranked_tail(self) -> None:
        """[4, 0, 2] -> top-3 reordered; [1, 3] appended in original order."""
        ranked = self.agent._apply_ranking(json.dumps([4, 0, 2]), list(_FIVE_PARSED))

        assert len(ranked) == 5
        assert ranked[0] == _FIVE_PARSED[4]
        assert ranked[1] == _FIVE_PARSED[0]
        assert ranked[2] == _FIVE_PARSED[2]
        assert ranked[3] == _FIVE_PARSED[1]   # unranked, original order
        assert ranked[4] == _FIVE_PARSED[3]   # unranked, original order

    # --- fallback on bad input ---

    def test_invalid_json_silently_returns_original(self) -> None:
        results = list(_FIVE_PARSED)
        assert self.agent._apply_ranking("not json{{", results) == results

    def test_json_object_not_list_returns_original(self) -> None:
        results = list(_FIVE_PARSED)
        assert self.agent._apply_ranking(
            json.dumps({"ranked": [4, 0, 2]}), results
        ) == results

    def test_json_scalar_returns_original(self) -> None:
        results = list(_FIVE_PARSED)
        assert self.agent._apply_ranking(json.dumps(42), results) == results

    def test_empty_json_array_returns_original(self) -> None:
        results = list(_FIVE_PARSED)
        assert self.agent._apply_ranking("[]", results) == results

    def test_empty_string_returns_original(self) -> None:
        results = list(_FIVE_PARSED)
        assert self.agent._apply_ranking("", results) == results

    # --- index boundary checks ---

    def test_out_of_bounds_positive_index_skipped(self) -> None:
        results = list(_FIVE_PARSED)
        ranked = self.agent._apply_ranking(json.dumps([99, 0]), results)
        assert ranked[0] == _FIVE_PARSED[0]
        assert len(ranked) == len(results)

    def test_negative_index_skipped(self) -> None:
        results = list(_FIVE_PARSED)
        ranked = self.agent._apply_ranking(json.dumps([-1, 0]), results)
        assert ranked[0] == _FIVE_PARSED[0]
        assert len(ranked) == len(results)

    def test_duplicate_indices_each_result_appears_exactly_once(self) -> None:
        results = list(_FIVE_PARSED)
        ranked = self.agent._apply_ranking(json.dumps([2, 2, 2, 0, 1, 3, 4]), results)
        links = [r["link"] for r in ranked]
        assert len(links) == len(set(links)), "Each result must appear exactly once"
        assert len(ranked) == len(results)

    def test_non_integer_items_in_array_skipped(self) -> None:
        """Strings and None mixed into the LLM array must be silently ignored."""
        results = list(_FIVE_PARSED)
        ranked = self.agent._apply_ranking(json.dumps(["best", 4, None, 0]), results)
        assert ranked[0] == _FIVE_PARSED[4]
        assert ranked[1] == _FIVE_PARSED[0]
        assert len(ranked) == len(results)

    # --- edge cases ---

    def test_empty_results_list_returns_empty_list(self) -> None:
        assert self.agent._apply_ranking(json.dumps([4, 0, 2]), []) == []

    def test_input_list_not_mutated_in_place(self) -> None:
        results = list(_FIVE_PARSED)
        snapshot = list(results)
        self.agent._apply_ranking(json.dumps([4, 0, 2]), results)
        assert results == snapshot, "_apply_ranking must not mutate the input list"


# ===========================================================================
# Group D -- Feature-flag / env-var gate
# ===========================================================================


class TestRankingFeatureFlag:
    """
    Verify RESEARCH_AGENT_LLM_RANKING precisely gates the LLM call.
    Only the exact string "1" with a non-blank ANTHROPIC_API_KEY triggers ranking.
    """

    def _sentinel(self, called: list[bool]) -> SimpleNamespace:
        """Fake anthropic module that records whether create() was invoked."""

        class _Msg:
            def create(self, **kw: object) -> SimpleNamespace:
                called.append(True)
                return SimpleNamespace(content=[SimpleNamespace(text="[4,0,2]")])

        class _Client:
            def __init__(self) -> None:
                self.messages = _Msg()

        return SimpleNamespace(Anthropic=lambda: _Client())

    def _assert_llm_skipped(
        self,
        monkeypatch: pytest.MonkeyPatch,
        flag: str | None,
        api_key: str | None,
        session: str,
    ) -> None:
        from concierge.agents import research_agent as m

        if flag is None:
            monkeypatch.delenv("RESEARCH_AGENT_LLM_RANKING", raising=False)
        else:
            monkeypatch.setenv("RESEARCH_AGENT_LLM_RANKING", flag)

        if api_key is None:
            monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        else:
            monkeypatch.setenv("ANTHROPIC_API_KEY", api_key)

        called: list[bool] = []
        monkeypatch.setattr(m, "search_duckduckgo", _fake_ddg)
        monkeypatch.setitem(sys.modules, "anthropic", self._sentinel(called))

        update = m.ResearchAgent().run(_make_state(session))

        assert not called, f"LLM must not be called (flag={flag!r}, key={api_key!r})"
        assert update["research_results"] == _FIVE_PARSED

    def test_flag_absent_skips_llm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._assert_llm_skipped(monkeypatch, None, "key-8d-absent", "8d-absent")

    def test_flag_zero_skips_llm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._assert_llm_skipped(monkeypatch, "0", "key-8d-zero", "8d-zero")

    def test_flag_truthy_non_one_skips_llm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Values like "true", "yes", "2" are NOT "1" and must not enable ranking."""
        self._assert_llm_skipped(monkeypatch, "true", "key-8d-true", "8d-true")

    def test_missing_api_key_skips_llm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Flag=1 but no API key -- ranking is silently bypassed."""
        self._assert_llm_skipped(monkeypatch, "1", None, "8d-nokey")

    def test_blank_api_key_treated_as_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Whitespace-only API key must be treated as absent."""
        self._assert_llm_skipped(monkeypatch, "1", "   ", "8d-blank")

    def test_flag_one_with_valid_key_invokes_llm(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Positive control: flag="1" + valid key must invoke the LLM."""
        from concierge.agents import research_agent as m

        monkeypatch.setenv("RESEARCH_AGENT_LLM_RANKING", "1")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "key-8d-active")
        monkeypatch.delenv("FAST_MODE", raising=False)
        called: list[bool] = []

        class _TrackMsg:
            def create(self, **kw: object) -> SimpleNamespace:
                called.append(True)
                return SimpleNamespace(
                    content=[SimpleNamespace(text=json.dumps(_LLM_INDICES))]
                )

        class _TrackClient:
            def __init__(self) -> None:
                self.messages = _TrackMsg()

        monkeypatch.setattr(m, "search_duckduckgo", _fake_ddg)
        monkeypatch.setitem(
            sys.modules, "anthropic", SimpleNamespace(Anthropic=lambda: _TrackClient())
        )

        m.ResearchAgent().run(_make_state("8d-active"))
        assert called, "LLM must be called when flag=1 and API key is set"


# ===========================================================================
# Group E -- Graph-level E2E via compiled_graph.invoke
# ===========================================================================


class TestGraphLevelE2E:
    """
    Drive the full scenario through compiled_graph.invoke to validate the
    complete node-execution pipeline under both ranking paths.
    """

    def _setup_graph(
        self,
        monkeypatch: pytest.MonkeyPatch,
        key_suffix: str,
        anthropic_mod: SimpleNamespace,
    ) -> None:
        from concierge.agents import research_agent as m

        monkeypatch.setenv("RESEARCH_AGENT_LLM_RANKING", "1")
        monkeypatch.setenv("ANTHROPIC_API_KEY", f"key-{key_suffix}")
        monkeypatch.delenv("FAST_MODE", raising=False)
        monkeypatch.setattr(m, "search_duckduckgo", _fake_ddg)
        monkeypatch.setitem(sys.modules, "anthropic", anthropic_mod)

    def test_graph_success_path_reorders_top_result(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from concierge.graph import compiled_graph

        self._setup_graph(monkeypatch, "graph-8a", _success_anthropic(_LLM_INDICES))

        state = _make_state("e2e-8-graph-ok")
        state["route"] = "research"
        state["rag_results"] = []

        result = compiled_graph.invoke(state)

        assert isinstance(result, dict)
        rr = result.get("research_results") or []
        assert len(rr) == 5
        assert rr[0]["title"] == "Result 4 Luxury resorts", (
            "Top graph result must be the LLM highest-confidence pick"
        )
        assert rr == _EXPECTED

    def test_graph_success_path_research_node_in_executed_nodes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from concierge.graph import compiled_graph

        self._setup_graph(monkeypatch, "graph-8a-node", _success_anthropic(_LLM_INDICES))

        state = _make_state("e2e-8-graph-node")
        state["route"] = "research"
        state["rag_results"] = []

        result = compiled_graph.invoke(state)
        assert "research" in (result.get("_executed_nodes") or [])

    def test_graph_failure_path_preserves_original_ddg_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from concierge.graph import compiled_graph

        self._setup_graph(monkeypatch, "graph-8b", _failing_anthropic())

        state = _make_state("e2e-8-graph-fail")
        state["route"] = "research"
        state["rag_results"] = []

        result = compiled_graph.invoke(state)

        assert isinstance(result, dict)
        rr = result.get("research_results") or []
        assert len(rr) == 5
        assert rr == _FIVE_PARSED, (
            "On LLM failure the graph must return results in original DDG order"
        )

    def test_graph_failure_path_no_error_state_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from concierge.graph import compiled_graph

        self._setup_graph(monkeypatch, "graph-8b-nocrash", _failing_anthropic())

        state = _make_state("e2e-8-graph-nocrash")
        state["route"] = "research"
        state["rag_results"] = []

        result = compiled_graph.invoke(state)

        assert result is not None
        assert not result.get("error"), "LLM ranking failure must not set error key"

    def test_graph_node_sequence_dispatcher_research_guardrail_synthesis_followup(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """For the research route the graph must execute the full canonical sequence."""
        from concierge.graph import compiled_graph

        self._setup_graph(monkeypatch, "graph-8-seq", _success_anthropic(_LLM_INDICES))

        state = _make_state("e2e-8-graph-seq")
        state["route"] = "research"
        state["rag_results"] = []

        result = compiled_graph.invoke(state)

        executed = result.get("_executed_nodes", [])
        expected = ["dispatcher", "research", "guardrail", "synthesis", "followup"]
        assert executed == expected, (
            "Research route must execute exactly: {}\nGot: {}".format(expected, executed)
        )
