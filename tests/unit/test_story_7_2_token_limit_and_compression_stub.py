from __future__ import annotations

import logging

import pytest

from concierge.agents.dispatcher import DispatcherAgent
from concierge.agents.token_budget_manager import TokenBudgetManager
from concierge.state import initialize_state


def test_dispatcher_runs_token_budget_check_before_stage2_llm_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_order: list[str] = []
    traces: list[tuple[str, dict[str, object]]] = []

    def _stage1(self: DispatcherAgent, current_input: str) -> tuple[str | None, float, str | None]:
        del self, current_input
        call_order.append("stage1")
        return None, 0.0, None

    def _check(self: TokenBudgetManager, history: list[dict[str, str]]) -> list[dict[str, str]]:
        del self
        call_order.append("token_budget")
        return history

    def _stage2(self: DispatcherAgent, current_input: str) -> tuple[str | None, float | None, str | None]:
        del self, current_input
        call_order.append("stage2")
        return "destination_research", 0.84, "research"

    def _capture_trace(node_name: str, outcome: str | None = None, **fields: object) -> None:
        del outcome
        traces.append((node_name, dict(fields)))

    monkeypatch.setattr(DispatcherAgent, "_evaluate_stage1", _stage1)
    monkeypatch.setattr(TokenBudgetManager, "check_and_summarize", _check)
    monkeypatch.setattr(DispatcherAgent, "_evaluate_stage2", _stage2)
    monkeypatch.setattr("concierge.agents.dispatcher.trace", _capture_trace)

    state = initialize_state("alex", "session-7-2-order", "ambiguous", turn_id=1)
    update = DispatcherAgent().run(state)

    assert call_order == ["stage1", "token_budget", "stage2"]
    assert update["route"] == "research"

    token_budget_trace = [fields for node_name, fields in traces if node_name == "token_budget"]
    assert len(token_budget_trace) == 1
    assert token_budget_trace[0] == {
        "event": "check_performed",
        "message_count": 1,
        "threshold": 6000,
    }


def test_token_budget_manager_creates_system_summary_messages_when_threshold_reached() -> None:
    manager = TokenBudgetManager()
    history = [{"role": "user", "content": f"turn-{index}"} for index in range(6000)]

    compressed = manager.check_and_summarize(history)

    assert compressed[0]["role"] == "system_summary"
    assert str(compressed[0]["content"]).startswith("[SUMMARY]")
    assert len(compressed) < len(history)


def test_dispatcher_logs_token_budget_warning_when_compression_is_triggered(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def _stage1(self: DispatcherAgent, current_input: str) -> tuple[str | None, float, str | None]:
        del self, current_input
        return None, 0.0, None

    def _stage2(self: DispatcherAgent, current_input: str) -> tuple[str | None, float | None, str | None]:
        del self, current_input
        return "destination_research", 0.84, "research"

    monkeypatch.setattr(DispatcherAgent, "_evaluate_stage1", _stage1)
    monkeypatch.setattr(DispatcherAgent, "_evaluate_stage2", _stage2)
    monkeypatch.setattr("concierge.agents.dispatcher.trace", lambda *args, **kwargs: None)

    state = initialize_state("alex", "session-7-2-log", "latest input", turn_id=1)
    state["conversation_history"] = [
        {"role": "user", "content": f"old-{index}"}
        for index in range(5999)
    ]

    with caplog.at_level(logging.INFO, logger="concierge.agents.dispatcher"):
        update = DispatcherAgent().run(state)

    assert "[TOKEN_BUDGET] Context approaching limit. In production, oldest turns would be summarized here." in caplog.text
    assert update["conversation_history"][0]["role"] == "system_summary"
