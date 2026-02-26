from __future__ import annotations

from pathlib import Path
from typing import get_type_hints


def test_token_budget_manager_exposes_required_interface_signature() -> None:
    from concierge.agents import token_budget_manager as token_module

    hints = get_type_hints(token_module.TokenBudgetManager.check_and_summarize)
    assert hints["history"] == list[token_module.Message]
    assert hints["return"] == list[token_module.Message]


def test_token_budget_manager_stub_comment_documents_activation_threshold() -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "concierge"
        / "agents"
        / "token_budget_manager.py"
    ).read_text(encoding="utf-8")

    assert (
        "Stub implementation. In production, summarize when token count exceeds 6000 "
        "(leaving 2000 for output). Activation threshold: 80% of model context window."
        in source
    )


def test_token_budget_manager_returns_history_unchanged_for_mvp_noop() -> None:
    from concierge.agents.token_budget_manager import TokenBudgetManager

    history = [{"role": "user", "content": "hello"}]
    manager = TokenBudgetManager()

    result = manager.check_and_summarize(history)

    assert result is history
    assert result == [{"role": "user", "content": "hello"}]


def test_token_budget_manager_threshold_property_is_hardcoded_with_todo_marker() -> None:
    from concierge.agents.token_budget_manager import TokenBudgetManager

    manager = TokenBudgetManager()
    assert manager.activation_threshold_tokens == 6000

    source = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "concierge"
        / "agents"
        / "token_budget_manager.py"
    ).read_text(encoding="utf-8")
    assert "# TODO: make configurable" in source
