from __future__ import annotations

import ast
import io
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
NODES_DIR = REPO_ROOT / "src" / "concierge" / "nodes"
NODE_FILES = [
    "dispatcher_node.py",
    "rag_node.py",
    "research_node.py",
    "booking_node.py",
    "guardrail_node.py",
    "synthesis_node.py",
    "followup_node.py",
]


def test_trace_prints_allowlisted_fields_in_expected_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from concierge import trace as trace_module

    buffer = io.StringIO()
    monkeypatch.setattr(trace_module, "_trace_writer", buffer)

    trace_module.trace("dispatcher", intent="travel", confidence=0.87)
    assert buffer.getvalue().strip() == "[dispatcher] intent=travel confidence=0.87"


def test_trace_rejects_denylisted_field_with_actionable_message() -> None:
    from concierge.trace import trace

    with pytest.raises(ValueError) as exc:
        trace("dispatcher", current_input="secret")
    assert str(exc.value) == "current_input is not emittable"


def test_trace_rejects_api_key_shaped_fields() -> None:
    from concierge.trace import trace

    with pytest.raises(ValueError) as exc:
        trace("dispatcher", anthropic_api_key="sk-secret")
    assert str(exc.value) == "anthropic_api_key is not emittable"


def test_allowlist_and_denylist_include_required_contract_fields() -> None:
    from concierge.trace import TRACE_ALLOWLIST, TRACE_DENYLIST

    assert {
        "intent",
        "confidence",
        "route",
        "model",
        "session_id",
        "turn_id",
    }.issubset(TRACE_ALLOWLIST)
    assert {"status", "event", "stage", "reason", "results_count"}.issubset(TRACE_ALLOWLIST)

    assert {
        "current_input",
        "memory_profile",
        "conversation_history",
        "rag_results",
        "research_results",
    }.issubset(TRACE_DENYLIST)


def test_each_node_calls_trace_once_and_does_not_use_print() -> None:
    for filename in NODE_FILES:
        path = NODES_DIR / filename
        module = ast.parse(path.read_text(encoding="utf-8"))

        trace_calls = 0
        print_calls = 0
        for node in ast.walk(module):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == "trace":
                    trace_calls += 1
                if isinstance(node.func, ast.Name) and node.func.id == "print":
                    print_calls += 1

        assert trace_calls == 1, f"{filename} must call trace() exactly once"
        assert print_calls == 0, f"{filename} must not use print(); use trace()"
