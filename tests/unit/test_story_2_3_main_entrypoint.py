from __future__ import annotations

import importlib
import subprocess
import sys
import types
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_main_module():
    return importlib.import_module("main")


def test_main_help_documents_user_and_fast_mode_flags() -> None:
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "main.py"), "--help"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--user" in result.stdout
    assert "--fast-mode" in result.stdout


def test_python_version_guard_runs_before_runtime_loader(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    app_main = _load_main_module()
    monkeypatch.setattr(app_main.sys, "version_info", (3, 10, 9))

    loader_called = {"value": False}

    def _runtime_loader():
        loader_called["value"] = True
        raise AssertionError("runtime loader must not run when python version is unsupported")

    monkeypatch.setattr(app_main, "_load_runtime_dependencies", _runtime_loader)

    code = app_main.main(["--user", "alex"], input_reader=lambda _prompt: "exit")

    assert code == 1
    assert loader_called["value"] is False
    assert "Python 3.11+ required" in capsys.readouterr().err


def test_validate_checks_run_before_graph_import(monkeypatch: pytest.MonkeyPatch) -> None:
    app_main = _load_main_module()
    events: list[str] = []

    def _fake_import(name: str):
        if name == "validate_config":
            return types.SimpleNamespace(run_checks=lambda: events.append("run_checks"))
        if name == "concierge.graph":
            events.append("graph_import")
            return types.SimpleNamespace(compiled_graph=object())
        if name == "concierge.state":
            return types.SimpleNamespace(initialize_state=lambda **kwargs: kwargs)
        raise AssertionError(f"Unexpected import requested: {name}")

    monkeypatch.setattr(app_main, "_ensure_src_on_path", lambda: None)
    monkeypatch.setattr(app_main.importlib, "import_module", _fake_import)

    compiled_graph, initialize_state = app_main._load_runtime_dependencies()

    assert compiled_graph is not None
    assert callable(initialize_state)
    assert events == ["run_checks", "graph_import"]


def test_validation_failure_results_in_exit_code_1(monkeypatch: pytest.MonkeyPatch) -> None:
    app_main = _load_main_module()
    monkeypatch.setattr(app_main.sys, "version_info", (3, 11, 9))

    def _runtime_loader():
        raise SystemExit(1)

    monkeypatch.setattr(app_main, "_load_runtime_dependencies", _runtime_loader)

    code = app_main.main(["--user", "alex"], input_reader=lambda _prompt: "exit")
    assert code == 1


def test_user_id_seeded_before_first_graph_invoke_and_startup_without_errors(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    app_main = _load_main_module()
    monkeypatch.setattr(app_main.sys, "version_info", (3, 11, 9))

    observed_states: list[dict[str, object]] = []

    class _FakeGraph:
        def invoke(self, state: dict[str, object]) -> dict[str, object]:
            observed_states.append(dict(state))
            return state

    def _fake_initialize_state(
        user_id: str,
        session_id: str,
        current_input: str = "",
        turn_id: int = 0,
    ) -> dict[str, object]:
        return {
            "user_id": user_id,
            "session_id": session_id,
            "turn_id": turn_id,
            "current_input": current_input,
        }

    monkeypatch.setattr(
        app_main,
        "_load_runtime_dependencies",
        lambda: (_FakeGraph(), _fake_initialize_state),
    )
    monkeypatch.setattr(app_main, "_load_memory_profile", lambda user_id: None)
    monkeypatch.setattr(app_main, "_emit_proactive_memory_greeting", lambda user_id, profile: None)

    inputs = iter(["hello", "exit"])
    code = app_main.main(
        ["--user", "alex"],
        input_reader=lambda _prompt: next(inputs),
    )

    assert code == 0
    assert observed_states
    assert observed_states[0]["user_id"] == "alex"
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_fast_mode_flag_is_parsed_without_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    app_main = _load_main_module()
    monkeypatch.setattr(app_main.sys, "version_info", (3, 11, 9))

    class _FakeGraph:
        def invoke(self, state: dict[str, object]) -> dict[str, object]:
            return state

    monkeypatch.setattr(
        app_main,
        "_load_runtime_dependencies",
        lambda: (_FakeGraph(), lambda **kwargs: dict(kwargs)),
    )

    code = app_main.main(
        ["--user", "alex", "--fast-mode"],
        input_reader=lambda _prompt: "exit",
    )

    assert code == 0
