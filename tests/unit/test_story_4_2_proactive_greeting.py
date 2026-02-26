from __future__ import annotations

import importlib
import types

import pytest


def test_build_proactive_greeting_uses_profile_data() -> None:
    from concierge.agents.memory_service import MemoryService, build_proactive_greeting

    profile = MemoryService().load_profile("alex")
    assert profile is not None

    greeting = build_proactive_greeting(profile, "alex")
    assert greeting == (
        "Welcome back, Alex. Based on your March trip to Bali, "
        "you might be interested in upcoming deals to Southeast Asia."
    )


def test_main_emits_greeting_and_trace_before_loop(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    app_main = importlib.import_module("main")
    monkeypatch.setattr(app_main.sys, "version_info", (3, 11, 9))

    observed_state: dict[str, object] = {}
    observed_pre_loop_output: dict[str, str] = {}
    trace_calls: list[tuple[str, dict[str, object]]] = []

    class _FakeGraph:
        def invoke(self, state: dict[str, object]) -> dict[str, object]:
            return state

    def _fake_init_state(**kwargs: object) -> dict[str, object]:
        return dict(kwargs)

    def _fake_run_loop(compiled_graph, state, input_reader):  # noqa: ANN001
        del compiled_graph, input_reader
        observed_pre_loop_output["stdout"] = capsys.readouterr().out
        observed_state.update(state)
        return 0

    monkeypatch.setattr(
        app_main,
        "_load_runtime_dependencies",
        lambda: (_FakeGraph(), _fake_init_state),
    )
    monkeypatch.setattr(app_main, "_run_loop", _fake_run_loop)
    original_import_module = app_main.importlib.import_module

    def _import_with_trace_spy(name: str):
        if name == "concierge.trace":
            return types.SimpleNamespace(
                trace=lambda node_name, outcome=None, **fields: trace_calls.append(
                    (node_name, dict(fields))
                )
            )
        return original_import_module(name)

    monkeypatch.setattr(app_main.importlib, "import_module", _import_with_trace_spy)

    code = app_main.main(["--user", "alex"], input_reader=lambda _prompt: "exit")
    capsys.readouterr()

    assert code == 0
    assert observed_state.get("memory_profile") is not None
    out = observed_pre_loop_output["stdout"]
    assert "Welcome back, Alex." in out
    assert trace_calls == [("memory", {"event": "greeting_fired", "user_id": "alex"})]


def test_main_fast_mode_prints_banner_then_greeting(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    app_main = importlib.import_module("main")
    monkeypatch.setattr(app_main.sys, "version_info", (3, 11, 9))

    observed_pre_loop_output: dict[str, str] = {}

    class _FakeGraph:
        def invoke(self, state: dict[str, object]) -> dict[str, object]:
            return state

    def _fake_init_state(**kwargs: object) -> dict[str, object]:
        return dict(kwargs)

    def _fake_run_loop(compiled_graph, state, input_reader):  # noqa: ANN001
        del compiled_graph, state, input_reader
        observed_pre_loop_output["stdout"] = capsys.readouterr().out
        return 0

    monkeypatch.setattr(
        app_main,
        "_load_runtime_dependencies",
        lambda: (_FakeGraph(), _fake_init_state),
    )
    monkeypatch.setattr(app_main, "_run_loop", _fake_run_loop)

    code = app_main.main(["--user", "alex", "--fast-mode"], input_reader=lambda _prompt: "exit")

    assert code == 0
    lines = [line for line in observed_pre_loop_output["stdout"].splitlines() if line.strip()]
    assert lines[0] == app_main._FAST_MODE_BANNER
    assert lines[1].startswith("Welcome back, Alex.")


def test_main_no_profile_keeps_warning_and_emits_no_greeting(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    app_main = importlib.import_module("main")
    monkeypatch.setattr(app_main.sys, "version_info", (3, 11, 9))

    observed_pre_loop_output: dict[str, str] = {}

    class _FakeGraph:
        def invoke(self, state: dict[str, object]) -> dict[str, object]:
            return state

    def _fake_init_state(**kwargs: object) -> dict[str, object]:
        return dict(kwargs)

    def _fake_run_loop(compiled_graph, state, input_reader):  # noqa: ANN001
        del compiled_graph, state, input_reader
        observed_pre_loop_output["stdout"] = capsys.readouterr().out
        return 0

    monkeypatch.setattr(
        app_main,
        "_load_runtime_dependencies",
        lambda: (_FakeGraph(), _fake_init_state),
    )
    monkeypatch.setattr(app_main, "_run_loop", _fake_run_loop)

    code = app_main.main(["--user", "no-such-user"], input_reader=lambda _prompt: "exit")

    assert code == 0
    out = observed_pre_loop_output["stdout"]
    assert "[MEMORY] Profile for no-such-user not found — starting fresh session" in out
    assert "Welcome back," not in out
    assert "greeting_fired" not in out
