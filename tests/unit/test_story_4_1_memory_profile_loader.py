from __future__ import annotations

import importlib
from pathlib import Path

import pytest


def test_memory_service_loads_alex_profile_with_required_fields() -> None:
    from concierge.agents.memory_service import MemoryService

    profile = MemoryService().load_profile("alex")

    assert profile is not None
    assert profile["user_id"] == "alex"
    for key in ("past_trips", "preferences", "cached_research_session", "last_seen"):
        assert key in profile


def test_memory_service_resolves_profiles_path_from_file_not_cwd(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from concierge.agents.memory_service import MemoryService

    monkeypatch.chdir(tmp_path)
    profile = MemoryService().load_profile("alex")

    assert profile is not None
    assert profile["user_id"] == "alex"


def test_memory_service_missing_profile_returns_none_and_warns(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from concierge.agents.memory_service import MemoryService

    profile = MemoryService().load_profile("missing-user")

    assert profile is None
    captured = capsys.readouterr()
    assert captured.out.strip() == (
        "[MEMORY] Profile for missing-user not found — starting fresh session"
    )


def test_main_populates_memory_profile_on_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app_main = importlib.import_module("main")
    monkeypatch.setattr(app_main.sys, "version_info", (3, 11, 9))

    observed_state: dict[str, object] = {}

    class _FakeGraph:
        def invoke(self, state: dict[str, object]) -> dict[str, object]:
            return state

    def _fake_init_state(**kwargs: object) -> dict[str, object]:
        return dict(kwargs)

    def _fake_run_loop(compiled_graph, state, input_reader):  # noqa: ANN001
        del compiled_graph, input_reader
        observed_state.update(state)
        return 0

    monkeypatch.setattr(
        app_main,
        "_load_runtime_dependencies",
        lambda: (_FakeGraph(), _fake_init_state),
    )
    monkeypatch.setattr(app_main, "_run_loop", _fake_run_loop)

    code = app_main.main(["--user", "alex"], input_reader=lambda _prompt: "exit")

    assert code == 0
    assert observed_state.get("memory_profile") is not None
    assert isinstance(observed_state["memory_profile"], dict)
    assert observed_state["memory_profile"]["user_id"] == "alex"


def test_main_missing_profile_sets_none_and_prints_warning(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    app_main = importlib.import_module("main")
    monkeypatch.setattr(app_main.sys, "version_info", (3, 11, 9))

    observed_state: dict[str, object] = {}

    class _FakeGraph:
        def invoke(self, state: dict[str, object]) -> dict[str, object]:
            return state

    def _fake_init_state(**kwargs: object) -> dict[str, object]:
        return dict(kwargs)

    def _fake_run_loop(compiled_graph, state, input_reader):  # noqa: ANN001
        del compiled_graph, input_reader
        observed_state.update(state)
        return 0

    monkeypatch.setattr(
        app_main,
        "_load_runtime_dependencies",
        lambda: (_FakeGraph(), _fake_init_state),
    )
    monkeypatch.setattr(app_main, "_run_loop", _fake_run_loop)

    code = app_main.main(["--user", "no-such-user"], input_reader=lambda _prompt: "exit")

    assert code == 0
    assert observed_state.get("memory_profile") is None
    captured = capsys.readouterr()
    assert captured.out.strip() == (
        "[MEMORY] Profile for no-such-user not found — starting fresh session"
    )
