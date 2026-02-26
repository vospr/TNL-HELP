from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_memory_service_malformed_json_returns_none_and_prints_warning(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from concierge.agents.memory_service import MemoryService

    profiles_dir = tmp_path / "memory" / "profiles"
    profiles_dir.mkdir(parents=True)
    (profiles_dir / "bad.json").write_text("{not-valid-json", encoding="utf-8")

    service = MemoryService(profiles_dir=profiles_dir)
    profile = service.load_profile("bad")

    assert profile is None
    captured = capsys.readouterr()
    assert captured.out.strip() == "[MEMORY] Profile for bad not found — starting fresh session"


def test_main_missing_profile_continues_to_loop_without_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app_main = importlib.import_module("main")
    monkeypatch.setattr(app_main.sys, "version_info", (3, 11, 9))
    monkeypatch.setattr(app_main, "_iso_utc_now", lambda: "2026-02-25T10:00:00Z")

    observed_state: dict[str, object] = {}
    observed_inputs: list[str] = []

    class _FakeGraph:
        def invoke(self, state: dict[str, object]) -> dict[str, object]:
            observed_inputs.append(str(state.get("current_input") or ""))
            return dict(state)

    def _fake_init_state(**kwargs: object) -> dict[str, object]:
        return dict(kwargs, conversation_history=[])

    def _fake_persist(state: dict[str, object], timestamp_start: str) -> None:
        observed_state.update(state)
        assert timestamp_start == "2026-02-25T10:00:00Z"

    monkeypatch.setattr(
        app_main,
        "_load_runtime_dependencies",
        lambda: (_FakeGraph(), _fake_init_state),
    )
    monkeypatch.setattr(app_main, "_persist_session_state", _fake_persist)
    monkeypatch.setattr(app_main, "_load_memory_profile", lambda user_id: None)
    monkeypatch.setattr(app_main, "_emit_proactive_memory_greeting", lambda user_id, profile: None)

    inputs = iter(["hello", "exit"])
    code = app_main.main(["--user", "unknown_user"], input_reader=lambda _prompt: next(inputs))

    assert code == 0
    assert observed_state.get("memory_profile") is None
    assert observed_inputs == ["hello"]


def test_main_malformed_profile_handled_at_memory_boundary_no_exception(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    app_main = importlib.import_module("main")
    monkeypatch.setattr(app_main.sys, "version_info", (3, 11, 9))
    monkeypatch.setattr(app_main, "_iso_utc_now", lambda: "2026-02-25T10:00:00Z")
    monkeypatch.setattr(app_main, "uuid4", lambda: SimpleNamespace(hex="story44"))

    profiles_dir = tmp_path / "memory" / "profiles"
    sessions_dir = tmp_path / "memory" / "sessions"
    profiles_dir.mkdir(parents=True)
    (profiles_dir / "broken.json").write_text("{bad-json", encoding="utf-8")

    from concierge.agents.memory_service import MemoryService

    original_import_module = app_main.importlib.import_module

    def _patched_import(name: str):
        if name == "concierge.agents.memory_service":
            return SimpleNamespace(
                MemoryService=lambda: MemoryService(
                    profiles_dir=profiles_dir,
                    sessions_dir=sessions_dir,
                ),
                build_proactive_greeting=lambda profile, user_id: None,
            )
        return original_import_module(name)

    monkeypatch.setattr(app_main.importlib, "import_module", _patched_import)

    observed_inputs: list[str] = []

    class _FakeGraph:
        def invoke(self, state: dict[str, object]) -> dict[str, object]:
            observed_inputs.append(str(state.get("current_input") or ""))
            return dict(state)

    monkeypatch.setattr(
        app_main,
        "_load_runtime_dependencies",
        lambda: (_FakeGraph(), lambda **kwargs: dict(kwargs, conversation_history=[])),
    )

    inputs = iter(["hello", "exit"])
    code = app_main.main(["--user", "broken"], input_reader=lambda _prompt: next(inputs))

    assert code == 0
    assert observed_inputs == ["hello"]
