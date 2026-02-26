from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_memory_service_writes_session_state_with_required_fields(tmp_path: Path) -> None:
    from concierge.agents.memory_service import MemoryService

    sessions_dir = tmp_path / "memory" / "sessions"
    service = MemoryService(
        profiles_dir=tmp_path / "memory" / "profiles",
        sessions_dir=sessions_dir,
    )

    state = {
        "session_id": "session-test-43",
        "user_id": "alex",
        "conversation_history": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ],
        "turn_id": 2,
    }

    written = service.write_session_state(
        state,
        timestamp_start="2026-02-25T10:00:00Z",
        timestamp_end="2026-02-25T10:01:00Z",
    )

    assert written == sessions_dir / "session-test-43" / "state.json"
    payload = json.loads(written.read_text(encoding="utf-8"))
    assert payload == {
        "session_id": "session-test-43",
        "user_id": "alex",
        "timestamp_start": "2026-02-25T10:00:00Z",
        "timestamp_end": "2026-02-25T10:01:00Z",
        "conversation_history": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ],
        "turn_count": 2,
    }


def test_memory_service_session_write_uses_file_relative_paths_not_cwd(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from concierge.agents.memory_service import MemoryService

    monkeypatch.chdir(tmp_path)
    service = MemoryService()
    state = {
        "session_id": "session-story-43-path",
        "user_id": "alex",
        "conversation_history": [],
        "turn_id": 0,
    }

    written = service.write_session_state(
        state,
        timestamp_start="2026-02-25T10:00:00Z",
        timestamp_end="2026-02-25T10:00:10Z",
    )
    assert written is not None
    assert written.exists()
    assert str(written).endswith("memory\\sessions\\session-story-43-path\\state.json")

    if written.parent.exists():
        written.unlink(missing_ok=True)
        written.parent.rmdir()


def test_main_writes_session_file_on_natural_exit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    app_main = importlib.import_module("main")
    monkeypatch.setattr(app_main.sys, "version_info", (3, 11, 9))
    monkeypatch.setattr(app_main, "uuid4", lambda: SimpleNamespace(hex="session43"))
    monkeypatch.setattr(app_main, "_iso_utc_now", lambda: "2026-02-25T10:00:00Z")

    class _FakeGraph:
        def invoke(self, state: dict[str, object]) -> dict[str, object]:
            history = list(state.get("conversation_history") or [])
            history.append({"role": "user", "content": str(state.get("current_input", ""))})
            state = dict(state)
            state["conversation_history"] = history
            return state

    monkeypatch.setattr(
        app_main,
        "_load_runtime_dependencies",
        lambda: (_FakeGraph(), lambda **kwargs: dict(kwargs, conversation_history=[])),
    )
    monkeypatch.setattr(app_main, "_load_memory_profile", lambda user_id: None)
    monkeypatch.setattr(app_main, "_emit_proactive_memory_greeting", lambda user_id, profile: None)

    from concierge.agents.memory_service import MemoryService

    sessions_dir = tmp_path / "memory" / "sessions"

    original_import_module = app_main.importlib.import_module

    def _patched_import(name: str):
        if name == "concierge.agents.memory_service":
            return SimpleNamespace(
                MemoryService=lambda: MemoryService(
                    profiles_dir=tmp_path / "memory" / "profiles",
                    sessions_dir=sessions_dir,
                )
            )
        return original_import_module(name)

    monkeypatch.setattr(app_main.importlib, "import_module", _patched_import)

    inputs = iter(["hello", "exit"])
    code = app_main.main(["--user", "alex"], input_reader=lambda _prompt: next(inputs))

    assert code == 0
    session_file = sessions_dir / "session-session43" / "state.json"
    assert session_file.exists()
    payload = json.loads(session_file.read_text(encoding="utf-8"))
    assert payload["session_id"] == "session-session43"
    assert payload["user_id"] == "alex"
    assert payload["timestamp_start"] == "2026-02-25T10:00:00Z"
    assert payload["timestamp_end"] == "2026-02-25T10:00:00Z"
    assert isinstance(payload["conversation_history"], list)
    assert payload["turn_count"] == 1
