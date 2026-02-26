from __future__ import annotations

import importlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_active_session_keeps_full_conversation_history_in_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app_main = importlib.import_module("main")
    monkeypatch.setattr(app_main.sys, "version_info", (3, 11, 9))
    monkeypatch.setattr(app_main, "_iso_utc_now", lambda: "2026-02-25T10:00:00Z")

    observed_state: dict[str, object] = {}

    class _FakeGraph:
        def invoke(self, state: dict[str, object]) -> dict[str, object]:
            history = list(state.get("conversation_history") or [])
            user_turn = {"role": "user", "content": str(state.get("current_input") or "")}
            assistant_turn = {
                "role": "assistant",
                "content": f"Echo: {state.get('current_input')}",
            }
            history.extend([user_turn, assistant_turn])
            updated = dict(state)
            updated["conversation_history"] = history
            updated["current_response"] = assistant_turn["content"]
            return updated

    def _fake_init_state(**kwargs: object) -> dict[str, object]:
        return dict(kwargs, conversation_history=[])

    def _capture_persist(state: dict[str, object], timestamp_start: str) -> None:
        del timestamp_start
        observed_state.update(state)

    monkeypatch.setattr(
        app_main,
        "_load_runtime_dependencies",
        lambda: (_FakeGraph(), _fake_init_state),
    )
    monkeypatch.setattr(app_main, "_persist_session_state", _capture_persist)
    monkeypatch.setattr(app_main, "_load_memory_profile", lambda user_id: None)
    monkeypatch.setattr(app_main, "_emit_proactive_memory_greeting", lambda user_id, profile: None)

    inputs = iter(["hello", "second turn", "exit"])
    code = app_main.main(["--user", "alex"], input_reader=lambda _prompt: next(inputs))

    assert code == 0
    assert observed_state["conversation_history"] == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "Echo: hello"},
        {"role": "user", "content": "second turn"},
        {"role": "assistant", "content": "Echo: second turn"},
    ]


def test_memory_service_persists_full_conversation_history_snapshot(tmp_path: Path) -> None:
    from concierge.agents.memory_service import MemoryService

    sessions_dir = tmp_path / "memory" / "sessions"
    service = MemoryService(
        profiles_dir=tmp_path / "memory" / "profiles",
        sessions_dir=sessions_dir,
        output_writer=io.StringIO(),
    )
    state = {
        "session_id": "session-story-46",
        "user_id": "alex",
        "conversation_history": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "beach options"},
        ],
        "turn_id": 2,
    }

    written = service.write_session_state(
        state,
        timestamp_start="2026-02-25T10:00:00Z",
        timestamp_end="2026-02-25T10:01:00Z",
    )
    assert written is not None
    payload = json.loads(written.read_text(encoding="utf-8"))
    assert payload["conversation_history"] == state["conversation_history"]


def test_memory_service_load_profile_reads_cached_context_from_previous_session(
    tmp_path: Path,
) -> None:
    from concierge.agents.memory_service import MemoryService

    profiles_dir = tmp_path / "memory" / "profiles"
    sessions_dir = tmp_path / "memory" / "sessions"
    profiles_dir.mkdir(parents=True)
    session_dir = sessions_dir / "session-prev"
    session_dir.mkdir(parents=True)

    (profiles_dir / "alex.json").write_text(
        json.dumps(
            {
                "user_id": "alex",
                "preferred_name": "Alex",
                "past_trips": [],
                "preferences": {"preferred_regions": ["Southeast Asia"]},
            }
        ),
        encoding="utf-8",
    )
    cached_context = {
        "topic": "Southeast Asia beach destinations 2026",
        "session_id": "session-prev",
        "cached_at": "2026-02-25T09:00:00Z",
    }
    (session_dir / "state.json").write_text(
        json.dumps(
            {
                "session_id": "session-prev",
                "user_id": "alex",
                "timestamp_start": "2026-02-25T08:00:00Z",
                "timestamp_end": "2026-02-25T09:00:00Z",
                "conversation_history": [],
                "turn_count": 2,
                "cached_research_session": cached_context,
            }
        ),
        encoding="utf-8",
    )

    loaded = MemoryService(
        profiles_dir=profiles_dir,
        sessions_dir=sessions_dir,
        output_writer=io.StringIO(),
    ).load_profile("alex")

    assert loaded is not None
    assert loaded["cached_research_session"] == cached_context


def test_proactive_greeting_can_use_cached_research_session_context() -> None:
    from concierge.agents.memory_service import build_proactive_greeting

    greeting = build_proactive_greeting(
        {
            "preferred_name": "Alex",
            "past_trips": [],
            "cached_research_session": {
                "topic": "Southeast Asia beach destinations 2026",
                "session_id": "session-prev",
            },
        },
        "alex",
    )

    assert greeting == (
        "Welcome back, Alex. Last time we explored Southeast Asia beach destinations 2026. "
        "Want to continue where you left off?"
    )


def test_persisted_file_is_authoritative_after_post_write_memory_mutation(
    tmp_path: Path,
) -> None:
    from concierge.agents.memory_service import MemoryService

    service = MemoryService(
        profiles_dir=tmp_path / "memory" / "profiles",
        sessions_dir=tmp_path / "memory" / "sessions",
        output_writer=io.StringIO(),
    )
    state = {
        "session_id": "session-authority",
        "user_id": "alex",
        "conversation_history": [{"role": "user", "content": "turn1"}],
        "turn_id": 1,
    }
    written = service.write_session_state(
        state,
        timestamp_start="2026-02-25T10:00:00Z",
        timestamp_end="2026-02-25T10:01:00Z",
    )
    assert written is not None

    state["conversation_history"].append({"role": "assistant", "content": "mutated"})
    payload = json.loads(written.read_text(encoding="utf-8"))
    assert payload["conversation_history"] == [{"role": "user", "content": "turn1"}]


def test_persist_session_state_degrades_gracefully_on_write_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    app_main = importlib.import_module("main")

    class _FailingMemoryService:
        def write_session_state(self, state, timestamp_start, timestamp_end):  # noqa: ANN001
            del state, timestamp_start, timestamp_end
            raise OSError("disk full")

    original_import = app_main.importlib.import_module

    def _patched_import(name: str):
        if name == "concierge.agents.memory_service":
            return SimpleNamespace(MemoryService=lambda: _FailingMemoryService())
        return original_import(name)

    monkeypatch.setattr(app_main.importlib, "import_module", _patched_import)
    monkeypatch.setattr(app_main, "_iso_utc_now", lambda: "2026-02-25T10:00:01Z")

    state: dict[str, object] = {
        "session_id": "session-fail",
        "user_id": "alex",
        "conversation_history": [],
        "turn_id": 0,
        "human_handoff": False,
    }
    app_main._persist_session_state(state, "2026-02-25T10:00:00Z")

    assert state["human_handoff"] is True
    assert state["error"] == "Session could not be saved — a human can assist"
    assert "Session could not be saved — a human can assist" in capsys.readouterr().out


def test_main_continues_and_exits_cleanly_when_session_persist_fails(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    app_main = importlib.import_module("main")
    monkeypatch.setattr(app_main.sys, "version_info", (3, 11, 9))
    monkeypatch.setattr(app_main, "_iso_utc_now", lambda: "2026-02-25T10:00:00Z")

    observed_inputs: list[str] = []

    class _FakeGraph:
        def invoke(self, state: dict[str, object]) -> dict[str, object]:
            observed_inputs.append(str(state.get("current_input") or ""))
            history = list(state.get("conversation_history") or [])
            history.append({"role": "user", "content": str(state.get("current_input") or "")})
            updated = dict(state)
            updated["conversation_history"] = history
            return updated

    class _FailingMemoryService:
        def write_session_state(self, state, timestamp_start, timestamp_end):  # noqa: ANN001
            del state, timestamp_start, timestamp_end
            raise OSError("disk full")

    original_import = app_main.importlib.import_module

    def _patched_import(name: str):
        if name == "concierge.agents.memory_service":
            return SimpleNamespace(MemoryService=lambda: _FailingMemoryService())
        return original_import(name)

    monkeypatch.setattr(
        app_main,
        "_load_runtime_dependencies",
        lambda: (_FakeGraph(), lambda **kwargs: dict(kwargs, conversation_history=[])),
    )
    monkeypatch.setattr(app_main.importlib, "import_module", _patched_import)
    monkeypatch.setattr(app_main, "_load_memory_profile", lambda user_id: None)
    monkeypatch.setattr(app_main, "_emit_proactive_memory_greeting", lambda user_id, profile: None)

    inputs = iter(["hello", "exit"])
    code = app_main.main(["--user", "alex"], input_reader=lambda _prompt: next(inputs))

    assert code == 0
    assert observed_inputs == ["hello"]
    assert "Session could not be saved — a human can assist" in capsys.readouterr().out
