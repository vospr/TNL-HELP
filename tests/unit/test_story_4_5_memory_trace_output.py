from __future__ import annotations

import io
import json
from pathlib import Path

import pytest


def test_memory_service_traces_profile_loaded_with_trip_count(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from concierge.agents import memory_service as memory_service_module

    profiles_dir = tmp_path / "memory" / "profiles"
    profiles_dir.mkdir(parents=True)
    (profiles_dir / "alex.json").write_text(
        json.dumps({"user_id": "alex", "past_trips": [{}, {}]}),
        encoding="utf-8",
    )

    trace_calls: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        memory_service_module,
        "trace",
        lambda node_name, outcome=None, **fields: trace_calls.append((node_name, dict(fields))),
    )

    service = memory_service_module.MemoryService(
        profiles_dir=profiles_dir,
        output_writer=io.StringIO(),
    )
    loaded = service.load_profile("alex")

    assert loaded is not None
    assert trace_calls == [
        (
            "memory",
            {
                "event": "profile_loaded",
                "user_id": "alex",
                "past_trips_count": 2,
            },
        )
    ]


def test_memory_service_traces_profile_not_found(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from concierge.agents import memory_service as memory_service_module

    trace_calls: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        memory_service_module,
        "trace",
        lambda node_name, outcome=None, **fields: trace_calls.append((node_name, dict(fields))),
    )

    service = memory_service_module.MemoryService(
        profiles_dir=tmp_path / "memory" / "profiles",
        output_writer=io.StringIO(),
    )
    loaded = service.load_profile("missing-user")

    assert loaded is None
    assert trace_calls == [
        ("memory", {"event": "profile_not_found", "user_id": "missing-user"})
    ]


def test_memory_service_traces_session_written(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from concierge.agents import memory_service as memory_service_module

    trace_calls: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        memory_service_module,
        "trace",
        lambda node_name, outcome=None, **fields: trace_calls.append((node_name, dict(fields))),
    )

    service = memory_service_module.MemoryService(
        profiles_dir=tmp_path / "memory" / "profiles",
        sessions_dir=tmp_path / "memory" / "sessions",
        output_writer=io.StringIO(),
    )
    written = service.write_session_state(
        {
            "session_id": "session-story-45",
            "user_id": "alex",
            "conversation_history": [{"role": "user", "content": "hello"}],
            "turn_id": 1,
        },
        timestamp_start="2026-02-25T10:00:00Z",
        timestamp_end="2026-02-25T10:00:10Z",
    )

    assert written is not None
    assert trace_calls == [
        (
            "memory",
            {
                "event": "session_written",
                "session_id": "session-story-45",
                "turn_count": 1,
            },
        )
    ]


def test_trace_renders_memory_label_with_outcome_oriented_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from concierge import trace as trace_module

    buffer = io.StringIO()
    monkeypatch.setattr(trace_module, "_trace_writer", buffer)

    trace_module.trace(
        "memory",
        event="profile_loaded",
        user_id="alex",
        past_trips_count=2,
    )

    assert (
        buffer.getvalue().strip()
        == "[memory] event=profile_loaded user_id=alex past_trips_count=2"
    )
