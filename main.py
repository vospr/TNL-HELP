from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
import os
import sys
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4


_MIN_PYTHON = (3, 11)
_EXIT_COMMANDS = {"exit", "quit"}
_FAST_MODE_BANNER = "[FAST MODE] Using claude-haiku-4-5 across all agents - not the demo path"
_SESSION_PERSIST_FAILURE_MESSAGE = "Session could not be saved — a human can assist"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TNL-HELP CLI entry point")
    parser.add_argument("--user", required=True, help="User id for session state initialization")
    parser.add_argument(
        "--fast-mode",
        action="store_true",
        help="Use claude-haiku-4-5 across agents for faster development iterations.",
    )
    return parser


def _ensure_supported_python() -> bool:
    if tuple(sys.version_info[:2]) >= _MIN_PYTHON:
        return True
    print("Python 3.11+ required", file=sys.stderr)
    return False


def _ensure_src_on_path() -> None:
    repo_root = Path(__file__).resolve().parent
    src_path = str(repo_root / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


def _load_runtime_dependencies() -> tuple[Any, Callable[..., dict[str, Any]]]:
    _ensure_src_on_path()

    validate_config = importlib.import_module("validate_config")
    validate_config.run_checks()

    graph_module = importlib.import_module("concierge.graph")
    state_module = importlib.import_module("concierge.state")
    return graph_module.compiled_graph, state_module.initialize_state


def _load_memory_profile(user_id: str) -> dict[str, Any] | None:
    memory_service_module = importlib.import_module("concierge.agents.memory_service")
    memory_service = memory_service_module.MemoryService()
    return memory_service.load_profile(user_id)


def _emit_proactive_memory_greeting(user_id: str, profile: dict[str, Any] | None) -> None:
    if not isinstance(profile, dict):
        return

    memory_service_module = importlib.import_module("concierge.agents.memory_service")
    greeting = memory_service_module.build_proactive_greeting(profile, user_id)
    if not greeting:
        return

    print(greeting)
    trace_module = importlib.import_module("concierge.trace")
    trace_module.trace("memory", event="greeting_fired", user_id=user_id)


def _iso_utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _persist_session_state(state: dict[str, Any], timestamp_start: str) -> None:
    session_id = str(state.get("session_id") or "").strip()
    user_id = str(state.get("user_id") or "").strip()
    if not session_id or not user_id:
        return

    memory_service_module = importlib.import_module("concierge.agents.memory_service")
    memory_service = memory_service_module.MemoryService()
    try:
        written = memory_service.write_session_state(
            state=state,
            timestamp_start=timestamp_start,
            timestamp_end=_iso_utc_now(),
        )
    except OSError:
        written = None

    if written is None:
        state["human_handoff"] = True
        state["error"] = _SESSION_PERSIST_FAILURE_MESSAGE
        print(_SESSION_PERSIST_FAILURE_MESSAGE)


def _run_loop(
    compiled_graph: Any,
    state: dict[str, Any],
    input_reader: Callable[[str], str],
) -> int:
    while True:
        try:
            user_input = input_reader("")
        except EOFError:
            return 0
        except KeyboardInterrupt:
            print()
            return 0

        normalized = user_input.strip()
        if not normalized:
            continue
        if normalized.lower() in _EXIT_COMMANDS:
            return 0

        state["turn_id"] = int(state.get("turn_id", 0)) + 1
        state["current_input"] = normalized

        result = compiled_graph.invoke(state)
        if isinstance(result, dict):
            state.clear()
            state.update(result)


def main(argv: list[str] | None = None, input_reader: Callable[[str], str] = input) -> int:
    if not _ensure_supported_python():
        return 1

    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.fast_mode:
        os.environ["FAST_MODE"] = "1"

    try:
        compiled_graph, initialize_state = _load_runtime_dependencies()
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 1

    if args.fast_mode:
        print(_FAST_MODE_BANNER)

    state = initialize_state(
        user_id=args.user,
        session_id=f"session-{uuid4().hex}",
        current_input="",
        turn_id=0,
    )
    timestamp_start = _iso_utc_now()
    memory_profile = _load_memory_profile(args.user)
    state["memory_profile"] = memory_profile
    _emit_proactive_memory_greeting(args.user, memory_profile)
    exit_code = _run_loop(compiled_graph=compiled_graph, state=state, input_reader=input_reader)
    _persist_session_state(state, timestamp_start)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
