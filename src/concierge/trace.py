from __future__ import annotations

import sys
from typing import Any, TextIO


TRACE_ALLOWLIST = {
    "intent",
    "confidence",
    "route",
    "model",
    "session_id",
    "turn_id",
    "status",
    "event",
    "stage",
    "reason",
    "results_count",
    "sources_cited",
    "threshold",
    "clarification_count",
    "query",
    "turn_count",
    "token_count",
    "message_count",
    "profile",
    "trips",
    "past_trips_count",
    "user_id",
    "outcome",
}

TRACE_DENYLIST = {
    "current_input",
    "memory_profile",
    "conversation_history",
    "rag_results",
    "research_results",
    "error",
}

_trace_writer: TextIO = sys.stdout


def _is_sensitive_field(field_name: str) -> bool:
    normalized = field_name.lower()
    return normalized in TRACE_DENYLIST or "api_key" in normalized


def _resolve_trace_writer() -> TextIO:
    global _trace_writer
    if getattr(_trace_writer, "closed", False):
        _trace_writer = sys.stdout
    return _trace_writer


def trace(node_name: str, outcome: str | None = None, **fields: Any) -> None:
    for key in fields:
        if _is_sensitive_field(key):
            raise ValueError(f"{key} is not emittable")

    safe_fields = {
        key: value
        for key, value in fields.items()
        if key in TRACE_ALLOWLIST and key not in TRACE_DENYLIST
    }
    details = " ".join(f"{key}={value}" for key, value in safe_fields.items())

    line = f"[{node_name}]"
    if details:
        line += f" {details}"
    if outcome:
        line += f" -> {outcome}"

    writer = _resolve_trace_writer()
    try:
        writer.write(line + "\n")
        writer.flush()
    except ValueError:
        _trace_writer = sys.stdout
        _trace_writer.write(line + "\n")
        _trace_writer.flush()
