from __future__ import annotations

import json
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any, TextIO

from concierge.trace import trace


class MemoryService:
    def __init__(
        self,
        profiles_dir: Path | None = None,
        sessions_dir: Path | None = None,
        output_writer: TextIO | None = None,
    ) -> None:
        base_dir = Path(__file__).resolve().parent
        repo_root = base_dir.parents[2]
        if profiles_dir is None:
            profiles_dir = repo_root / "memory" / "profiles"
        if sessions_dir is None:
            sessions_dir = repo_root / "memory" / "sessions"
        self._profiles_dir = profiles_dir
        self._sessions_dir = sessions_dir
        self._output_writer = output_writer or sys.stdout

    def load_profile(self, user_id: str) -> dict[str, Any] | None:
        profile_path = self._profiles_dir / f"{user_id}.json"
        try:
            with profile_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except FileNotFoundError:
            self._warn_not_found(user_id)
            return None
        except json.JSONDecodeError:
            self._warn_not_found(user_id)
            return None

        if not isinstance(payload, dict):
            self._warn_not_found(user_id)
            return None
        cached_research_session = self._load_latest_cached_research_session(user_id)
        if isinstance(cached_research_session, dict):
            payload["cached_research_session"] = cached_research_session
        past_trips = payload.get("past_trips")
        past_trips_count = len(past_trips) if isinstance(past_trips, list) else 0
        trace(
            "memory",
            event="profile_loaded",
            user_id=user_id,
            past_trips_count=past_trips_count,
        )
        return payload

    def _warn_not_found(self, user_id: str) -> None:
        self._output_writer.write(
            f"[MEMORY] Profile for {user_id} not found — starting fresh session\n"
        )
        self._output_writer.flush()
        trace("memory", event="profile_not_found", user_id=user_id)

    def write_session_state(
        self,
        state: dict[str, Any],
        timestamp_start: str,
        timestamp_end: str,
    ) -> Path | None:
        session_id = str(state.get("session_id") or "").strip()
        user_id = str(state.get("user_id") or "").strip()
        if not session_id or not user_id:
            return None

        history = state.get("conversation_history")
        if not isinstance(history, list):
            history = []
        turn_count = int(state.get("turn_id") or 0)

        payload = {
            "session_id": session_id,
            "user_id": user_id,
            "timestamp_start": timestamp_start,
            "timestamp_end": timestamp_end,
            "conversation_history": history,
            "turn_count": turn_count,
        }
        cached_research_session = self._extract_cached_research_session(state)
        if isinstance(cached_research_session, dict):
            payload["cached_research_session"] = cached_research_session

        session_dir = self._sessions_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        session_file = session_dir / "state.json"
        with session_file.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        trace(
            "memory",
            event="session_written",
            session_id=session_id,
            turn_count=turn_count,
        )
        return session_file

    def _extract_cached_research_session(self, state: dict[str, Any]) -> dict[str, Any] | None:
        profile = state.get("memory_profile")
        if isinstance(profile, dict):
            profile_cached = profile.get("cached_research_session")
            if isinstance(profile_cached, dict):
                return profile_cached

        state_cached = state.get("cached_research_session")
        if isinstance(state_cached, dict):
            return state_cached
        return None

    def _load_latest_cached_research_session(self, user_id: str) -> dict[str, Any] | None:
        if not self._sessions_dir.exists():
            return None

        latest_context: dict[str, Any] | None = None
        latest_sort_key = ""
        for state_file in self._sessions_dir.glob("*/state.json"):
            try:
                with state_file.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            if str(payload.get("user_id") or "").strip() != user_id:
                continue

            cached_research_session = payload.get("cached_research_session")
            if not isinstance(cached_research_session, dict):
                continue

            sort_key = str(
                payload.get("timestamp_end")
                or payload.get("timestamp_start")
                or state_file.parent.name
            )
            if latest_context is None or sort_key > latest_sort_key:
                latest_context = cached_research_session
                latest_sort_key = sort_key
        return latest_context


def build_proactive_greeting(profile: dict[str, Any], user_id: str) -> str | None:
    preferred_name = str(profile.get("preferred_name") or user_id).strip() or user_id
    past_trips = profile.get("past_trips")
    if not isinstance(past_trips, list) or not past_trips:
        cached_research_session = profile.get("cached_research_session")
        if not isinstance(cached_research_session, dict):
            return None
        topic = str(cached_research_session.get("topic") or "").strip()
        if not topic:
            return None
        return (
            f"Welcome back, {preferred_name}. "
            f"Last time we explored {topic}. "
            "Want to continue where you left off?"
        )

    first_trip = past_trips[0]
    if not isinstance(first_trip, dict):
        return None

    destination_raw = str(first_trip.get("destination") or "").strip()
    if not destination_raw:
        return None
    destination = destination_raw.split(",")[0].strip() or destination_raw

    dates_raw = str(first_trip.get("dates") or "").strip()
    month_name = _month_from_trip_dates(dates_raw)
    if month_name is None:
        return None

    preferred_regions = (
        profile.get("preferences", {}).get("preferred_regions")
        if isinstance(profile.get("preferences"), dict)
        else None
    )
    if isinstance(preferred_regions, list) and preferred_regions:
        region = str(preferred_regions[0] or "").strip()
    else:
        region = ""
    if not region:
        return None

    return (
        f"Welcome back, {preferred_name}. "
        f"Based on your {month_name} trip to {destination}, "
        f"you might be interested in upcoming deals to {region}."
    )


def _month_from_trip_dates(value: str) -> str | None:
    match = re.search(r"(\d{4})-(\d{2})-(\d{2})", value)
    if not match:
        return None
    year, month, day = match.groups()
    try:
        parsed = date(int(year), int(month), int(day))
    except ValueError:
        return None
    return parsed.strftime("%B")
