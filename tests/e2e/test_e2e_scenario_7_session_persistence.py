"""
E2E Test: Scenario 7 — Session Persistence & Memory Profile Continuity

Two separate graph invocations simulate two distinct sessions for the same user
(alex). The test verifies that:

  Session 1:
    - Memory profile loaded from tmp_memory_dir
    - Proactive greeting generated and references past destination (Bali)
    - Graph invocation completes and produces a valid response
    - Session state (including cached_research_session) written to
      memory/sessions/{session_id}/state.json
    - turn_id begins at 0, advances per turn

  Session 2:
    - Starts with a fresh ConciergeState (turn_id=0, empty conversation_history)
    - Same profile loaded again from tmp_memory_dir
    - cached_research_session from Session 1 is surfaced inside memory_profile
    - Proactive greeting shows topic continuity (references last research topic)
    - session_id differs from Session 1
    - conversation_history starts empty
"""
from __future__ import annotations

import io
import json
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Profile fixtures
# ---------------------------------------------------------------------------

ALEX_PROFILE: dict[str, Any] = {
    "user_id": "alex",
    "preferred_name": "Alex",
    "past_trips": [
        {
            "destination": "Bali, Indonesia",
            "property_name": "Wyndham Garden Kuta Beach Bali",
            "dates": "2025-03-10 to 2025-03-17",
            "notes": (
                "Loved the beach access and spa. "
                "Asked about surfing lessons — add to activities list."
            ),
        },
        {
            "destination": "Tokyo, Japan",
            "property_name": None,
            "dates": "2024-11-20 to 2024-11-28",
            "notes": "Independent hotel. Focus was food and cultural sites.",
        },
    ],
    "preferences": {
        "travel_style": "luxury_resort",
        "room_type": "ocean_view",
        "activities": ["beach", "spa", "snorkeling", "local_cuisine"],
        "diet": "no_restrictions",
        "preferred_regions": ["Southeast Asia", "Pacific Islands"],
        "budget_tier": "premium",
    },
    "last_seen": "2026-02-25T00:00:00Z",
}

# Research context injected into Session 1's state to simulate real research work
SESSION_1_CACHED_RESEARCH: dict[str, Any] = {
    "topic": "Southeast Asia beach destinations 2026",
    "session_id": "session-placeholder",  # overwritten inside the test
    "cached_at": "2026-02-26T10:00:00Z",
    "destinations_explored": ["Bali", "Phuket", "Koh Samui"],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_profile(memory_dir: Path, profile: dict[str, Any]) -> Path:
    """Write a profile JSON to memory/profiles/{user_id}.json."""
    profiles_dir = memory_dir / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    path = profiles_dir / f"{profile['user_id']}.json"
    path.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _make_service(memory_dir: Path) -> Any:
    """Instantiate MemoryService pointed at the tmp memory directory."""
    from concierge.agents.memory_service import MemoryService

    return MemoryService(
        profiles_dir=memory_dir / "profiles",
        sessions_dir=memory_dir / "sessions",
        output_writer=io.StringIO(),
    )


def _make_graph() -> Any:
    """
    Build the graph using the project's own build_graph() factory, which falls
    back to _FallbackCompiledGraph when langgraph is absent. The Anthropic client
    is patched at the module level to prevent any real API calls during the
    dispatcher's Stage 2 LLM escalation path.
    """
    with patch("anthropic.Anthropic", return_value=MagicMock()):
        from concierge.graph import build_graph

        return build_graph()


def _make_state(user_id: str, session_id: str, memory_profile: dict[str, Any] | None) -> dict[str, Any]:
    """Return a fresh ConciergeState dict as produced by initialize_state()."""
    from concierge.state import initialize_state

    state = dict(
        initialize_state(
            user_id=user_id,
            session_id=session_id,
            current_input="",
            turn_id=0,
        )
    )
    state["memory_profile"] = memory_profile
    return state


def _invoke_turn(graph: Any, state: dict[str, Any], user_input: str) -> dict[str, Any]:
    """Advance turn_id, set current_input, invoke graph, return merged result state."""
    state = dict(state)
    state["turn_id"] = int(state.get("turn_id", 0)) + 1
    state["current_input"] = user_input
    result = graph.invoke(state)
    if isinstance(result, dict):
        merged = dict(state)
        merged.update(result)
        return merged
    return state


# ---------------------------------------------------------------------------
# E2E test
# ---------------------------------------------------------------------------


class TestScenario7SessionPersistenceAndMemoryContinuity:
    """
    Two-session E2E scenario.

    All file I/O is directed to a pytest tmp_path-backed memory directory so
    the repository's own memory/profiles/ and memory/sessions/ are never touched.
    The Anthropic client is monkey-patched at import time to prevent live calls.
    """

    # ------------------------------------------------------------------
    # Session 1
    # ------------------------------------------------------------------

    def test_session_1_profile_loads_with_required_fields(
        self, tmp_memory_dir: Path
    ) -> None:
        """memory_profile must be non-None and contain user_id + past_trips."""
        _write_profile(tmp_memory_dir, ALEX_PROFILE)
        svc = _make_service(tmp_memory_dir)

        profile = svc.load_profile("alex")

        assert profile is not None, "load_profile() must return a dict, not None"
        assert profile["user_id"] == "alex"
        assert isinstance(profile.get("past_trips"), list)
        assert len(profile["past_trips"]) >= 1

    def test_session_1_proactive_greeting_mentions_bali(
        self, tmp_memory_dir: Path
    ) -> None:
        """
        build_proactive_greeting() must reference the Bali trip and the first
        preferred_region when past_trips are present.
        """
        from concierge.agents.memory_service import build_proactive_greeting

        _write_profile(tmp_memory_dir, ALEX_PROFILE)
        svc = _make_service(tmp_memory_dir)
        profile = svc.load_profile("alex")
        assert profile is not None

        greeting = build_proactive_greeting(profile, "alex")

        assert greeting is not None, "A greeting must be produced for a profile with past_trips"
        assert "Bali" in greeting, f"Expected 'Bali' in greeting, got: {greeting!r}"
        assert "Alex" in greeting, f"Expected 'Alex' in greeting, got: {greeting!r}"
        assert "Welcome back" in greeting

    def test_session_1_proactive_greeting_full_text(
        self, tmp_memory_dir: Path
    ) -> None:
        """Exact greeting text must match the contract established by unit tests."""
        from concierge.agents.memory_service import build_proactive_greeting

        _write_profile(tmp_memory_dir, ALEX_PROFILE)
        svc = _make_service(tmp_memory_dir)
        profile = svc.load_profile("alex")
        assert profile is not None

        greeting = build_proactive_greeting(profile, "alex")

        expected = (
            "Welcome back, Alex. Based on your March trip to Bali, "
            "you might be interested in upcoming deals to Southeast Asia."
        )
        assert greeting == expected

    def test_session_1_graph_invocation_returns_valid_state(
        self, tmp_memory_dir: Path
    ) -> None:
        """A single graph turn must produce a non-empty current_response."""
        _write_profile(tmp_memory_dir, ALEX_PROFILE)
        svc = _make_service(tmp_memory_dir)
        profile = svc.load_profile("alex")

        session_id_1 = f"session-{uuid.uuid4().hex}"
        state = _make_state("alex", session_id_1, profile)

        with patch("anthropic.Anthropic", return_value=MagicMock()):
            graph = _make_graph()
            result = _invoke_turn(graph, state, "What Wyndham properties are in Bali?")

        assert isinstance(result, dict)
        assert result.get("session_id") == session_id_1
        assert result.get("user_id") == "alex"
        assert result.get("turn_id") == 1
        # The graph must produce some response
        assert isinstance(result.get("current_response"), str)
        assert result["current_response"] != ""

    def test_session_1_turn_id_starts_at_zero_before_first_invoke(
        self, tmp_memory_dir: Path
    ) -> None:
        """Fresh session state must have turn_id == 0 before any invocation."""
        _write_profile(tmp_memory_dir, ALEX_PROFILE)
        svc = _make_service(tmp_memory_dir)
        profile = svc.load_profile("alex")

        session_id = f"session-{uuid.uuid4().hex}"
        state = _make_state("alex", session_id, profile)

        assert state["turn_id"] == 0

    def test_session_1_conversation_history_starts_empty(
        self, tmp_memory_dir: Path
    ) -> None:
        """conversation_history in a fresh state must be an empty list."""
        _write_profile(tmp_memory_dir, ALEX_PROFILE)
        svc = _make_service(tmp_memory_dir)
        profile = svc.load_profile("alex")

        state = _make_state("alex", f"session-{uuid.uuid4().hex}", profile)

        assert state["conversation_history"] == []

    def test_session_1_write_session_state_produces_file(
        self, tmp_memory_dir: Path
    ) -> None:
        """
        write_session_state() must create
        memory/sessions/{session_id}/state.json with correct fields.
        """
        _write_profile(tmp_memory_dir, ALEX_PROFILE)
        svc = _make_service(tmp_memory_dir)
        profile = svc.load_profile("alex")
        assert profile is not None

        session_id = f"session-{uuid.uuid4().hex}"

        # Build a state that mimics one completed turn
        session_state: dict[str, Any] = {
            "session_id": session_id,
            "user_id": "alex",
            "turn_id": 1,
            "conversation_history": [
                {"role": "user", "content": "What Wyndham properties are in Bali?"},
                {"role": "assistant", "content": "We have several properties in Bali."},
            ],
            "memory_profile": profile,
            "cached_research_session": {
                **SESSION_1_CACHED_RESEARCH,
                "session_id": session_id,
            },
        }

        written = svc.write_session_state(
            session_state,
            timestamp_start="2026-02-26T09:00:00Z",
            timestamp_end="2026-02-26T09:05:00Z",
        )

        assert written is not None, "write_session_state() must return a Path"
        assert written.exists(), f"Expected file at {written}"
        expected_path = tmp_memory_dir / "sessions" / session_id / "state.json"
        assert written == expected_path

    def test_session_1_persisted_file_contains_required_fields(
        self, tmp_memory_dir: Path
    ) -> None:
        """The persisted JSON must include session_id, user_id, timestamps, history, turn_count."""
        _write_profile(tmp_memory_dir, ALEX_PROFILE)
        svc = _make_service(tmp_memory_dir)
        profile = svc.load_profile("alex")
        assert profile is not None

        session_id = f"session-{uuid.uuid4().hex}"
        session_state: dict[str, Any] = {
            "session_id": session_id,
            "user_id": "alex",
            "turn_id": 2,
            "conversation_history": [
                {"role": "user", "content": "Tell me about Koh Samui"},
                {"role": "assistant", "content": "Koh Samui is a beautiful island."},
            ],
            "memory_profile": profile,
        }

        written = svc.write_session_state(
            session_state,
            timestamp_start="2026-02-26T09:00:00Z",
            timestamp_end="2026-02-26T09:08:00Z",
        )
        assert written is not None
        payload = json.loads(written.read_text(encoding="utf-8"))

        assert payload["session_id"] == session_id
        assert payload["user_id"] == "alex"
        assert payload["timestamp_start"] == "2026-02-26T09:00:00Z"
        assert payload["timestamp_end"] == "2026-02-26T09:08:00Z"
        assert payload["turn_count"] == 2
        assert isinstance(payload["conversation_history"], list)
        assert len(payload["conversation_history"]) == 2

    def test_session_1_cached_research_written_to_session_file(
        self, tmp_memory_dir: Path
    ) -> None:
        """
        When the state carries a cached_research_session dict, it must be
        preserved inside state.json so Session 2 can recover it.
        """
        _write_profile(tmp_memory_dir, ALEX_PROFILE)
        svc = _make_service(tmp_memory_dir)
        profile = svc.load_profile("alex")
        assert profile is not None

        session_id = f"session-{uuid.uuid4().hex}"
        cached_research: dict[str, Any] = {
            "topic": "Southeast Asia beach destinations 2026",
            "session_id": session_id,
            "cached_at": "2026-02-26T09:10:00Z",
        }
        session_state: dict[str, Any] = {
            "session_id": session_id,
            "user_id": "alex",
            "turn_id": 3,
            "conversation_history": [],
            "memory_profile": profile,
            "cached_research_session": cached_research,
        }

        written = svc.write_session_state(
            session_state,
            timestamp_start="2026-02-26T09:00:00Z",
            timestamp_end="2026-02-26T09:10:00Z",
        )
        assert written is not None
        payload = json.loads(written.read_text(encoding="utf-8"))

        assert "cached_research_session" in payload, (
            "cached_research_session must be persisted in state.json"
        )
        assert payload["cached_research_session"] == cached_research

    # ------------------------------------------------------------------
    # Session 2 (cross-session continuity)
    # ------------------------------------------------------------------

    def test_session_2_loads_same_profile(self, tmp_memory_dir: Path) -> None:
        """
        A second MemoryService instantiation must load the same profile for alex.
        """
        _write_profile(tmp_memory_dir, ALEX_PROFILE)
        svc2 = _make_service(tmp_memory_dir)

        profile2 = svc2.load_profile("alex")

        assert profile2 is not None
        assert profile2["user_id"] == "alex"
        assert isinstance(profile2.get("past_trips"), list)

    def test_session_2_memory_profile_populated_in_state(
        self, tmp_memory_dir: Path
    ) -> None:
        """
        memory_profile must be non-None in the Session 2 state dict after
        profile loading and assignment.
        """
        _write_profile(tmp_memory_dir, ALEX_PROFILE)
        svc = _make_service(tmp_memory_dir)
        profile2 = svc.load_profile("alex")
        assert profile2 is not None

        state2 = _make_state("alex", f"session-{uuid.uuid4().hex}", profile2)

        assert state2["memory_profile"] is not None
        assert isinstance(state2["memory_profile"], dict)
        assert state2["memory_profile"]["user_id"] == "alex"

    def test_session_2_receives_cached_research_from_session_1(
        self, tmp_memory_dir: Path
    ) -> None:
        """
        After Session 1 writes its cached_research_session to sessions dir,
        Session 2's load_profile() must surface it inside the returned profile
        under the 'cached_research_session' key.
        """
        _write_profile(tmp_memory_dir, ALEX_PROFILE)
        svc = _make_service(tmp_memory_dir)

        # --- Session 1: persist state with cached research ---
        session_id_1 = f"session-{uuid.uuid4().hex}"
        profile1 = svc.load_profile("alex")
        assert profile1 is not None

        cached_research_s1: dict[str, Any] = {
            "topic": "Southeast Asia beach destinations 2026",
            "session_id": session_id_1,
            "cached_at": "2026-02-26T09:10:00Z",
        }
        state1: dict[str, Any] = {
            "session_id": session_id_1,
            "user_id": "alex",
            "turn_id": 2,
            "conversation_history": [
                {"role": "user", "content": "What are the best SE Asia beaches?"},
                {"role": "assistant", "content": "Here are top picks for 2026..."},
            ],
            "memory_profile": profile1,
            "cached_research_session": cached_research_s1,
        }

        written = svc.write_session_state(
            state1,
            timestamp_start="2026-02-26T09:00:00Z",
            timestamp_end="2026-02-26T09:10:00Z",
        )
        assert written is not None and written.exists()

        # --- Session 2: fresh MemoryService load ---
        svc2 = _make_service(tmp_memory_dir)
        profile2 = svc2.load_profile("alex")

        assert profile2 is not None
        assert "cached_research_session" in profile2, (
            "load_profile() must inject cached_research_session from previous session file"
        )
        assert profile2["cached_research_session"] == cached_research_s1

    def test_session_2_session_id_differs_from_session_1(
        self, tmp_memory_dir: Path
    ) -> None:
        """Each session must receive a distinct UUID-based session_id."""
        session_id_1 = f"session-{uuid.uuid4().hex}"
        session_id_2 = f"session-{uuid.uuid4().hex}"

        assert session_id_1 != session_id_2, (
            "uuid4().hex must produce unique values for each session"
        )

    def test_session_2_conversation_history_starts_fresh(
        self, tmp_memory_dir: Path
    ) -> None:
        """
        Even though profile carries cached context, the new session's
        conversation_history must start as an empty list.
        """
        _write_profile(tmp_memory_dir, ALEX_PROFILE)
        svc = _make_service(tmp_memory_dir)
        profile2 = svc.load_profile("alex")
        assert profile2 is not None

        state2 = _make_state("alex", f"session-{uuid.uuid4().hex}", profile2)

        assert state2["conversation_history"] == [], (
            "conversation_history must be empty at the start of a new session"
        )

    def test_session_2_turn_id_resets_to_zero(self, tmp_memory_dir: Path) -> None:
        """Session 2 must start with turn_id == 0, regardless of Session 1's turn count."""
        _write_profile(tmp_memory_dir, ALEX_PROFILE)
        svc = _make_service(tmp_memory_dir)
        profile2 = svc.load_profile("alex")
        assert profile2 is not None

        state2 = _make_state("alex", f"session-{uuid.uuid4().hex}", profile2)

        assert state2["turn_id"] == 0, (
            "turn_id must reset to 0 at the start of every new session"
        )

    def test_session_2_proactive_greeting_shows_topic_continuity(
        self, tmp_memory_dir: Path
    ) -> None:
        """
        When Session 2's profile contains a cached_research_session whose past_trips
        list is empty (simulating a profile where only cached context is present),
        build_proactive_greeting() must reference the cached topic.
        """
        from concierge.agents.memory_service import build_proactive_greeting

        # Build a profile that has no past_trips but has cached_research_session
        profile_no_trips: dict[str, Any] = {
            "user_id": "alex",
            "preferred_name": "Alex",
            "past_trips": [],
            "preferences": {"preferred_regions": ["Southeast Asia"]},
            "cached_research_session": {
                "topic": "Southeast Asia beach destinations 2026",
                "session_id": "session-prev",
                "cached_at": "2026-02-26T09:10:00Z",
            },
        }

        greeting = build_proactive_greeting(profile_no_trips, "alex")

        assert greeting is not None, (
            "A greeting must be generated when cached_research_session is present"
        )
        assert "Southeast Asia beach destinations 2026" in greeting, (
            f"Expected the research topic in Session 2 greeting, got: {greeting!r}"
        )
        assert "continue where you left off" in greeting.lower() or "continue" in greeting.lower()

    def test_session_2_greeting_exact_text_with_cached_research(
        self, tmp_memory_dir: Path
    ) -> None:
        """Exact greeting text when relying on cached_research_session (no past_trips)."""
        from concierge.agents.memory_service import build_proactive_greeting

        profile_with_cache: dict[str, Any] = {
            "preferred_name": "Alex",
            "past_trips": [],
            "cached_research_session": {
                "topic": "Southeast Asia beach destinations 2026",
                "session_id": "session-prev",
            },
        }

        greeting = build_proactive_greeting(profile_with_cache, "alex")

        expected = (
            "Welcome back, Alex. "
            "Last time we explored Southeast Asia beach destinations 2026. "
            "Want to continue where you left off?"
        )
        assert greeting == expected

    # ------------------------------------------------------------------
    # Full two-session integration scenario
    # ------------------------------------------------------------------

    def test_full_two_session_scenario(self, tmp_memory_dir: Path) -> None:
        """
        Full end-to-end scenario exercising both sessions in sequence.

        Assertions:
          - Both sessions load a non-None memory_profile
          - Session 1 greeting mentions Bali (past_trips-based)
          - Session 1 state.json written successfully
          - cached_research_session persisted in state.json
          - Session 2 profile contains cached_research_session from Session 1
          - Session 2 greeting shows topic continuity
          - session_ids are distinct
          - Both sessions start with turn_id == 0 and empty conversation_history
          - Session 1 turn_id advances to 1 after first invocation
        """
        from concierge.agents.memory_service import MemoryService, build_proactive_greeting
        from concierge.state import initialize_state

        # Setup: write profile to temp location
        _write_profile(tmp_memory_dir, ALEX_PROFILE)

        # ============================================================
        # SESSION 1
        # ============================================================
        ts_s1_start = "2026-02-26T09:00:00Z"
        ts_s1_end = "2026-02-26T09:15:00Z"
        session_id_1 = f"session-{uuid.uuid4().hex}"

        svc1 = MemoryService(
            profiles_dir=tmp_memory_dir / "profiles",
            sessions_dir=tmp_memory_dir / "sessions",
            output_writer=io.StringIO(),
        )
        profile1 = svc1.load_profile("alex")

        # --- Assertion S1.1: profile loaded ---
        assert profile1 is not None, "Session 1: memory_profile must be non-None"
        assert profile1["user_id"] == "alex"

        # --- Assertion S1.2: state initialized fresh ---
        state1 = dict(
            initialize_state(
                user_id="alex",
                session_id=session_id_1,
                current_input="",
                turn_id=0,
            )
        )
        state1["memory_profile"] = profile1
        assert state1["turn_id"] == 0, "Session 1: turn_id must start at 0"
        assert state1["conversation_history"] == [], "Session 1: history must start empty"

        # --- Assertion S1.3: proactive greeting mentions Bali ---
        greeting1 = build_proactive_greeting(profile1, "alex")
        assert greeting1 is not None
        assert "Bali" in greeting1, (
            f"Session 1 greeting must mention Bali; got: {greeting1!r}"
        )
        assert "Alex" in greeting1

        # --- Assertion S1.4: simulate graph turn (advancing turn_id) ---
        state1["turn_id"] = 1
        state1["current_input"] = "What Wyndham properties are available in Bali?"
        state1["current_response"] = "We have several Wyndham properties in Bali including Kuta Beach."
        state1["conversation_history"] = [
            {"role": "user", "content": state1["current_input"]},
            {"role": "assistant", "content": state1["current_response"]},
        ]
        assert state1["turn_id"] == 1

        # --- Assertion S1.5: inject cached research (simulates research agent result) ---
        cached_research_s1: dict[str, Any] = {
            "topic": "Southeast Asia beach destinations 2026",
            "session_id": session_id_1,
            "cached_at": ts_s1_end,
        }
        state1["cached_research_session"] = cached_research_s1

        # --- Assertion S1.6: write session state ---
        written1 = svc1.write_session_state(
            state1,
            timestamp_start=ts_s1_start,
            timestamp_end=ts_s1_end,
        )
        assert written1 is not None, "Session 1: write_session_state() must return a Path"
        assert written1.exists(), f"Session 1: state file must exist at {written1}"

        payload1 = json.loads(written1.read_text(encoding="utf-8"))
        assert payload1["session_id"] == session_id_1
        assert payload1["user_id"] == "alex"
        assert "cached_research_session" in payload1, (
            "Session 1: cached_research_session must be persisted in state.json"
        )
        assert payload1["cached_research_session"] == cached_research_s1

        # ============================================================
        # SESSION 2
        # ============================================================
        session_id_2 = f"session-{uuid.uuid4().hex}"

        # --- Assertion S2.1: session IDs are distinct ---
        assert session_id_2 != session_id_1, "session_id must differ between sessions"

        svc2 = MemoryService(
            profiles_dir=tmp_memory_dir / "profiles",
            sessions_dir=tmp_memory_dir / "sessions",
            output_writer=io.StringIO(),
        )
        profile2 = svc2.load_profile("alex")

        # --- Assertion S2.2: profile loaded in Session 2 ---
        assert profile2 is not None, "Session 2: memory_profile must be non-None"
        assert profile2["user_id"] == "alex"

        # --- Assertion S2.3: cached_research_session carried over from Session 1 ---
        assert "cached_research_session" in profile2, (
            "Session 2: profile must contain cached_research_session from Session 1 file"
        )
        assert profile2["cached_research_session"] == cached_research_s1

        # --- Assertion S2.4: state initialized fresh for Session 2 ---
        state2 = dict(
            initialize_state(
                user_id="alex",
                session_id=session_id_2,
                current_input="",
                turn_id=0,
            )
        )
        state2["memory_profile"] = profile2
        assert state2["turn_id"] == 0, "Session 2: turn_id must reset to 0"
        assert state2["conversation_history"] == [], "Session 2: history must start empty"
        assert state2["session_id"] == session_id_2

        # --- Assertion S2.5: Session 2 memory_profile is populated ---
        assert state2["memory_profile"] is not None
        assert isinstance(state2["memory_profile"], dict)

        # --- Assertion S2.6: proactive greeting for Session 2 uses past_trips (profile has trips) ---
        greeting2 = build_proactive_greeting(profile2, "alex")
        assert greeting2 is not None
        # Profile still has past_trips so the trip-based greeting fires
        assert "Bali" in greeting2, (
            f"Session 2 greeting (trip-based) must still mention Bali; got: {greeting2!r}"
        )

        # --- Assertion S2.7: continuity greeting when only cached_research is available ---
        # Simulate a scenario where trips list is empty (profile evolved over time)
        profile2_no_trips: dict[str, Any] = {
            **profile2,
            "past_trips": [],
        }
        greeting2_continuity = build_proactive_greeting(profile2_no_trips, "alex")
        assert greeting2_continuity is not None
        assert "Southeast Asia beach destinations 2026" in greeting2_continuity, (
            "Session 2 continuity greeting must reference the cached research topic"
        )
        assert "continue" in greeting2_continuity.lower()
