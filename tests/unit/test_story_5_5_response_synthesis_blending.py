from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from concierge.agents.response_synthesis import ResponseSynthesisAgent
from concierge.state import initialize_state


class _BookingAgentResponse(BaseModel):
    status: str
    message: str
    integration_point: str
    required_env_vars: list[str]


def test_response_synthesis_blends_rag_and_web_with_inline_tags_and_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from concierge.agents import response_synthesis as module

    trace_calls: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(
        module,
        "trace",
        lambda node_name, outcome=None, **fields: trace_calls.append((node_name, dict(fields))),
    )

    state = initialize_state("alex", "session-5-5-blended", "best destinations", turn_id=1)
    state["rag_results"] = [
        {"id": "dest-bali", "name": "Bali"},
        {"id": "dest-phuket", "name": "Phuket"},
    ]
    state["research_results"] = [
        {
            "title": "Latest Bali travel trends from travel blogs",
            "link": "https://example.com/bali-trends",
            "snippet": "Bali demand remains strong.",
        }
    ]

    update = ResponseSynthesisAgent().run(state)

    assert "[RAG]" in update["current_response"]
    assert "[Web]" in update["current_response"]
    assert any(item.startswith("[RAG]") for item in update["source_attribution"])
    assert any(item.startswith("[Web]") for item in update["source_attribution"])
    assert trace_calls == [
        ("response_synthesis", {"event": "synthesis_complete", "sources_cited": 2})
    ]


def test_response_synthesis_filters_system_summary_before_synthesis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from concierge.agents import response_synthesis as module

    captured_history: list[dict[str, str]] = []

    def _fake_build(
        self: ResponseSynthesisAgent,
        rag_results: list[dict[str, Any]],
        research_results: list[dict[str, Any]],
        filtered_history: list[dict[str, str]],
    ) -> str:
        del self, rag_results, research_results
        captured_history.extend(filtered_history)
        return "synthetic response [RAG] [Web]"

    monkeypatch.setattr(module.ResponseSynthesisAgent, "_build_response_text", _fake_build)
    monkeypatch.setattr(module, "trace", lambda *args, **kwargs: None)

    state = initialize_state("alex", "session-5-5-filter", "query", turn_id=2)
    state["conversation_history"] = [
        {"role": "system_summary", "content": "MUST_NOT_APPEAR"},
        {"role": "user", "content": "Prefer beach destinations"},
        {"role": "assistant", "content": "Noted"},
    ]
    state["rag_results"] = [{"id": "dest-bali", "name": "Bali"}]
    state["research_results"] = [
        {"title": "Trend", "link": "https://example.com/trend", "snippet": "x"}
    ]

    update = module.ResponseSynthesisAgent().run(state)

    assert update["current_response"] == "synthetic response [RAG] [Web]"
    assert all(item.get("role") != "system_summary" for item in captured_history)
    assert len(captured_history) == 2


def test_response_synthesis_uses_rag_only_when_web_results_missing() -> None:
    state = initialize_state("alex", "session-5-5-rag-only", "query", turn_id=3)
    state["rag_results"] = [{"id": "dest-bangkok", "name": "Bangkok"}]
    state["research_results"] = []

    update = ResponseSynthesisAgent().run(state)

    assert "[RAG]" in update["current_response"]
    assert "[Web]" not in update["current_response"]
    assert update["source_attribution"]
    assert all(item.startswith("[RAG]") for item in update["source_attribution"])


def test_response_synthesis_formats_booking_model_message_without_crashing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from concierge.agents import response_synthesis as module

    monkeypatch.setattr(module, "trace", lambda *args, **kwargs: None)

    state = initialize_state("alex", "session-5-5-booking", "book a flight", turn_id=4)
    state["route"] = "booking_stub"
    state["current_response"] = _BookingAgentResponse(
        status="unavailable",
        message="Booking is not available in this version. A human travel specialist can assist.",
        integration_point="Replace with BedrockBookingAPI(...)",
        required_env_vars=["BOOKING_API_KEY", "BOOKING_REGION"],
    )

    update = module.ResponseSynthesisAgent().run(state)

    assert update["current_response"].startswith("Booking is not available")


def test_response_synthesis_kb_only_with_degradation_label_still_returns_full_response() -> None:
    state = initialize_state("alex", "session-5-5-degraded", "query", turn_id=5)
    state["rag_results"] = [{"id": "dest-bali", "name": "Bali"}]
    state["research_results"] = []
    state["degradation_label"] = "[WEB SEARCH UNAVAILABLE — serving from internal KB only]"

    update = ResponseSynthesisAgent().run(state)

    assert update["current_response"].startswith("[WEB SEARCH UNAVAILABLE")
    assert "[RAG]" in update["current_response"]

