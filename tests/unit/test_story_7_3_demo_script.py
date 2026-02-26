from __future__ import annotations

from pathlib import Path

import pytest

from concierge.agents.dispatcher import DispatcherAgent


REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_SCRIPT_PATH = REPO_ROOT / "demo_script.md"


def test_demo_script_contains_five_queries_in_required_order() -> None:
    text = DEMO_SCRIPT_PATH.read_text(encoding="utf-8")
    queries = [
        '"I\'m thinking Southeast Asia but not sure where"',
        '"What about the Wyndham property in Phuket?"',
        '"Can you help me?"',
        '"What\'s the weather?"',
        '"Book me a flight"',
    ]

    positions = [text.index(query) for query in queries]
    assert positions == sorted(positions)


def test_demo_script_includes_required_expected_outcomes() -> None:
    text = DEMO_SCRIPT_PATH.read_text(encoding="utf-8")

    assert "Expected confidence: `0.84` (expected range: `0.79-0.89`)" in text
    assert "Expected confidence: `0.91`" in text
    assert "Expected confidence: `< 0.75`" in text
    assert "Expected confidence: `0.32` (expected range: `0.27-0.37`)" in text
    assert "Expected confidence: `0.95`" in text

    assert "Expected routing destination: `research` (Research Agent)" in text
    assert "Expected routing destination: `rag` (RAG Agent)" in text
    assert "Expected routing destination: `fallback` with Guardrail clarification" in text
    assert "Expected routing destination: `fallback`" in text
    assert "Expected routing destination: `booking_stub` (Booking Agent stub)" in text

    assert (
        "Of course - are you looking to research a destination, check on a booking, or something else?"
        in text
    )
    assert "I specialize in travel planning and concierge services..." in text
    assert "Booking is not available in this version." in text


def test_demo_script_includes_required_stage2_variance_note() -> None:
    text = DEMO_SCRIPT_PATH.read_text(encoding="utf-8")
    assert "Queries 2 and 5 route via Stage 1 (deterministic)." in text
    assert "Queries 1, 3, and 4 depend on Stage 2 escalation (non-deterministic)." in text
    assert (
        '"Stage 2 routing may vary ±0.05 confidence; expected range provided. '
        'Pre-validate actual confidence scores before demo."'
        in text
    )


def test_stage1_prevalidation_snapshot_matches_current_routing_rules() -> None:
    agent = DispatcherAgent()

    query2 = "What about the Wyndham property in Phuket?"
    query5 = "Book me a flight"
    query4 = "What's the weather?"

    intent2, confidence2, route2 = agent._evaluate_stage1(query2)
    intent5, confidence5, route5 = agent._evaluate_stage1(query5)
    intent4, confidence4, route4 = agent._evaluate_stage1(query4)

    assert (intent2, route2) == ("property_lookup", "rag")
    assert confidence2 == pytest.approx(0.91)

    assert (intent5, route5) == ("booking_intent", "booking_stub")
    assert confidence5 == pytest.approx(0.95)

    assert intent4 == "out_of_domain"
    assert confidence4 == pytest.approx(0.30)
    assert route4 is None
