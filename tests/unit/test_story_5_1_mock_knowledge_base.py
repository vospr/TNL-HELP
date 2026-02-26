from __future__ import annotations

import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
KB_PATH = REPO_ROOT / "agents" / "kb" / "knowledge_base.json"
SWAP_POINT_PATH = REPO_ROOT / "src" / "concierge" / "agents" / "mock_knowledge_base.py"
REQUIRED_FIELDS = {"id", "name", "region", "description", "amenities", "pricing"}
REQUIRED_DESTINATIONS = {"Bali", "Phuket", "Koh Samui", "Tokyo", "Bangkok"}
SWAP_POINT_COMMENT = (
    "# Production swap point: replace MockKnowledgeBase with "
    "BedrockKnowledgeBase(kb_id=os.getenv(\"BEDROCK_KB_ID\"))"
)


def _load_kb() -> list[dict[str, object]]:
    payload = json.loads(KB_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, list), "knowledge_base.json must contain a top-level array"
    return payload


def _sentence_count(value: str) -> int:
    parts = [part.strip() for part in re.split(r"[.!?]+", value) if part.strip()]
    return len(parts)


def test_kb_json_parses_with_required_destination_fields() -> None:
    entries = _load_kb()

    assert entries, "knowledge_base.json must not be empty"
    for entry in entries:
        assert REQUIRED_FIELDS.issubset(entry.keys())


def test_kb_contains_expected_destinations_and_substantive_descriptions() -> None:
    entries = _load_kb()
    names = {str(entry["name"]) for entry in entries}

    assert len(entries) >= 5
    assert REQUIRED_DESTINATIONS.issubset(names)
    for entry in entries:
        assert _sentence_count(str(entry["description"])) >= 3


def test_production_swap_point_comment_exists_with_exact_guidance() -> None:
    source = SWAP_POINT_PATH.read_text(encoding="utf-8")
    assert SWAP_POINT_COMMENT in source


def test_rag_query_keyword_matching_returns_expected_destinations() -> None:
    from concierge.agents.rag_agent import query_mock_knowledge_base

    beach_names = {str(item["name"]) for item in query_mock_knowledge_base("beach")}
    city_names = {str(item["name"]) for item in query_mock_knowledge_base("city")}

    assert {"Bali", "Phuket"}.issubset(beach_names)
    assert {"Tokyo", "Bangkok"}.issubset(city_names)


def test_kb_entries_are_self_documenting_with_consistent_structure() -> None:
    entries = _load_kb()
    template_keys = tuple(entries[0].keys())

    for entry in entries:
        assert tuple(entry.keys()) == template_keys
        assert isinstance(entry["amenities"], list)
        assert entry["amenities"]
        assert isinstance(entry["pricing"], dict)
