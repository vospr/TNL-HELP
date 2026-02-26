from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_KB_PATH = REPO_ROOT / "agents" / "kb" / "knowledge_base.json"


class MockKnowledgeBase:
    # Production swap point: replace MockKnowledgeBase with BedrockKnowledgeBase(kb_id=os.getenv("BEDROCK_KB_ID"))
    def __init__(self, kb_path: Path | None = None) -> None:
        self._kb_path = kb_path or DEFAULT_KB_PATH

    def load_entries(self) -> list[dict[str, Any]]:
        payload = json.loads(self._kb_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("knowledge_base.json must contain a top-level array")
        return [entry for entry in payload if isinstance(entry, dict)]

    def query(self, query: str) -> list[dict[str, Any]]:
        normalized_query = str(query or "").strip().lower()
        if not normalized_query:
            return []

        keywords = {token for token in re.split(r"[^a-z0-9]+", normalized_query) if token}
        if not keywords:
            return []

        results: list[dict[str, Any]] = []
        for entry in self.load_entries():
            name = str(entry.get("name") or "")
            region = str(entry.get("region") or "")
            description = str(entry.get("description") or "")
            amenities = entry.get("amenities")
            amenity_text = (
                " ".join(str(item) for item in amenities)
                if isinstance(amenities, list)
                else ""
            )
            searchable = " ".join([name, region, description, amenity_text]).lower()

            if any(keyword in searchable for keyword in keywords):
                results.append(entry)
        return results
