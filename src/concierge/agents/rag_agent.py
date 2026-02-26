from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import yaml

from concierge.agents.mock_knowledge_base import MockKnowledgeBase
from concierge.state import ConciergeState
from concierge.trace import trace


NO_RAG_MATCH_MESSAGE = "No matching internal KB destinations found."
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "best",
    "for",
    "in",
    "is",
    "me",
    "of",
    "on",
    "the",
    "to",
    "what",
}


def query_mock_knowledge_base(query: str) -> list[dict[str, Any]]:
    return MockKnowledgeBase().query(query)


class RAGAgent:
    def __init__(
        self,
        policy_path: Path | None = None,
        prompt_dir: Path | None = None,
    ) -> None:
        repo_root = Path(__file__).resolve().parents[3]
        self._policy_path = policy_path or repo_root / "prompts" / "rag_agent" / "policy.yaml"
        self._prompt_dir = prompt_dir or repo_root / "prompts" / "rag_agent"
        self._rag_model, self._rag_max_tokens, prompt_version = self._load_policy()
        self._rag_prompt = self._load_prompt(prompt_version)
        self._llm_ranking_enabled = os.environ.get("RAG_AGENT_LLM_RANKING") == "1"

    @property
    def rag_model(self) -> str:
        if os.environ.get("FAST_MODE") == "1":
            return "claude-haiku-4-5"
        return self._rag_model

    def run(self, state: ConciergeState) -> ConciergeState:
        current_query = str(state.get("current_input") or "")
        results = _apply_query_constraints(current_query, query_mock_knowledge_base(current_query))
        ranked_results = self._rank_results_with_llm(current_query, results)
        trace("rag_agent", event="retrieval_complete", results_count=len(ranked_results))

        if ranked_results:
            return {"rag_results": ranked_results, "current_response": None}
        return {"rag_results": [], "current_response": NO_RAG_MATCH_MESSAGE}

    def _load_policy(self) -> tuple[str, int, str]:
        with self._policy_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        if not isinstance(data, dict):
            raise ValueError("rag agent policy must contain a YAML mapping")

        model = str(data.get("model") or "").strip()
        if not model:
            raise ValueError("rag agent model must be non-empty")

        max_tokens_raw = data.get("max_tokens")
        if not isinstance(max_tokens_raw, int) or max_tokens_raw <= 0:
            raise ValueError("rag agent max_tokens must be a positive integer")

        prompt_version = str(data.get("prompt_version") or "").strip()
        if not prompt_version:
            raise ValueError("rag agent prompt_version must be non-empty")

        return model, max_tokens_raw, prompt_version

    def _load_prompt(self, prompt_version: str) -> str:
        prompt_path = self._prompt_dir / f"{prompt_version}.yaml"
        with prompt_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        if not isinstance(data, dict):
            raise ValueError("rag agent prompt must contain a YAML mapping")

        role = str(data.get("role") or "").strip()
        context = str(data.get("context") or "").strip()
        output_format = str(data.get("output_format") or "").strip()
        constraints = data.get("constraints") or []
        constraints_text = "\n".join(f"- {item}" for item in constraints)
        return (
            f"{role}\n\n"
            f"Context:\n{context}\n\n"
            f"Constraints:\n{constraints_text}\n\n"
            f"Output format:\n{output_format}"
        ).strip()

    def _rank_results_with_llm(
        self,
        query: str,
        results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not self._llm_ranking_enabled or not results:
            return results
        if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
            return results

        try:
            import anthropic
        except Exception:
            return results

        ranking_prompt = (
            f"Query:\n{query}\n\n"
            f"Results:\n{json.dumps(results, ensure_ascii=False)}\n\n"
            "Return a JSON array of zero-based indexes in best-first order."
        )

        try:
            response = anthropic.Anthropic().messages.create(
                model=self.rag_model,
                max_tokens=self._rag_max_tokens,
                system=self._rag_prompt,
                messages=[{"role": "user", "content": ranking_prompt}],
            )
        except Exception:
            trace("rag_agent", event="ranking_skipped", reason="llm_api_error")
            return results
        ranking_text = self._extract_llm_text(response)
        return self._apply_ranking(ranking_text, results)

    def _extract_llm_text(self, response: object) -> str:
        content = getattr(response, "content", None)
        if not isinstance(content, list):
            return ""
        parts: list[str] = []
        for item in content:
            text = getattr(item, "text", None)
            if isinstance(text, str):
                parts.append(text)
        return "".join(parts)

    def _apply_ranking(self, ranking_text: str, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        try:
            parsed = json.loads(ranking_text)
        except Exception:
            return results
        if not isinstance(parsed, list):
            return results

        ranked: list[dict[str, Any]] = []
        used_indexes: set[int] = set()
        for item in parsed:
            if not isinstance(item, int):
                continue
            if item < 0 or item >= len(results) or item in used_indexes:
                continue
            used_indexes.add(item)
            ranked.append(results[item])

        if not ranked:
            return results

        for index, entry in enumerate(results):
            if index not in used_indexes:
                ranked.append(entry)
        return ranked


def _apply_query_constraints(
    query: str,
    entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    lowered_query = query.lower()
    filtered = list(entries)
    query_tokens = {
        token
        for token in re.findall(r"[a-z0-9]+", lowered_query)
        if token not in _STOPWORDS
    }

    if query_tokens:
        filtered = [
            entry
            for entry in filtered
            if _entry_matches_query_tokens(entry, query_tokens)
        ]

    if "southeast asia" in lowered_query:
        filtered = [
            entry
            for entry in filtered
            if str(entry.get("region") or "").strip().lower() == "southeast asia"
        ]

    if "beach" in lowered_query:
        filtered = [
            entry
            for entry in filtered
            if _entry_has_keyword(entry, "beach")
        ]

    if "city" in lowered_query:
        filtered = [
            entry
            for entry in filtered
            if _entry_has_keyword(entry, "city")
        ]

    unique: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for entry in filtered:
        entry_id = str(entry.get("id") or "")
        if entry_id and entry_id in seen_ids:
            continue
        if entry_id:
            seen_ids.add(entry_id)
        unique.append(entry)
    return unique


def _entry_has_keyword(entry: dict[str, Any], keyword: str) -> bool:
    target = keyword.lower()
    amenities = entry.get("amenities")
    if isinstance(amenities, list):
        amenity_values = {str(item).strip().lower() for item in amenities}
        if target in amenity_values:
            return True

    description = str(entry.get("description") or "").lower()
    return target in description


def _entry_matches_query_tokens(entry: dict[str, Any], query_tokens: set[str]) -> bool:
    searchable_values = [
        str(entry.get("name") or ""),
        str(entry.get("region") or ""),
        str(entry.get("description") or ""),
    ]
    amenities = entry.get("amenities")
    if isinstance(amenities, list):
        searchable_values.extend(str(item) for item in amenities)

    entry_tokens: set[str] = set()
    for value in searchable_values:
        entry_tokens.update(re.findall(r"[a-z0-9]+", value.lower()))

    return any(token in entry_tokens for token in query_tokens)
