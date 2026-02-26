from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml

from concierge.state import ConciergeState
from concierge.trace import trace


WEB_SEARCH_UNAVAILABLE_LABEL = "[WEB SEARCH UNAVAILABLE — serving from internal KB only]"


def search_duckduckgo(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    from duckduckgo_search import DDGS

    with DDGS() as ddgs:
        return list(ddgs.text(query, max_results=max_results))


class ResearchAgent:
    def __init__(
        self,
        policy_path: Path | None = None,
        prompt_dir: Path | None = None,
    ) -> None:
        repo_root = Path(__file__).resolve().parents[3]
        self._policy_path = policy_path or repo_root / "prompts" / "research_agent" / "policy.yaml"
        self._prompt_dir = prompt_dir or repo_root / "prompts" / "research_agent"
        (
            self._research_model,
            self._research_max_tokens,
            prompt_version,
        ) = self._load_policy()
        self._research_prompt = self._load_prompt(prompt_version)
        self._llm_ranking_enabled = os.environ.get("RESEARCH_AGENT_LLM_RANKING") == "1"

    @property
    def research_model(self) -> str:
        if os.environ.get("FAST_MODE") == "1":
            return "claude-haiku-4-5"
        return self._research_model

    def run(self, state: ConciergeState) -> ConciergeState:
        current_query = str(state.get("current_input") or "").strip()
        scoped_query = self._build_scoped_query(
            current_query,
            state.get("conversation_history"),
        )
        try:
            raw_results = search_duckduckgo(scoped_query, max_results=5)
        except Exception as exc:
            reason = self._degradation_reason(exc)
            trace("research_agent", event="search_unavailable", reason=reason)
            return {
                "research_results": [],
                "degradation_label": WEB_SEARCH_UNAVAILABLE_LABEL,
                "current_response": WEB_SEARCH_UNAVAILABLE_LABEL,
            }
        parsed_results = self._parse_results(raw_results)
        ranked_results = self._rank_results_with_llm(scoped_query, parsed_results)

        trace("research_agent", event="search_complete", results_count=len(ranked_results))
        return {
            "research_results": ranked_results,
            "degradation_label": None,
            "current_response": None,
        }

    def _load_policy(self) -> tuple[str, int, str]:
        with self._policy_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        if not isinstance(data, dict):
            raise ValueError("research agent policy must contain a YAML mapping")

        model = str(data.get("model") or "").strip()
        if not model:
            raise ValueError("research agent model must be non-empty")

        max_tokens_raw = data.get("max_tokens")
        if not isinstance(max_tokens_raw, int) or max_tokens_raw <= 0:
            raise ValueError("research agent max_tokens must be a positive integer")

        prompt_version = str(data.get("prompt_version") or "").strip()
        if not prompt_version:
            raise ValueError("research agent prompt_version must be non-empty")

        return model, max_tokens_raw, prompt_version

    def _load_prompt(self, prompt_version: str) -> str:
        prompt_path = self._prompt_dir / f"{prompt_version}.yaml"
        with prompt_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        if not isinstance(data, dict):
            raise ValueError("research agent prompt must contain a YAML mapping")

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

    def _build_scoped_query(self, current_query: str, history: object) -> str:
        normalized_query = current_query.strip()
        if not isinstance(history, list) or not history:
            return normalized_query

        recent_turns = history[-3:]
        context_lines: list[str] = []
        for turn in recent_turns:
            if not isinstance(turn, dict):
                continue
            role = str(turn.get("role") or "").strip() or "user"
            content = str(turn.get("content") or "").strip()
            if not content:
                continue
            context_lines.append(f"{role}: {content}")

        if not context_lines:
            return normalized_query
        return f"{normalized_query}\n\nRecent context:\n" + "\n".join(context_lines)

    def _parse_results(self, raw_results: object) -> list[dict[str, str]]:
        if not isinstance(raw_results, list):
            return []

        parsed: list[dict[str, str]] = []
        for item in raw_results:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            link = str(item.get("href") or item.get("link") or item.get("url") or "").strip()
            snippet = str(
                item.get("body") or item.get("snippet") or item.get("description") or ""
            ).strip()
            if not title or not link or not snippet:
                continue
            parsed.append({"title": title, "link": link, "snippet": snippet})
        return parsed

    def _rank_results_with_llm(
        self,
        scoped_query: str,
        results: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        if not self._llm_ranking_enabled or not results:
            return results
        if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
            return results

        try:
            import anthropic
        except Exception:
            return results

        ranking_prompt = (
            f"Query:\n{scoped_query}\n\n"
            f"Results:\n{json.dumps(results, ensure_ascii=False)}\n\n"
            "Return a JSON array of zero-based indexes in best-first order."
        )
        try:
            response = anthropic.Anthropic().messages.create(
                model=self.research_model,
                max_tokens=self._research_max_tokens,
                system=self._research_prompt,
                messages=[{"role": "user", "content": ranking_prompt}],
            )
        except Exception:
            trace("research_agent", event="ranking_skipped", reason="llm_api_error")
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

    def _apply_ranking(self, ranking_text: str, results: list[dict[str, str]]) -> list[dict[str, str]]:
        try:
            parsed = json.loads(ranking_text)
        except Exception:
            return results
        if not isinstance(parsed, list):
            return results

        ranked: list[dict[str, str]] = []
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

    def _degradation_reason(self, exc: Exception) -> str:
        message = str(exc).lower()
        error_type = type(exc).__name__.lower()
        if "timeout" in message or "timed out" in message or "timeout" in error_type:
            return "timeout"
        if "rate limit" in message or "429" in message:
            return "rate_limit"
        return "unavailable"
