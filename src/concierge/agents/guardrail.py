from __future__ import annotations

import re
from pathlib import Path

import yaml

from concierge.state import ConciergeState
from concierge.trace import trace


CLARIFICATION_QUESTION = (
    "Of course - are you looking to research a destination, check on a booking, or something else?"
)
HUMAN_HANDOFF_MESSAGE = (
    "I'm having trouble understanding your request. "
    "A human travel specialist can assist - please wait or contact support."
)
OUT_OF_DOMAIN_DEFLECTION_TEMPLATE = (
    "I specialize in travel planning and concierge services. "
    "For weather, I'd suggest a weather service - but I can tell you the best time of year to visit {location} if that helps."
)
OUT_OF_DOMAIN_CONFIDENCE_MAX = 0.4


class GuardrailAgent:
    def __init__(
        self,
        dispatcher_policy_path: Path | None = None,
        guardrail_policy_path: Path | None = None,
    ) -> None:
        repo_root = Path(__file__).resolve().parents[3]
        self._dispatcher_policy_path = (
            dispatcher_policy_path
            or repo_root / "prompts" / "dispatcher" / "policy.yaml"
        )
        self._guardrail_policy_path = (
            guardrail_policy_path
            or repo_root / "prompts" / "guardrail" / "policy.yaml"
        )
        self._dispatcher_confidence_threshold = self._load_dispatcher_confidence_threshold()
        self._max_clarifications = self._load_guardrail_max_clarifications()

    def run(self, state: ConciergeState) -> ConciergeState:
        if bool(state.get("human_handoff")):
            return {}

        confidence_raw = state.get("confidence")
        if not isinstance(confidence_raw, (int, float)):
            return {
                "guardrail_passed": True,
                "clarification_needed": False,
                "clarification_question": None,
                "human_handoff": False,
                "max_clarifications": self._max_clarifications,
            }

        confidence = float(confidence_raw)
        if self._should_deflect_out_of_domain(state, confidence):
            normalized_query = self._normalize_query_for_trace(str(state.get("current_input") or ""))
            trace(
                "guardrail",
                event="out_of_domain",
                confidence=round(confidence, 2),
                query=normalized_query,
            )
            return {
                "guardrail_passed": False,
                "clarification_needed": False,
                "clarification_question": None,
                "human_handoff": False,
                "current_response": self._build_out_of_domain_deflection(
                    str(state.get("current_input") or "")
                ),
                "max_clarifications": self._max_clarifications,
            }

        if confidence >= self._dispatcher_confidence_threshold:
            return {
                "guardrail_passed": True,
                "clarification_needed": False,
                "clarification_question": None,
                "human_handoff": False,
                "max_clarifications": self._max_clarifications,
            }

        clarification_count = self._count_prior_clarifications(state.get("conversation_history"))
        if clarification_count >= self._max_clarifications:
            trace(
                "guardrail",
                event="human_handoff",
                clarification_count=clarification_count,
                session_id=state.get("session_id"),
            )
            return {
                "guardrail_passed": False,
                "clarification_needed": False,
                "clarification_question": None,
                "human_handoff": True,
                "current_response": HUMAN_HANDOFF_MESSAGE,
                "max_clarifications": self._max_clarifications,
            }

        trace(
            "guardrail",
            event="clarification_fired",
            confidence=round(confidence, 2),
            threshold=self._dispatcher_confidence_threshold,
        )
        return {
            "guardrail_passed": False,
            "clarification_needed": True,
            "clarification_question": CLARIFICATION_QUESTION,
            "human_handoff": False,
            "max_clarifications": self._max_clarifications,
        }

    def _load_dispatcher_confidence_threshold(self) -> float:
        with self._dispatcher_policy_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        if not isinstance(data, dict):
            raise ValueError("dispatcher policy must contain a YAML mapping")

        threshold_raw = data.get("confidence_threshold")
        if not isinstance(threshold_raw, (int, float)):
            raise ValueError("dispatcher confidence_threshold must be numeric")
        threshold = float(threshold_raw)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("dispatcher confidence_threshold must be in [0.0, 1.0]")
        return threshold

    def _load_guardrail_max_clarifications(self) -> int:
        with self._guardrail_policy_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        if not isinstance(data, dict):
            raise ValueError("guardrail policy must contain a YAML mapping")

        max_clarifications_raw = data.get("max_clarifications")
        if not isinstance(max_clarifications_raw, int) or max_clarifications_raw < 0:
            raise ValueError("guardrail max_clarifications must be a non-negative integer")
        return max_clarifications_raw

    def _should_deflect_out_of_domain(self, state: ConciergeState, confidence: float) -> bool:
        intent = str(state.get("intent") or "").strip().lower()
        if intent == "out_of_domain":
            return True

        if confidence >= OUT_OF_DOMAIN_CONFIDENCE_MAX:
            return False

        query = str(state.get("current_input") or "")
        return self._looks_weather_query(query)

    def _looks_weather_query(self, query: str) -> bool:
        normalized = query.strip().lower()
        if not normalized:
            return False
        return "weather" in normalized

    def _normalize_query_for_trace(self, query: str) -> str:
        lowered = query.strip().lower()
        if not lowered:
            return ""
        lowered = re.sub(r"[^a-z0-9\s]+", " ", lowered)
        lowered = re.sub(r"\s+", " ", lowered).strip()
        if lowered.startswith("what s the "):
            lowered = lowered[len("what s the ") :]
        if lowered.startswith("what is the "):
            lowered = lowered[len("what is the ") :]
        if lowered.startswith("in "):
            lowered = lowered[3:]
        lowered = lowered.replace("weather in ", "weather ")
        return lowered

    def _build_out_of_domain_deflection(self, query: str) -> str:
        location = self._extract_location(query) or "that destination"
        return OUT_OF_DOMAIN_DEFLECTION_TEMPLATE.format(location=location)

    def _extract_location(self, query: str) -> str | None:
        cleaned = query.strip()
        if not cleaned:
            return None

        match = re.search(r"\bin\s+([A-Za-z][A-Za-z\s-]{1,40})\??$", cleaned)
        if not match:
            return None

        location = match.group(1).strip(" ?.,!")
        if not location:
            return None
        return location

    def _count_prior_clarifications(self, history: object) -> int:
        if not isinstance(history, list):
            return 0

        count = 0
        for item in history:
            if not isinstance(item, dict):
                continue
            content = str(item.get("content") or "").strip()
            if content == CLARIFICATION_QUESTION:
                count += 1
        return count
