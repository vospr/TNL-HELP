from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from concierge.state import ConciergeState, TurnResetUpdate
from concierge.trace import trace
from concierge.agents.token_budget_manager import TokenBudgetManager


LOGGER = logging.getLogger(__name__)


def reset_turn_state(state: ConciergeState) -> TurnResetUpdate:
    del state
    return TurnResetUpdate(
        intent=None,
        confidence=None,
        route=None,
        rag_results=None,
        research_results=None,
        source_attribution=[],
        degradation_label=None,
        guardrail_passed=True,
        proactive_suggestion=None,
        clarification_needed=False,
        clarification_question=None,
        human_handoff=False,
        error=None,
    )


class DispatcherAgent:
    _INTENT_TO_ROUTE = {
        "property_lookup": "rag",
        "destination_research": "research",
        "trend_research": "research",
        "booking_intent": "booking_stub",
        "booking_request": "booking_stub",
        "out_of_domain": "fallback",
    }

    @dataclass(frozen=True)
    class Stage1Rule:
        intent: str
        route: str
        confidence: float
        pattern: re.Pattern[str]

    def __init__(
        self,
        routing_rules_path: Path | None = None,
        dispatcher_policy_path: Path | None = None,
        dispatcher_prompt_dir: Path | None = None,
    ) -> None:
        repo_root = Path(__file__).resolve().parents[3]
        self._routing_rules_path = routing_rules_path or repo_root / "config" / "routing_rules.yaml"
        self._dispatcher_policy_path = (
            dispatcher_policy_path
            or repo_root / "prompts" / "dispatcher" / "policy.yaml"
        )
        self._dispatcher_prompt_dir = dispatcher_prompt_dir or repo_root / "prompts" / "dispatcher"

        self._escalation_threshold, self._routing_rules = self._load_stage1_rules()
        (
            self._dispatcher_model,
            self._dispatcher_max_tokens,
            self._dispatcher_confidence_threshold,
            prompt_version,
        ) = self._load_dispatcher_policy()
        self._dispatcher_prompt = self._load_dispatcher_prompt(prompt_version)

    @property
    def escalation_threshold(self) -> float:
        return self._escalation_threshold

    @property
    def routing_rules(self) -> tuple[Stage1Rule, ...]:
        return self._routing_rules

    @property
    def dispatcher_model(self) -> str:
        if os.environ.get("FAST_MODE") == "1":
            return "claude-haiku-4-5"
        return self._dispatcher_model

    def _load_dispatcher_policy(self) -> tuple[str, int, float, str]:
        with self._dispatcher_policy_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)

        if not isinstance(data, dict):
            raise ValueError("dispatcher policy must contain a YAML mapping")

        model = str(data.get("model") or "").strip()
        if not model:
            raise ValueError("dispatcher policy model must be non-empty")

        max_tokens_raw = data.get("max_tokens")
        if not isinstance(max_tokens_raw, int) or max_tokens_raw <= 0:
            raise ValueError("dispatcher policy max_tokens must be a positive integer")

        confidence_raw = data.get("confidence_threshold")
        if not isinstance(confidence_raw, (int, float)):
            raise ValueError("dispatcher confidence_threshold must be numeric")
        confidence_threshold = float(confidence_raw)
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("dispatcher confidence_threshold must be in [0.0, 1.0]")

        prompt_version = str(data.get("prompt_version") or "").strip()
        if not prompt_version:
            raise ValueError("dispatcher prompt_version must be non-empty")

        return model, max_tokens_raw, confidence_threshold, prompt_version

    def _load_dispatcher_prompt(self, prompt_version: str) -> str:
        prompt_path = self._dispatcher_prompt_dir / f"{prompt_version}.yaml"
        with prompt_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)

        if not isinstance(data, dict):
            raise ValueError("dispatcher prompt file must contain a YAML mapping")

        role = str(data.get("role") or "").strip()
        context = str(data.get("context") or "").strip()
        output_format = str(data.get("output_format") or "").strip()
        constraints = data.get("constraints") or []
        examples = data.get("examples") or []

        constraints_text = "\n".join(f"- {item}" for item in constraints)
        examples_text = "\n".join(
            f"- input: {example.get('input')} output: {example.get('output')}"
            for example in examples
            if isinstance(example, dict)
        )

        return (
            f"{role}\n\n"
            f"Context:\n{context}\n\n"
            f"Constraints:\n{constraints_text}\n\n"
            f"Output format:\n{output_format}\n\n"
            f"Examples:\n{examples_text}"
        ).strip()

    def _load_stage1_rules(self) -> tuple[float, tuple[Stage1Rule, ...]]:
        with self._routing_rules_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)

        if not isinstance(data, dict):
            raise ValueError("routing_rules.yaml must contain a YAML mapping")

        threshold_raw = data.get("escalation_threshold")
        if not isinstance(threshold_raw, (int, float)):
            raise ValueError("routing_rules.yaml escalation_threshold must be numeric")
        threshold = float(threshold_raw)

        rules_raw = data.get("rules")
        if not isinstance(rules_raw, list) or not rules_raw:
            raise ValueError("routing_rules.yaml rules must be a non-empty list")

        rules: list[DispatcherAgent.Stage1Rule] = []
        for index, item in enumerate(rules_raw, start=1):
            if not isinstance(item, dict):
                raise ValueError(f"routing_rules.yaml rules[{index}] must be a mapping")

            intent = str(item.get("intent") or "").strip()
            route = str(item.get("route") or "").strip()
            pattern = str(item.get("pattern") or "").strip()
            confidence_raw = item.get("score")

            if not intent:
                raise ValueError(f"routing_rules.yaml rules[{index}].intent must be non-empty")
            if not route:
                raise ValueError(f"routing_rules.yaml rules[{index}].route must be non-empty")
            if not pattern:
                raise ValueError(f"routing_rules.yaml rules[{index}].pattern must be non-empty")
            if not isinstance(confidence_raw, (int, float)):
                raise ValueError(f"routing_rules.yaml rules[{index}].score must be numeric")

            rules.append(
                DispatcherAgent.Stage1Rule(
                    intent=intent,
                    route=route,
                    confidence=float(confidence_raw),
                    pattern=re.compile(pattern, flags=re.IGNORECASE),
                )
            )

        return threshold, tuple(rules)

    def _evaluate_stage1(self, current_input: str) -> tuple[str | None, float, str | None]:
        text = current_input.strip()
        if not text:
            return None, 0.0, None

        matched_rules = [
            rule
            for rule in self._routing_rules
            if rule.pattern.search(text) is not None
        ]
        if not matched_rules:
            return None, 0.0, None

        best_rule = max(matched_rules, key=lambda rule: rule.confidence)
        if best_rule.confidence < self._escalation_threshold:
            return best_rule.intent, best_rule.confidence, None
        return best_rule.intent, best_rule.confidence, best_rule.route

    def _extract_llm_text(self, response: Any) -> str:
        content = getattr(response, "content", None)
        if not isinstance(content, list):
            return ""
        parts: list[str] = []
        for block in content:
            text = getattr(block, "text", None)
            if isinstance(text, str) and text.strip():
                parts.append(text)
        return "\n".join(parts).strip()

    def _parse_stage2_response(self, raw_text: str) -> tuple[str, float] | None:
        text = raw_text.strip()
        if not text:
            return None

        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        payload = match.group(0) if match else text

        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            return None

        if not isinstance(data, dict):
            return None
        intent = str(data.get("intent") or "").strip()
        confidence_raw = data.get("confidence")
        if not intent or not isinstance(confidence_raw, (int, float)):
            return None
        confidence = float(confidence_raw)
        if not 0.0 <= confidence <= 1.0:
            return None
        return intent, confidence

    def _evaluate_stage2(self, current_input: str) -> tuple[str | None, float | None, str | None]:
        if not current_input.strip():
            return None, None, None
        if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
            return None, None, None

        try:
            import anthropic
        except Exception:
            return None, None, None

        client = anthropic.Anthropic()
        response = client.messages.create(
            model=self.dispatcher_model,
            max_tokens=self._dispatcher_max_tokens,
            system=self._dispatcher_prompt,
            messages=[{"role": "user", "content": current_input}],
        )
        parsed = self._parse_stage2_response(self._extract_llm_text(response))
        if parsed is None:
            return None, None, None

        intent, confidence = parsed
        mapped_route = self._INTENT_TO_ROUTE.get(intent, "fallback")
        route = (
            mapped_route
            if confidence >= self._dispatcher_confidence_threshold
            else "fallback"
        )
        return intent, confidence, route

    def _apply_token_budget(self, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        manager = TokenBudgetManager()
        threshold = manager.activation_threshold_tokens
        compressed_history = manager.check_and_summarize(history)
        trace(
            "token_budget",
            event="check_performed",
            message_count=len(history),
            threshold=threshold,
        )
        if compressed_history != history:
            LOGGER.info(
                "[TOKEN_BUDGET] Context approaching limit. "
                "In production, oldest turns would be summarized here."
            )
        return compressed_history

    def run(self, state: ConciergeState) -> ConciergeState:
        existing_route = state.get("route")
        if isinstance(existing_route, str) and existing_route.strip():
            return {}

        current_input = str(state.get("current_input") or "")
        history = list(state.get("conversation_history") or [])
        history.append({"role": "user", "content": current_input})
        update: ConciergeState = {
            **reset_turn_state(state),
            "conversation_history": history,
            "current_response": None,
        }

        stage1_intent, stage1_confidence, stage1_route = self._evaluate_stage1(current_input)
        if stage1_route is not None:
            update["intent"] = stage1_intent
            update["confidence"] = stage1_confidence
            update["route"] = stage1_route
            trace(
                "dispatcher",
                intent=stage1_intent,
                confidence=stage1_confidence,
                route=stage1_route,
                stage="pre_filter",
            )
            return update

        update["conversation_history"] = self._apply_token_budget(history)

        stage2_intent, stage2_confidence, stage2_route = self._evaluate_stage2(current_input)
        if stage2_intent is not None and stage2_confidence is not None and stage2_route is not None:
            update["intent"] = stage2_intent
            update["confidence"] = stage2_confidence
            update["route"] = stage2_route
            trace(
                "dispatcher",
                intent=stage2_intent,
                confidence=stage2_confidence,
                route=stage2_route,
                stage="llm_escalation",
            )
            return update

        update["intent"] = stage1_intent
        update["confidence"] = stage1_confidence
        update["route"] = stage1_route
        trace(
            "dispatcher",
            intent=stage1_intent,
            confidence=stage1_confidence,
            route=stage1_route,
            stage="llm_escalation",
        )
        return update
