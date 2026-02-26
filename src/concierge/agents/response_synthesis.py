from __future__ import annotations

from typing import Any

from concierge.nodes.error_handling import HUMAN_HANDOFF_SUFFIX
from concierge.state import ConciergeState
from concierge.trace import trace


class ResponseSynthesisAgent:
    def run(self, state: ConciergeState) -> ConciergeState:
        clarification_needed = bool(state.get("clarification_needed"))
        if clarification_needed:
            clarification_question = str(state.get("clarification_question") or "").strip()
            if clarification_question:
                return {"current_response": clarification_question}

        error_message = str(state.get("error") or "").strip()
        human_handoff = bool(state.get("human_handoff"))
        if not error_message and not human_handoff:
            return self._run_success_path(state)

        existing_response = str(state.get("current_response") or "").strip()
        response = (
            error_message
            or existing_response
            or "Unable to complete this request right now. Please try again."
        )
        if human_handoff and (error_message or not existing_response):
            if not response.endswith(HUMAN_HANDOFF_SUFFIX):
                response = f"{response} {HUMAN_HANDOFF_SUFFIX}".strip()

        return {"current_response": response}

    def _run_success_path(self, state: ConciergeState) -> ConciergeState:
        booking_message, booking_integration_point, booking_required_env_vars = (
            self._extract_booking_payload(state)
        )
        if booking_message and str(state.get("route") or "").strip() == "booking_stub":
            booking_response = booking_message
            if booking_integration_point:
                booking_response = (
                    f"{booking_response} Integration point: {booking_integration_point}."
                ).strip()
            if booking_required_env_vars:
                booking_response = (
                    f"{booking_response} Required env vars: {', '.join(booking_required_env_vars)}."
                ).strip()
            trace("response_synthesis", event="synthesis_complete", sources_cited=0)
            return {"current_response": booking_response, "source_attribution": []}

        rag_results = self._normalize_rag_results(state.get("rag_results"))
        research_results = self._normalize_research_results(state.get("research_results"))
        filtered_history = self._filter_history(state.get("conversation_history"))

        if not rag_results and not research_results:
            existing = str(state.get("current_response") or "").strip()
            if existing:
                return {"current_response": existing}
            return {}

        response = self._build_response_text(rag_results, research_results, filtered_history)
        degradation_label = str(state.get("degradation_label") or "").strip()
        if degradation_label and rag_results and not research_results:
            response = f"{degradation_label} {response}".strip()

        source_attribution = self._build_source_attribution(rag_results, research_results)
        trace(
            "response_synthesis",
            event="synthesis_complete",
            sources_cited=int(bool(rag_results)) + int(bool(research_results)),
        )
        return {
            "current_response": response,
            "source_attribution": source_attribution,
        }

    def _extract_booking_payload(
        self,
        state: ConciergeState,
    ) -> tuple[str, str, list[str]]:
        current_response = state.get("current_response")
        if isinstance(current_response, str):
            return current_response.strip(), "", []

        if isinstance(current_response, dict):
            message = str(current_response.get("message") or "").strip()
            integration_point = str(current_response.get("integration_point") or "").strip()
            required_env_vars = current_response.get("required_env_vars")
            if not isinstance(required_env_vars, list):
                return message, integration_point, []
            return (
                message,
                integration_point,
                [str(item).strip() for item in required_env_vars if str(item).strip()],
            )

        message_attr = getattr(current_response, "message", None)
        integration_point_attr = getattr(current_response, "integration_point", None)
        required_env_vars_attr = getattr(current_response, "required_env_vars", None)

        message = str(message_attr or "").strip()
        integration_point = str(integration_point_attr or "").strip()
        if not isinstance(required_env_vars_attr, list):
            return message, integration_point, []

        return (
            message,
            integration_point,
            [str(item).strip() for item in required_env_vars_attr if str(item).strip()],
        )

    def _normalize_rag_results(self, value: object) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []
        return [item for item in value if isinstance(item, dict)]

    def _normalize_research_results(self, value: object) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []
        return [item for item in value if isinstance(item, dict)]

    def _filter_history(self, history: object) -> list[dict[str, str]]:
        if not isinstance(history, list):
            return []

        filtered: list[dict[str, str]] = []
        for item in history:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "").strip()
            content = str(item.get("content") or "").strip()
            if not role or not content:
                continue
            if role == "system_summary":
                continue
            filtered.append({"role": role, "content": content})
        return filtered

    def _build_response_text(
        self,
        rag_results: list[dict[str, Any]],
        research_results: list[dict[str, Any]],
        filtered_history: list[dict[str, str]],
    ) -> str:
        context_hint = self._build_context_hint(filtered_history)
        if rag_results and research_results:
            rag_names = self._join_display_values(rag_results, primary_key="name", fallback_key="id")
            trend_title = str(research_results[0].get("title") or "current travel trends").strip()
            response = (
                "Based on our current offerings [RAG] and latest travel trends [Web], "
                f"the top destinations are {rag_names}. {trend_title} supports this direction."
            )
        elif rag_results:
            rag_names = self._join_display_values(rag_results, primary_key="name", fallback_key="id")
            response = f"Based on our internal knowledge base [RAG], top destinations include {rag_names}."
        else:
            web_titles = self._join_display_values(
                research_results,
                primary_key="title",
                fallback_key="link",
            )
            response = f"Based on latest travel trends [Web], consider {web_titles}."

        if context_hint:
            response = f"{response} You noted: {context_hint}."
        return response.strip()

    def _build_context_hint(self, filtered_history: list[dict[str, str]]) -> str:
        for item in reversed(filtered_history):
            if item.get("role") != "user":
                continue
            return str(item.get("content") or "").strip().rstrip(".!?")
        return ""

    def _build_source_attribution(
        self,
        rag_results: list[dict[str, Any]],
        research_results: list[dict[str, Any]],
    ) -> list[str]:
        attributions: list[str] = []
        for item in rag_results[:3]:
            name = str(item.get("name") or item.get("id") or "internal KB entry").strip()
            attributions.append(f"[RAG] {name} - internal KB entry")

        for item in research_results[:3]:
            title = str(item.get("title") or "Latest travel trend").strip()
            link = str(item.get("link") or "").strip()
            if link:
                attributions.append(f"[Web] {title} - {link}")
            else:
                attributions.append(f"[Web] {title}")
        return attributions

    def _join_display_values(
        self,
        items: list[dict[str, Any]],
        primary_key: str,
        fallback_key: str,
    ) -> str:
        values: list[str] = []
        for item in items[:3]:
            primary = str(item.get(primary_key) or "").strip()
            fallback = str(item.get(fallback_key) or "").strip()
            value = primary or fallback
            if value:
                values.append(value)

        if not values:
            return "relevant options"
        if len(values) == 1:
            return values[0]
        if len(values) == 2:
            return f"{values[0]} and {values[1]}"
        return f"{', '.join(values[:-1])}, and {values[-1]}"
