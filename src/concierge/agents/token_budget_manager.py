from __future__ import annotations

from typing_extensions import TypedDict


class Message(TypedDict, total=False):
    role: str
    content: str


class TokenBudgetManager:
    _ACTIVATION_THRESHOLD_MESSAGES = 6000  # message count approximation (stub — replace with tiktoken in production)
    _RECENT_TURNS_TO_KEEP = 8

    @property
    def activation_threshold_tokens(self) -> int:
        # TODO: make configurable
        return self._ACTIVATION_THRESHOLD_MESSAGES

    def check_and_summarize(self, history: list[Message]) -> list[Message]:
        # Stub implementation. In production, summarize when token count exceeds 6000 (leaving 2000 for output). Activation threshold: 80% of model context window.
        threshold = self._ACTIVATION_THRESHOLD_MESSAGES
        if len(history) < threshold:
            return history

        summary_message = self._build_system_summary(history)
        recent_history = history[-self._RECENT_TURNS_TO_KEEP :]
        return [summary_message, *recent_history]

    def _build_system_summary(self, history: list[Message]) -> Message:
        user_contents = [
            str(message.get("content", "")).strip()
            for message in history
            if str(message.get("role", "")).strip() == "user"
            and str(message.get("content", "")).strip()
        ]
        highlights = "; ".join(user_contents[:3]) if user_contents else "Conversation compressed."
        return {
            "role": "system_summary",
            "content": f"[SUMMARY] Key facts preserved: {highlights}",
        }
