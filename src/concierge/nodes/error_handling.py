from __future__ import annotations


TIMEOUT_ERROR_MESSAGE = "LLM connection timeout. Please try again."
INVALID_API_KEY_ERROR_MESSAGE = "ANTHROPIC_API_KEY not set or invalid"
GENERIC_ERROR_MESSAGE = "Unable to complete this request right now. Please try again."
HUMAN_HANDOFF_SUFFIX = "A human travel specialist can help - please wait"


def build_node_error_message(exc: Exception) -> str:
    message = str(exc).strip()
    lowered = message.lower()
    error_type = type(exc).__name__.lower()

    if (
        "anthropic_api_key" in lowered
        or "api key" in lowered
        or "authentication" in lowered
        or "unauthorized" in lowered
        or "invalid key" in lowered
    ):
        return INVALID_API_KEY_ERROR_MESSAGE

    if "timeout" in lowered or "timed out" in lowered or "timeout" in error_type:
        return TIMEOUT_ERROR_MESSAGE

    if message:
        return GENERIC_ERROR_MESSAGE
    return GENERIC_ERROR_MESSAGE


def build_node_error_update(exc: Exception) -> dict[str, object]:
    return {
        "error": build_node_error_message(exc),
        "human_handoff": True,
    }
