from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = REPO_ROOT / "prompts"
AGENTS = [
    "dispatcher",
    "rag_agent",
    "research_agent",
    "response_synthesis",
    "guardrail",
    "followup",
]
REQUIRED_SECTIONS = {"role", "context", "constraints", "output_format", "examples"}
ALLOWED_MODEL_PREFIXES = (
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-haiku-4-5",
)


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    assert isinstance(data, dict), f"{path} must contain a YAML mapping/object"
    return data


def _model_is_allowed(model: str) -> bool:
    return model.startswith(ALLOWED_MODEL_PREFIXES)


def test_policy_yaml_fields_and_types() -> None:
    for agent in AGENTS:
        path = PROMPTS_DIR / agent / "policy.yaml"
        data = _load_yaml(path)

        assert data.get("agent_name") == agent, f"{path} agent_name must be '{agent}'"
        model = data.get("model")
        assert isinstance(model, str) and _model_is_allowed(model), (
            f"{path} has invalid model: {model!r}"
        )
        assert data.get("prompt_version") == "v1", f"{path} prompt_version must be 'v1'"

        max_tokens = data.get("max_tokens")
        assert isinstance(max_tokens, int) and max_tokens > 0, (
            f"{path} max_tokens must be positive integer"
        )

        confidence = data.get("confidence_threshold")
        assert isinstance(confidence, (int, float)) and 0.0 <= float(confidence) <= 1.0, (
            f"{path} confidence_threshold must be numeric in [0.0, 1.0]"
        )

        max_clarifications = data.get("max_clarifications")
        assert isinstance(max_clarifications, int) and max_clarifications >= 0, (
            f"{path} max_clarifications must be non-negative integer"
        )

        allowed_tools = data.get("allowed_tools")
        assert isinstance(allowed_tools, list), f"{path} allowed_tools must be a list"

        prompt_sections = data.get("prompt_sections")
        assert isinstance(prompt_sections, list), f"{path} prompt_sections must be a list"
        assert set(prompt_sections) == REQUIRED_SECTIONS, (
            f"{path} prompt_sections must match required sections {sorted(REQUIRED_SECTIONS)}"
        )


def test_dispatcher_policy_max_tokens_exact_128() -> None:
    path = PROMPTS_DIR / "dispatcher" / "policy.yaml"
    data = _load_yaml(path)
    assert data.get("max_tokens") == 128, (
        "prompts/dispatcher/policy.yaml max_tokens must be exactly 128"
    )


def test_model_validation_allows_supported_version_updates() -> None:
    assert _model_is_allowed("claude-sonnet-4-6-20260201"), (
        "Model validation should allow supported version updates without exact-string pinning"
    )
