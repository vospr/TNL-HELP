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
REQUIRED_KEYS = ["role", "context", "constraints", "output_format", "examples"]
AGENT_SPECIFIC_MARKERS = {
    "dispatcher": ["dispatcher", "route"],
    "rag_agent": ["rag", "retrieval"],
    "research_agent": ["research", "web"],
    "response_synthesis": ["synthesis", "response"],
    "guardrail": ["guardrail", "clarification"],
    "followup": ["follow-up", "suggestion"],
}


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    assert isinstance(data, dict), f"{path} must contain a YAML mapping/object"
    return data


def test_v1_contains_all_required_non_empty_sections() -> None:
    for agent in AGENTS:
        path = PROMPTS_DIR / agent / "v1.yaml"
        data = _load_yaml(path)
        for key in REQUIRED_KEYS:
            assert key in data, f"{path} missing section '{key}'"
            value = data[key]
            if isinstance(value, str):
                assert value.strip(), f"{path} section '{key}' must be non-empty"
            elif isinstance(value, list):
                assert value, f"{path} section '{key}' list must be non-empty"
            elif isinstance(value, dict):
                assert value, f"{path} section '{key}' object must be non-empty"
            else:
                assert value is not None, f"{path} section '{key}' must be non-empty"


def test_v1_content_is_agent_specific() -> None:
    for agent in AGENTS:
        path = PROMPTS_DIR / agent / "v1.yaml"
        data = _load_yaml(path)
        combined = " ".join(str(data[key]).lower() for key in REQUIRED_KEYS if key in data)
        for marker in AGENT_SPECIFIC_MARKERS[agent]:
            assert marker in combined, (
                f"{path} must include agent-specific marker '{marker}' to avoid generic prompt content"
            )
