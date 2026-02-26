from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = REPO_ROOT / "prompts"
AGENT_PROMPT_DIRS = [
    "dispatcher",
    "rag_agent",
    "research_agent",
    "response_synthesis",
    "guardrail",
    "followup",
]


def test_required_prompt_directories_exist() -> None:
    for name in AGENT_PROMPT_DIRS:
        path = PROMPTS_DIR / name
        assert path.exists() and path.is_dir(), f"Missing prompt directory: {path}"


def test_policy_and_v1_exist_for_each_prompt_directory() -> None:
    for name in AGENT_PROMPT_DIRS:
        policy_file = PROMPTS_DIR / name / "policy.yaml"
        v1_file = PROMPTS_DIR / name / "v1.yaml"
        assert policy_file.exists(), f"Missing {policy_file}"
        assert v1_file.exists(), f"Missing {v1_file}"
