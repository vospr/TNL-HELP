from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import yaml

from concierge.agents.rag_agent import RAGAgent
from concierge.state import initialize_state


def test_rag_agent_llm_ranking_enforces_policy_max_tokens(monkeypatch) -> None:
    from concierge.agents import rag_agent as rag_agent_module

    monkeypatch.setenv("RAG_AGENT_LLM_RANKING", "1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("FAST_MODE", raising=False)
    monkeypatch.setattr(
        rag_agent_module,
        "query_mock_knowledge_base",
        lambda query: [
            {"id": "dest-bali", "name": "Bali", "region": "Southeast Asia", "amenities": ["beach"]},
            {"id": "dest-phuket", "name": "Phuket", "region": "Southeast Asia", "amenities": ["beach"]},
        ],
    )

    captured_kwargs: dict[str, object] = {}

    class _FakeMessages:
        def create(self, **kwargs: object) -> object:
            captured_kwargs.update(kwargs)
            return SimpleNamespace(content=[SimpleNamespace(text="[1, 0]")])

    class _FakeAnthropicClient:
        def __init__(self) -> None:
            self.messages = _FakeMessages()

    monkeypatch.setitem(
        sys.modules,
        "anthropic",
        SimpleNamespace(Anthropic=lambda: _FakeAnthropicClient()),
    )

    update = RAGAgent().run(
        initialize_state("alex", "session-5-6-rag-ranking", "best beach destinations", turn_id=1)
    )

    assert captured_kwargs["max_tokens"] == 512
    assert captured_kwargs["model"] == "claude-haiku-4-5-20251001"
    assert [item["id"] for item in update["rag_results"]] == ["dest-phuket", "dest-bali"]


def test_validate_config_flags_missing_max_tokens_before_runtime(tmp_path: Path) -> None:
    import validate_config

    _seed_minimal_prompts_tree(tmp_path)

    policy_path = tmp_path / "prompts" / "research_agent" / "policy.yaml"
    with policy_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    del data["max_tokens"]
    with policy_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)

    errors = validate_config.check_policy_yaml_schema(repo_root=tmp_path)

    assert errors
    assert any("max_tokens" in line for line in errors)


def test_validate_config_flags_invalid_max_tokens_for_dispatcher_cap(tmp_path: Path) -> None:
    import validate_config

    _seed_minimal_prompts_tree(tmp_path)

    policy_path = tmp_path / "prompts" / "dispatcher" / "policy.yaml"
    with policy_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    data["max_tokens"] = 256
    with policy_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)

    policies = validate_config._load_policy_map(repo_root=tmp_path)
    errors = validate_config.check_dispatcher_max_tokens_exact_128(
        policies,
        repo_root=tmp_path,
    )

    assert errors == ["Dispatcher max_tokens must be exactly 128"]


def _seed_minimal_prompts_tree(repo_root: Path) -> None:
    prompts_dir = repo_root / "prompts"
    required_sections = ["role", "context", "constraints", "output_format", "examples"]
    agent_tokens = {
        "dispatcher": 128,
        "rag_agent": 512,
        "research_agent": 1024,
        "response_synthesis": 1024,
        "guardrail": 256,
        "followup": 256,
    }

    for agent, max_tokens in agent_tokens.items():
        agent_dir = prompts_dir / agent
        agent_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "agent_name": agent,
            "model": "claude-sonnet-4-6",
            "prompt_version": "v1",
            "max_tokens": max_tokens,
            "confidence_threshold": 0.7,
            "max_clarifications": 3,
            "allowed_tools": [],
            "prompt_sections": required_sections,
        }
        with (agent_dir / "policy.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False)
