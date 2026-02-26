from __future__ import annotations

from pathlib import Path

import yaml

import validate_config
from concierge.agents.dispatcher import DispatcherAgent


def test_dispatcher_loads_v2_prompt_when_policy_version_changes(tmp_path: Path) -> None:
    routing_rules_path = tmp_path / "routing_rules.yaml"
    routing_rules_path.write_text(
        yaml.safe_dump(
            {
                "escalation_threshold": 0.72,
                "rules": [
                    {
                        "pattern": r"\b(book)\b",
                        "intent": "booking_intent",
                        "route": "booking_stub",
                        "score": 0.95,
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    prompt_dir = tmp_path / "prompts" / "dispatcher"
    prompt_dir.mkdir(parents=True, exist_ok=True)

    policy_path = prompt_dir / "policy.yaml"
    policy_path.write_text(
        yaml.safe_dump(
            {
                "agent_name": "dispatcher",
                "model": "claude-opus-4-6",
                "prompt_version": "v2",
                "max_tokens": 128,
                "confidence_threshold": 0.75,
                "max_clarifications": 3,
                "allowed_tools": [],
                "prompt_sections": [
                    "role",
                    "context",
                    "constraints",
                    "output_format",
                    "examples",
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    (prompt_dir / "v1.yaml").write_text(
        yaml.safe_dump(
            {
                "role": "DISPATCHER-V1",
                "context": "v1",
                "constraints": ["v1"],
                "output_format": "json",
                "examples": [],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (prompt_dir / "v2.yaml").write_text(
        yaml.safe_dump(
            {
                "role": "DISPATCHER-V2-MARKER",
                "context": "v2",
                "constraints": ["v2"],
                "output_format": "json",
                "examples": [],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    agent = DispatcherAgent(
        routing_rules_path=routing_rules_path,
        dispatcher_policy_path=policy_path,
        dispatcher_prompt_dir=prompt_dir,
    )

    assert "DISPATCHER-V2-MARKER" in agent._dispatcher_prompt
    assert "DISPATCHER-V1" not in agent._dispatcher_prompt


def test_validate_config_reports_missing_prompt_version_file_actionable_error(tmp_path: Path) -> None:
    policies = {
        "dispatcher": validate_config.AgentPolicy.model_validate(
            {
                "agent_name": "dispatcher",
                "model": "claude-opus-4-6",
                "prompt_version": "v2",
                "max_tokens": 128,
                "confidence_threshold": 0.75,
                "max_clarifications": 3,
                "allowed_tools": [],
                "prompt_sections": [
                    "role",
                    "context",
                    "constraints",
                    "output_format",
                    "examples",
                ],
            }
        )
    }

    dispatcher_dir = tmp_path / "prompts" / "dispatcher"
    dispatcher_dir.mkdir(parents=True, exist_ok=True)
    (dispatcher_dir / "v1.yaml").write_text("role: r\ncontext: c\nconstraints: []\noutput_format: o\nexamples: []\n", encoding="utf-8")

    errors = validate_config.check_prompt_version_files_exist(
        policies=policies,
        repo_root=tmp_path,
    )

    assert errors == ["prompts/dispatcher/v2.yaml not found"]
