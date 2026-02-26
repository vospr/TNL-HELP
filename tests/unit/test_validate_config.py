from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_check_order_contract() -> None:
    import validate_config

    assert validate_config.CHECK_NAMES == (
        "check_python_version",
        "check_anthropic_api_key_present",
        "check_policy_yaml_schema",
        "check_prompt_version_files_exist",
        "check_model_allowlist",
        "check_dispatcher_max_tokens_exact_128",
        "check_guardrail_threshold_ordering",
        "check_routing_rules_schema",
        "check_langgraph_version_major_1",
        "check_alex_profile_json",
        "check_anthropic_api_probe",
    )


def test_python_version_failure_is_human_readable(monkeypatch: pytest.MonkeyPatch) -> None:
    import validate_config

    monkeypatch.setattr(validate_config.sys, "version_info", (3, 10, 9))

    errors = validate_config.check_python_version()
    assert errors
    assert "Python 3.11+ required" in errors[0]


def test_missing_api_key_mentions_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    import validate_config

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    errors = validate_config.check_anthropic_api_key_present()
    assert errors
    assert "ANTHROPIC_API_KEY" in errors[0]


def test_dispatcher_max_tokens_exact_128_enforced() -> None:
    import validate_config

    policies = validate_config._load_policy_map(REPO_ROOT)
    errors = validate_config.check_dispatcher_max_tokens_exact_128(policies)
    assert not errors


def test_threshold_ordering_guardrail_less_than_dispatcher() -> None:
    import validate_config

    policies = validate_config._load_policy_map(REPO_ROOT)
    errors = validate_config.check_guardrail_threshold_ordering(policies)
    assert not errors


def test_run_checks_lists_multiple_failures_before_exit(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import validate_config

    monkeypatch.setenv("SKIP_API_PROBE", "1")

    def _ok(*_args, **_kwargs) -> list[str]:
        return []

    monkeypatch.setattr(validate_config, "check_python_version", _ok)
    monkeypatch.setattr(validate_config, "check_anthropic_api_key_present", _ok)
    monkeypatch.setattr(validate_config, "check_policy_yaml_schema", _ok)
    monkeypatch.setattr(validate_config, "check_prompt_version_files_exist", _ok)
    monkeypatch.setattr(validate_config, "check_model_allowlist", _ok)
    monkeypatch.setattr(validate_config, "check_routing_rules_schema", _ok)
    monkeypatch.setattr(validate_config, "check_langgraph_version_major_1", _ok)
    monkeypatch.setattr(validate_config, "check_alex_profile_json", _ok)
    monkeypatch.setattr(validate_config, "check_anthropic_api_probe", _ok)

    monkeypatch.setattr(
        validate_config,
        "check_dispatcher_max_tokens_exact_128",
        lambda *_args, **_kwargs: ["Dispatcher max_tokens must be exactly 128"],
    )
    monkeypatch.setattr(
        validate_config,
        "check_guardrail_threshold_ordering",
        lambda *_args, **_kwargs: [
            "guardrail.confidence_threshold must be lower than dispatcher.confidence_threshold"
        ],
    )

    with pytest.raises(SystemExit) as exc:
        validate_config.run_checks(repo_root=REPO_ROOT)

    assert exc.value.code == 1
    output = capsys.readouterr().err
    assert "Dispatcher max_tokens must be exactly 128" in output
    assert "guardrail.confidence_threshold must be lower than dispatcher.confidence_threshold" in output
    assert "[CONFIG ERROR]" in output


def test_standalone_validate_config_prints_config_ok() -> None:
    pytest.importorskip("langgraph")

    env = os.environ.copy()
    env["ANTHROPIC_API_KEY"] = "test-key"
    env["SKIP_API_PROBE"] = "1"
    env.pop("FAST_MODE", None)

    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "validate_config.py")],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "[CONFIG OK]" in result.stdout
