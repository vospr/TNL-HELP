from __future__ import annotations

import importlib
import os

import pytest


def _load_main_module():
    return importlib.import_module("main")


def _policy_payload(model_name: str) -> dict[str, object]:
    return {
        "agent_name": "dispatcher",
        "model": model_name,
        "prompt_version": "v1",
        "max_tokens": 128,
        "confidence_threshold": 0.8,
        "max_clarifications": 2,
        "allowed_tools": [],
        "prompt_sections": ["role", "context", "constraints", "output_format", "examples"],
    }


def test_main_sets_fast_mode_env_before_runtime_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app_main = _load_main_module()
    monkeypatch.setattr(app_main.sys, "version_info", (3, 11, 9))
    monkeypatch.delenv("FAST_MODE", raising=False)

    observed_fast_mode: dict[str, str | None] = {"value": None}

    class _FakeGraph:
        def invoke(self, state: dict[str, object]) -> dict[str, object]:
            return state

    def _runtime_loader():
        observed_fast_mode["value"] = os.environ.get("FAST_MODE")
        return _FakeGraph(), lambda **kwargs: dict(kwargs)

    monkeypatch.setattr(app_main, "_load_runtime_dependencies", _runtime_loader)

    code = app_main.main(
        ["--user", "alex", "--fast-mode"],
        input_reader=lambda _prompt: "exit",
    )

    assert code == 0
    assert observed_fast_mode["value"] == "1"


def test_fast_mode_banner_prints_once(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    app_main = _load_main_module()
    monkeypatch.setattr(app_main.sys, "version_info", (3, 11, 9))

    class _FakeGraph:
        def invoke(self, state: dict[str, object]) -> dict[str, object]:
            return state

    monkeypatch.setattr(
        app_main,
        "_load_runtime_dependencies",
        lambda: (_FakeGraph(), lambda **kwargs: dict(kwargs)),
    )

    code = app_main.main(
        ["--user", "alex", "--fast-mode"],
        input_reader=lambda _prompt: "exit",
    )

    assert code == 0
    stdout = capsys.readouterr().out
    banner = "[FAST MODE] Using claude-haiku-4-5 across all agents - not the demo path"
    assert stdout.count(banner) == 1


def test_agent_policy_model_returns_configured_value_when_fast_mode_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import validate_config

    monkeypatch.delenv("FAST_MODE", raising=False)
    policy = validate_config.AgentPolicy.model_validate(
        _policy_payload(model_name="claude-opus-4-6-20260101")
    )

    assert policy.model == "claude-opus-4-6-20260101"


def test_agent_policy_model_returns_haiku_when_fast_mode_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import validate_config

    monkeypatch.setenv("FAST_MODE", "1")
    policy = validate_config.AgentPolicy.model_validate(
        _policy_payload(model_name="claude-opus-4-6-20260101")
    )

    assert policy.model == "claude-haiku-4-5"


def test_model_allowlist_is_skipped_when_fast_mode_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import validate_config

    monkeypatch.setenv("FAST_MODE", "1")

    def _should_not_be_called(*_args, **_kwargs):
        raise AssertionError("_load_policy_map must not run in FAST_MODE")

    monkeypatch.setattr(validate_config, "_load_policy_map", _should_not_be_called)

    assert validate_config.check_model_allowlist() == []
