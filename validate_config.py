from __future__ import annotations

import ast
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


AGENTS = (
    "dispatcher",
    "rag_agent",
    "research_agent",
    "response_synthesis",
    "guardrail",
    "followup",
)
REQUIRED_PROMPT_SECTIONS = {"role", "context", "constraints", "output_format", "examples"}
ALLOWED_MODEL_PREFIXES = (
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-haiku-4-5",
)
CHECK_NAMES = (
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


class AgentPolicy(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    agent_name: str
    configured_model: str = Field(alias="model")
    prompt_version: str
    max_tokens: int
    confidence_threshold: float
    max_clarifications: int
    allowed_tools: list[str]
    prompt_sections: list[str]

    @property
    def model(self) -> str:
        if os.environ.get("FAST_MODE") == "1":
            return "claude-haiku-4-5"
        return self.configured_model

    @field_validator("max_tokens")
    @classmethod
    def validate_max_tokens(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("must be a positive integer")
        return value

    @field_validator("confidence_threshold")
    @classmethod
    def validate_confidence_threshold(cls, value: float) -> float:
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError("must be in [0.0, 1.0]")
        return float(value)

    @field_validator("max_clarifications")
    @classmethod
    def validate_max_clarifications(cls, value: int) -> int:
        if value < 0:
            raise ValueError("must be non-negative")
        return value

    @field_validator("prompt_sections")
    @classmethod
    def validate_prompt_sections(cls, value: list[str]) -> list[str]:
        if set(value) != REQUIRED_PROMPT_SECTIONS:
            raise ValueError(
                "must include exactly these sections: "
                + ", ".join(sorted(REQUIRED_PROMPT_SECTIONS))
            )
        return value


def _resolve_repo_root(repo_root: Path | str | None = None) -> Path:
    if repo_root is None:
        return Path(__file__).resolve().parent
    return Path(repo_root).resolve()


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping/object")
    return data


def _load_policy_map(repo_root: Path | str | None = None) -> dict[str, AgentPolicy]:
    root = _resolve_repo_root(repo_root)
    prompts_dir = root / "prompts"
    errors: list[str] = []
    policies: dict[str, AgentPolicy] = {}

    for agent in AGENTS:
        policy_path = prompts_dir / agent / "policy.yaml"
        if not policy_path.exists():
            errors.append(f"{policy_path} is missing")
            continue

        try:
            raw_policy = _load_yaml_mapping(policy_path)
            policy = AgentPolicy.model_validate(raw_policy)
        except yaml.YAMLError as exc:
            errors.append(f"{policy_path} is not valid YAML: {exc}")
            continue
        except ValidationError as exc:
            errors.append(
                f"{policy_path} failed AgentPolicy validation: "
                f"{exc.errors(include_url=False)}"
            )
            continue
        except ValueError as exc:
            errors.append(str(exc))
            continue

        if policy.agent_name != agent:
            errors.append(
                f"{policy_path} has agent_name={policy.agent_name!r}, expected {agent!r}"
            )
            continue

        policies[agent] = policy

    if errors:
        raise ValueError("\n".join(errors))
    return policies


def _model_is_allowed(model_name: str) -> bool:
    return model_name.startswith(ALLOWED_MODEL_PREFIXES)


def check_python_version(repo_root: Path | str | None = None) -> list[str]:
    del repo_root
    if tuple(sys.version_info[:2]) >= (3, 11):
        return []
    current = ".".join(map(str, sys.version_info[:3]))
    return [f"Python 3.11+ required. Current version: {current}. Install Python 3.11 or newer."]


def check_anthropic_api_key_present(repo_root: Path | str | None = None) -> list[str]:
    del repo_root
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if api_key:
        return []
    return [
        "ANTHROPIC_API_KEY is missing. Set ANTHROPIC_API_KEY in your environment or .env file."
    ]


def check_policy_yaml_schema(repo_root: Path | str | None = None) -> list[str]:
    try:
        _load_policy_map(repo_root)
        return []
    except ValueError as exc:
        return [line for line in str(exc).splitlines() if line.strip()]


def check_model_allowlist(repo_root: Path | str | None = None) -> list[str]:
    if os.environ.get("FAST_MODE") == "1":
        return []

    try:
        policies = _load_policy_map(repo_root)
    except ValueError:
        return []

    errors: list[str] = []
    for agent, policy in policies.items():
        if not _model_is_allowed(policy.model):
            errors.append(
                f"prompts/{agent}/policy.yaml has unsupported model {policy.model!r}. "
                f"Use one of families: {', '.join(ALLOWED_MODEL_PREFIXES)}."
            )
    return errors


def check_prompt_version_files_exist(
    policies: dict[str, AgentPolicy] | None = None,
    repo_root: Path | str | None = None,
) -> list[str]:
    root = _resolve_repo_root(repo_root)
    if policies is None:
        try:
            policies = _load_policy_map(root)
        except ValueError:
            return []

    errors: list[str] = []
    for agent, policy in policies.items():
        prompt_path = root / "prompts" / agent / f"{policy.prompt_version}.yaml"
        if not prompt_path.exists():
            errors.append(f"prompts/{agent}/{policy.prompt_version}.yaml not found")
    return errors


def check_dispatcher_max_tokens_exact_128(
    policies: dict[str, AgentPolicy] | None = None,
    repo_root: Path | str | None = None,
) -> list[str]:
    if policies is None:
        try:
            policies = _load_policy_map(repo_root)
        except ValueError:
            return []

    dispatcher_policy = policies.get("dispatcher")
    if dispatcher_policy is None:
        return ["prompts/dispatcher/policy.yaml missing or invalid; cannot validate max_tokens."]
    if dispatcher_policy.max_tokens != 128:
        return ["Dispatcher max_tokens must be exactly 128"]
    return []


def check_guardrail_threshold_ordering(
    policies: dict[str, AgentPolicy] | None = None,
    repo_root: Path | str | None = None,
) -> list[str]:
    if policies is None:
        try:
            policies = _load_policy_map(repo_root)
        except ValueError:
            return []

    guardrail = policies.get("guardrail")
    dispatcher = policies.get("dispatcher")
    if guardrail is None or dispatcher is None:
        return [
            "prompts/guardrail/policy.yaml and prompts/dispatcher/policy.yaml must be valid "
            "to compare confidence thresholds."
        ]

    if guardrail.confidence_threshold >= dispatcher.confidence_threshold:
        return [
            "guardrail.confidence_threshold must be lower than "
            "dispatcher.confidence_threshold"
        ]
    return []


def check_routing_rules_schema(repo_root: Path | str | None = None) -> list[str]:
    root = _resolve_repo_root(repo_root)
    routing_path = root / "config" / "routing_rules.yaml"
    if not routing_path.exists():
        return [f"{routing_path} is missing"]

    try:
        config = _load_yaml_mapping(routing_path)
    except (ValueError, yaml.YAMLError) as exc:
        return [f"{routing_path} could not be parsed: {exc}"]

    errors: list[str] = []

    threshold = config.get("escalation_threshold")
    if not isinstance(threshold, (int, float)):
        errors.append("config/routing_rules.yaml escalation_threshold must be numeric")
    elif not 0.0 <= float(threshold) <= 1.0:
        errors.append("config/routing_rules.yaml escalation_threshold must be in [0.0, 1.0]")

    rules = config.get("rules")
    if not isinstance(rules, list) or not rules:
        errors.append("config/routing_rules.yaml rules must be a non-empty list")
        return errors

    for idx, rule in enumerate(rules, start=1):
        if not isinstance(rule, dict):
            errors.append(f"config/routing_rules.yaml rules[{idx}] must be a mapping")
            continue

        for key in ("pattern", "intent", "route"):
            value = rule.get(key)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"config/routing_rules.yaml rules[{idx}].{key} must be non-empty")

        score = rule.get("score")
        if not isinstance(score, (int, float)) or not 0.0 <= float(score) <= 1.0:
            errors.append(f"config/routing_rules.yaml rules[{idx}].score must be in [0.0, 1.0]")

    return errors


def check_langgraph_version_major_1(repo_root: Path | str | None = None) -> list[str]:
    del repo_root
    try:
        import langgraph
    except Exception as exc:  # pragma: no cover - environment dependent
        return [f"langgraph import failed: {exc}. Install dependencies from requirements.txt."]

    version = getattr(langgraph, "__version__", "")
    if not version:
        try:
            import importlib.metadata
            version = importlib.metadata.version("langgraph")
        except Exception:
            pass
    if version.startswith("1."):
        return []
    return [f"langgraph major version must start with '1.', found {version!r}"]


def check_alex_profile_json(repo_root: Path | str | None = None) -> list[str]:
    root = _resolve_repo_root(repo_root)
    alex_path = root / "memory" / "profiles" / "alex.json"
    if not alex_path.exists():
        return [f"{alex_path} is missing"]

    try:
        with alex_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        return [f"{alex_path} is not valid JSON: {exc}"]

    if not isinstance(data, dict):
        return [f"{alex_path} must contain a JSON object"]
    return []


def check_anthropic_api_probe(repo_root: Path | str | None = None) -> list[str]:
    del repo_root
    if os.environ.get("SKIP_API_PROBE") == "1":
        return []
    if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
        return []

    try:
        import anthropic
    except Exception as exc:  # pragma: no cover - environment dependent
        return [f"anthropic SDK import failed: {exc}. Install dependencies from requirements.txt."]

    try:
        client = anthropic.Anthropic(timeout=5.0)
        client.models.list(limit=1)
    except Exception as exc:  # pragma: no cover - network/auth dependent
        error_type = type(exc).__name__.lower()
        message = str(exc)
        message_lower = message.lower()

        if "auth" in error_type or "authentication" in message_lower:
            return [
                "ANTHROPIC_API_KEY appears invalid or expired. "
                "Update ANTHROPIC_API_KEY with a valid active key."
            ]
        if "timeout" in error_type or "timed out" in message_lower:
            return ["API probe timed out - check network. Key format appears valid."]

        return [f"Anthropic API probe failed: {message}"]

    return []


def check_reset_turn_state_contract_non_blocking(
    repo_root: Path | str | None = None,
) -> list[str]:
    root = _resolve_repo_root(repo_root)
    state_path = root / "src" / "concierge" / "state.py"
    dispatcher_path = root / "src" / "concierge" / "agents" / "dispatcher.py"
    if not state_path.exists() or not dispatcher_path.exists():
        return []

    expected_fields = _typed_dict_fields_from_class(state_path, "TurnResetUpdate")
    if not expected_fields:
        return [
            "src/concierge/state.py exists but TurnResetUpdate contract was not found.",
        ]

    reset_fields = _reset_turn_state_fields(dispatcher_path)
    if reset_fields is None:
        return [
            "src/concierge/agents/dispatcher.py exists but reset_turn_state() was not found."
        ]

    missing = sorted(expected_fields - reset_fields)
    if missing:
        return [
            "reset_turn_state() is missing required TurnResetUpdate fields: "
            + ", ".join(missing)
        ]

    return []


def _typed_dict_fields_from_class(path: Path, class_name: str) -> set[str]:
    module = ast.parse(path.read_text(encoding="utf-8"))
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            fields: set[str] = set()
            for body_node in node.body:
                if isinstance(body_node, ast.AnnAssign) and isinstance(
                    body_node.target, ast.Name
                ):
                    fields.add(body_node.target.id)
            return fields
    return set()


def _reset_turn_state_fields(path: Path) -> set[str] | None:
    module = ast.parse(path.read_text(encoding="utf-8"))
    for node in module.body:
        if not isinstance(node, ast.FunctionDef) or node.name != "reset_turn_state":
            continue
        for child in ast.walk(node):
            if not isinstance(child, ast.Return) or child.value is None:
                continue
            if isinstance(child.value, ast.Call):
                return {kw.arg for kw in child.value.keywords if kw.arg is not None}
            if isinstance(child.value, ast.Dict):
                names: set[str] = set()
                for key in child.value.keys:
                    if isinstance(key, ast.Constant) and isinstance(key.value, str):
                        names.add(key.value)
                return names
        return set()
    return None


def run_checks(repo_root: Path | str | None = None) -> None:
    resolved_root = _resolve_repo_root(repo_root)
    failures: list[str] = []
    policy_cache: dict[str, AgentPolicy] | None = None

    for check_name in CHECK_NAMES:
        check_fn = globals()[check_name]

        if check_name in {
            "check_prompt_version_files_exist",
            "check_dispatcher_max_tokens_exact_128",
            "check_guardrail_threshold_ordering",
        }:
            if policy_cache is None:
                try:
                    policy_cache = _load_policy_map(resolved_root)
                except ValueError:
                    policy_cache = {}
            errors = check_fn(policy_cache, repo_root=resolved_root)
        else:
            errors = check_fn(repo_root=resolved_root)

        failures.extend(errors)

    failures.extend(check_reset_turn_state_contract_non_blocking(resolved_root))

    if failures:
        print("[CONFIG ERROR] One or more configuration checks failed:", file=sys.stderr)
        for index, message in enumerate(failures, start=1):
            print(f"{index}. {message}", file=sys.stderr)
        raise SystemExit(1)


def main() -> int:
    run_checks()
    print("[CONFIG OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
