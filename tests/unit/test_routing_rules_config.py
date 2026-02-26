from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
ROUTING_RULES_PATH = REPO_ROOT / "config" / "routing_rules.yaml"
PROMPTS_DIR = REPO_ROOT / "prompts"
POLICY_AGENTS = [
    "dispatcher",
    "rag_agent",
    "research_agent",
    "response_synthesis",
    "guardrail",
    "followup",
]


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    assert isinstance(data, dict), f"{path} must contain a YAML mapping/object"
    return data


def test_routing_rules_has_escalation_threshold_and_rules() -> None:
    data = _load_yaml(ROUTING_RULES_PATH)

    threshold = data.get("escalation_threshold")
    assert isinstance(threshold, (int, float)), "escalation_threshold must be numeric"
    assert 0.0 <= float(threshold) <= 1.0, "escalation_threshold must be in [0.0, 1.0]"

    rules = data.get("rules")
    assert isinstance(rules, list) and rules, "rules must be a non-empty list"
    for idx, rule in enumerate(rules, start=1):
        assert isinstance(rule, dict), f"rules[{idx}] must be an object"
        assert isinstance(rule.get("pattern"), str) and rule["pattern"].strip(), (
            f"rules[{idx}].pattern must be non-empty string"
        )
        assert isinstance(rule.get("intent"), str) and rule["intent"].strip(), (
            f"rules[{idx}].intent must be non-empty string"
        )
        assert isinstance(rule.get("route"), str) and rule["route"].strip(), (
            f"rules[{idx}].route must be non-empty string"
        )
        score = rule.get("score")
        assert isinstance(score, (int, float)) and 0.0 <= float(score) <= 1.0, (
            f"rules[{idx}].score must be numeric in [0.0, 1.0]"
        )


def test_routing_rule_patterns_use_word_boundaries() -> None:
    data = _load_yaml(ROUTING_RULES_PATH)
    rules = data.get("rules") or []
    for idx, rule in enumerate(rules, start=1):
        pattern = str(rule.get("pattern", ""))
        assert "\\b" in pattern, (
            f"rules[{idx}].pattern should use word boundaries to reduce false-positive matches"
        )


def test_routing_rules_have_inline_rule_comments() -> None:
    raw = ROUTING_RULES_PATH.read_text(encoding="utf-8")
    rule_lines = [
        line
        for line in raw.splitlines()
        if line.strip().startswith("- pattern:")
    ]
    assert rule_lines, "routing_rules.yaml must define rule entries using '- pattern:'"
    for line in rule_lines:
        assert "#" in line, (
            "Each routing rule line must include an inline comment explaining intent"
        )


def test_escalation_threshold_distinct_from_policy_confidence_thresholds() -> None:
    routing = _load_yaml(ROUTING_RULES_PATH)
    escalation_threshold = float(routing["escalation_threshold"])

    policy_thresholds = set()
    for agent in POLICY_AGENTS:
        policy = _load_yaml(PROMPTS_DIR / agent / "policy.yaml")
        confidence = policy.get("confidence_threshold")
        assert isinstance(confidence, (int, float)), (
            f"prompts/{agent}/policy.yaml confidence_threshold must be numeric"
        )
        policy_thresholds.add(float(confidence))

    assert escalation_threshold not in policy_thresholds, (
        "escalation_threshold must be distinct from all per-agent confidence_threshold values"
    )
