from __future__ import annotations

import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "spec" / "concierge-spec.md"
README_PATH = REPO_ROOT / "README.md"


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout.strip()


def _first_added_commit_epoch(path: str) -> int:
    output = _git("log", "--diff-filter=A", "--follow", "--format=%ct", "--", path)
    assert output, f"No git history found for {path}"
    return int(output.splitlines()[-1])


def _section_between(content: str, section_name: str) -> str:
    marker = re.search(rf"^## .*{re.escape(section_name)}\s*$", content, flags=re.MULTILINE)
    assert marker, f"Missing section header containing: {section_name}"
    start = marker.start()
    next_header = content.find("\n## ", start + len(section_name))
    if next_header == -1:
        return content[start:]
    return content[start:next_header]


def test_spec_committed_before_root_agents_nodes_graph_python_files() -> None:
    spec_commit = _first_added_commit_epoch("spec/concierge-spec.md")

    tracked_files = _git("ls-files").splitlines()
    constrained_prefixes = (
        "agents/",
        "nodes/",
        "graph/",
        "src/concierge/agents/",
        "src/concierge/nodes/",
        "src/concierge/graph/",
    )
    constrained_py = [
        p
        for p in tracked_files
        if p.endswith(".py")
        and p.startswith(constrained_prefixes)
    ]

    assert constrained_py, "No constrained implementation .py files found for AC1 validation"

    for py_file in constrained_py:
        py_commit = _first_added_commit_epoch(py_file)
        # Git stores add time at commit granularity; same-commit introduction is treated as compliant.
        assert spec_commit <= py_commit, (
            f"spec/concierge-spec.md must be committed no later than {py_file}. "
            f"spec={spec_commit}, {py_file}={py_commit}"
        )


def test_spec_first_line_has_datestamp_and_written_before_implementation() -> None:
    content = SPEC_PATH.read_text(encoding="utf-8")
    first_line = content.splitlines()[0]
    assert first_line.startswith("# "), "First line should be a markdown heading"
    assert "TNL-HELP Concierge Spec" in first_line, "Heading should retain spec title readability"
    assert re.search(r"\b\d{4}-\d{2}-\d{2}\b", first_line), (
        "First line must contain a YYYY-MM-DD datestamp"
    )
    assert "written before implementation" in first_line.lower(), (
        "First line must contain declaration 'written before implementation'"
    )


def test_spec_contains_required_contracts_and_decision_notes() -> None:
    content = SPEC_PATH.read_text(encoding="utf-8")

    required_state_fields = [
        "class ConciergeState(TypedDict):",
        "user_id:",
        "session_id:",
        "turn_id:",
        "conversation_history:",
        "current_input:",
        "current_response:",
        "intent:",
        "confidence:",
        "route:",
        "rag_results:",
        "research_results:",
        "source_attribution:",
        "memory_profile:",
        "guardrail_passed:",
        "proactive_suggestion:",
        "clarification_needed:",
        "clarification_question:",
        "human_handoff:",
        "error: str | None",
    ]
    for marker in required_state_fields:
        assert marker in content, f"Missing ConciergeState contract marker: {marker}"

    required_policy_fields = [
        "class AgentPolicy(BaseModel):",
        "agent_name:",
        "model:",
        "prompt_version:",
        "max_tokens:",
        "confidence_threshold:",
        "max_clarifications:",
        "allowed_tools:",
        "prompt_sections:",
    ]
    for marker in required_policy_fields:
        assert marker in content, f"Missing AgentPolicy contract marker: {marker}"

    required_edge_markers = [
        "## §GraphTopology",
        "START",
        "dispatcher_node",
        '(route == "rag")',
        '(route == "research")',
        '(route == "booking_stub")',
        '(route == "fallback")',
        "synthesis_node",
        "should_run_followup()",
        "followup_node",
        "END",
    ]
    for marker in required_edge_markers:
        assert marker in content, f"Missing graph topology marker: {marker}"

    major_decision_sections = [
        "AgentPolicy",
        "FollowUpCondition",
        "TokenBudgetManager",
    ]
    for section_header in major_decision_sections:
        section_content = _section_between(content, section_header)
        assert re.search(r"alternatives considered", section_content, flags=re.IGNORECASE), (
            f"Missing alternatives considered note in major decision section: {section_header}"
        )


def test_spec_has_at_least_two_tbd_markers() -> None:
    content = SPEC_PATH.read_text(encoding="utf-8")
    assert content.count("[TBD]") >= 2, "Spec must include at least 2 genuine [TBD] markers"


def test_readme_links_spec_as_authoritative_contract_source() -> None:
    assert README_PATH.exists(), "README.md must exist"
    content = README_PATH.read_text(encoding="utf-8")
    assert "spec/concierge-spec.md" in content, "README.md must link spec/concierge-spec.md"
    assert "authoritative source of agent contracts" in content.lower(), (
        "README.md must explicitly frame spec as the authoritative source of agent contracts"
    )
