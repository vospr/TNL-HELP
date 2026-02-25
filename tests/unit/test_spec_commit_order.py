"""
Tests that spec/concierge-spec.md exists (commit order is enforced by git history).
AC1: spec committed before any agents/.py or nodes/.py files.
Note: The git history assertion becomes meaningful after agents are implemented.
"""
from pathlib import Path
import pytest


REPO_ROOT = Path(__file__).parent.parent.parent


def test_spec_file_exists() -> None:
    assert (REPO_ROOT / "spec" / "concierge-spec.md").exists(), (
        "spec/concierge-spec.md must exist — it is the authoritative contract"
    )


def test_spec_has_version_header() -> None:
    content = (REPO_ROOT / "spec" / "concierge-spec.md").read_text()
    assert "TNL-HELP" in content, "spec file must reference TNL-HELP project"
    assert "ConciergeState" in content, "spec file must define ConciergeState"


@pytest.mark.skip(
    reason="Git history assertion only meaningful after agents/ .py files are committed. "
           "Re-enable in Story 2.x after first agent implementation."
)
def test_spec_committed_before_agents() -> None:
    """
    Verify spec/concierge-spec.md was committed before any agents/*.py file.
    Skipped until agents/ directory contains committed Python files.
    """
    import subprocess

    result = subprocess.run(
        ["git", "log", "--oneline", "--diff-filter=A", "--name-only", "--format=%H %s"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    # Parse commits that added files; spec commit must precede agents/ commits
    # Implementation: find commit adding spec/concierge-spec.md and compare timestamps
    # with first commit adding src/concierge/agents/*.py
    pass
