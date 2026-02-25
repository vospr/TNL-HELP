"""
Tests that .env.example exists with ANTHROPIC_API_KEY and .env is NOT committed.
AC4: .env.example documents ANTHROPIC_API_KEY without a value; no .env in repo.
"""
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent.parent


def test_env_example_exists() -> None:
    assert (REPO_ROOT / ".env.example").exists(), ".env.example must exist"


def test_env_file_not_present() -> None:
    assert not (REPO_ROOT / ".env").exists(), (
        ".env must never be committed — it contains secrets"
    )


def test_env_example_contains_anthropic_key() -> None:
    content = (REPO_ROOT / ".env.example").read_text()
    assert "ANTHROPIC_API_KEY" in content, (
        ".env.example must document ANTHROPIC_API_KEY"
    )


def test_env_example_key_has_no_value() -> None:
    content = (REPO_ROOT / ".env.example").read_text()
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("ANTHROPIC_API_KEY"):
            # Must be ANTHROPIC_API_KEY= with nothing after the =
            assert line == "ANTHROPIC_API_KEY=", (
                f"ANTHROPIC_API_KEY must have no value in .env.example, got: {line!r}"
            )
            return
    raise AssertionError("ANTHROPIC_API_KEY line not found in .env.example")
