"""
Tests that .gitignore contains required entries.
AC3: memory/sessions/ listed in .gitignore; only .gitkeep tracked.
"""
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent.parent


def test_gitignore_exists() -> None:
    assert (REPO_ROOT / ".gitignore").exists()


def test_gitignore_covers_sessions_json() -> None:
    content = (REPO_ROOT / ".gitignore").read_text()
    assert "memory/sessions" in content or "sessions/*.json" in content, (
        ".gitignore must cover memory/sessions/*.json"
    )


def test_gitignore_covers_env() -> None:
    content = (REPO_ROOT / ".gitignore").read_text()
    assert ".env" in content, ".gitignore must include .env"


def test_gitignore_covers_venv() -> None:
    content = (REPO_ROOT / ".gitignore").read_text()
    assert ".venv" in content or "venv/" in content, ".gitignore must cover .venv/"


def test_gitignore_covers_pycache() -> None:
    content = (REPO_ROOT / ".gitignore").read_text()
    assert "__pycache__" in content, ".gitignore must cover __pycache__/"


def test_gitkeep_present_in_sessions() -> None:
    assert (REPO_ROOT / "memory" / "sessions" / ".gitkeep").exists(), (
        "memory/sessions/.gitkeep must exist so git tracks the directory"
    )


def test_gitignore_negation_preserves_gitkeep() -> None:
    content = (REPO_ROOT / ".gitignore").read_text()
    assert "!memory/sessions/.gitkeep" in content, (
        ".gitignore must have !memory/sessions/.gitkeep to preserve the sentinel file"
    )
