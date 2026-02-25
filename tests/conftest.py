"""
Root conftest.py — shared state factories and tmp path helpers.
No API mocking at root level (per architecture decision: integration/conftest.py owns that).
"""
from pathlib import Path
import pytest


REPO_ROOT = Path(__file__).parent.parent


@pytest.fixture
def repo_root() -> Path:
    """Returns the absolute path to the repository root."""
    return REPO_ROOT


@pytest.fixture
def tmp_memory_dir(tmp_path: Path) -> Path:
    """Creates a temporary memory directory structure for tests."""
    profiles = tmp_path / "memory" / "profiles"
    sessions = tmp_path / "memory" / "sessions"
    profiles.mkdir(parents=True)
    sessions.mkdir(parents=True)
    (sessions / ".gitkeep").touch()
    return tmp_path / "memory"
