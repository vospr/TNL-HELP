"""
Tests that pyproject.toml exists and is valid TOML with required structure.
AC2: supports reproducible dependency management via uv.
"""
from pathlib import Path
import tomllib


REPO_ROOT = Path(__file__).parent.parent.parent


def test_pyproject_toml_exists() -> None:
    assert (REPO_ROOT / "pyproject.toml").exists(), "pyproject.toml must exist"


def test_pyproject_toml_is_valid_toml() -> None:
    content = (REPO_ROOT / "pyproject.toml").read_bytes()
    parsed = tomllib.loads(content.decode())
    assert "project" in parsed, "pyproject.toml must have [project] section"


def test_pyproject_requires_python_311() -> None:
    content = (REPO_ROOT / "pyproject.toml").read_bytes()
    parsed = tomllib.loads(content.decode())
    requires = parsed["project"].get("requires-python", "")
    assert "3.11" in requires, (
        f"pyproject.toml must require Python 3.11+, got: {requires!r}"
    )


def test_uv_lock_exists() -> None:
    assert (REPO_ROOT / "uv.lock").exists(), (
        "uv.lock must be committed for exact reproducibility"
    )


def test_pyproject_has_pytest_config() -> None:
    content = (REPO_ROOT / "pyproject.toml").read_bytes()
    parsed = tomllib.loads(content.decode())
    tool = parsed.get("tool", {})
    pytest_section = tool.get("pytest", {})
    assert "ini_options" in pytest_section, (
        "pyproject.toml must contain [tool.pytest.ini_options]"
    )
