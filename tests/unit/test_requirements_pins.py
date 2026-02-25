"""
Tests that requirements.txt uses exact == pins and contains required packages.
AC2: all dependencies use == and include langgraph, anthropic, pydantic,
     duckduckgo-search, and pytest.
"""
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent.parent
REQUIRED_PACKAGES = ["langgraph", "anthropic", "pydantic", "duckduckgo-search", "pytest"]


def _get_package_lines() -> list[str]:
    req = (REPO_ROOT / "requirements.txt").read_text()
    return [
        line.strip()
        for line in req.splitlines()
        if line.strip() and not line.startswith(("#", "-", " ", "\t"))
    ]


def test_requirements_txt_exists() -> None:
    assert (REPO_ROOT / "requirements.txt").exists(), "requirements.txt must exist"


def test_all_packages_use_exact_pins() -> None:
    for line in _get_package_lines():
        assert "==" in line, (
            f"Dependency not exact-pinned with ==: {line!r}. "
            "All deps must use == for reproducibility."
        )


def test_required_packages_present() -> None:
    req_content = (REPO_ROOT / "requirements.txt").read_text()
    for pkg in REQUIRED_PACKAGES:
        assert pkg in req_content, (
            f"Required package missing from requirements.txt: {pkg}"
        )
