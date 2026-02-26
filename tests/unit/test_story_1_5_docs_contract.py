from __future__ import annotations

import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
README_PATH = REPO_ROOT / "README.md"
MEMORY_README_PATH = REPO_ROOT / "memory" / "README.md"
ALEX_PROFILE_PATH = REPO_ROOT / "memory" / "profiles" / "alex.json"
ADR_DIR = REPO_ROOT / "docs" / "adr"
EXPECTED_ADRS = [
    "001-context-window-ownership.md",
    "002-multi-model-strategy.md",
    "003-anthropic-only-mvp.md",
    "004-file-based-memory.md",
    "005-langgraph-framework-selection.md",
]


def _readme_lines() -> list[str]:
    return README_PATH.read_text(encoding="utf-8").splitlines()


def _find_readme_line(lines: list[str], prefix: str) -> int:
    for idx, line in enumerate(lines):
        if line.strip() == prefix:
            return idx
    raise AssertionError(f"README is missing required line: {prefix!r}")


def test_readme_top_has_mermaid_and_jd_mapping_table() -> None:
    lines = _readme_lines()
    first_screen = "\n".join(lines[:90])
    assert "```mermaid" in first_screen, "README top section must include Mermaid topology"
    assert "## JD Mapping Table" in first_screen, "README top section must include JD mapping table"


def test_readme_navigating_section_is_immediately_after_mermaid() -> None:
    lines = _readme_lines()
    mermaid_start = _find_readme_line(lines, "```mermaid")

    mermaid_end = -1
    for idx in range(mermaid_start + 1, len(lines)):
        if lines[idx].strip() == "```":
            mermaid_end = idx
            break
    assert mermaid_end != -1, "README Mermaid block is not properly closed"

    nav_idx = _find_readme_line(lines, "## Navigating This Repo")
    assert nav_idx > mermaid_end, "Navigating section must appear after Mermaid block"
    assert nav_idx - mermaid_end <= 3, (
        "Navigating section must be immediately after Mermaid (allowing only blank spacer lines)"
    )


def test_readme_navigating_section_has_five_ordered_steps() -> None:
    lines = _readme_lines()
    nav_idx = _find_readme_line(lines, "## Navigating This Repo")
    step_pattern = re.compile(r"^\d+\.\s+")

    ordered_steps: list[str] = []
    for line in lines[nav_idx + 1 :]:
        if line.strip().startswith("## "):
            break
        if step_pattern.match(line.strip()):
            ordered_steps.append(line.strip())

    assert len(ordered_steps) == 5, "Navigating section must contain exactly 5 ordered steps"


def test_jd_mapping_table_has_nine_rows_and_at_least_seven_mapped() -> None:
    lines = _readme_lines()
    table_idx = _find_readme_line(lines, "## JD Mapping Table")

    table_lines: list[str] = []
    for line in lines[table_idx + 1 :]:
        stripped = line.strip()
        if stripped.startswith("## "):
            break
        if stripped.startswith("|"):
            table_lines.append(stripped)

    assert len(table_lines) >= 11, "JD mapping table must include header + separator + 9 data rows"

    data_rows: list[list[str]] = []
    for row in table_lines:
        if set(row.replace("|", "").strip()) <= {"-"}:
            continue
        cols = [col.strip() for col in row.split("|")[1:-1]]
        if cols and cols[0].lower() == "mvp item":
            continue
        if cols:
            data_rows.append(cols)

    assert len(data_rows) == 9, "JD mapping table must contain exactly 9 MVP rows"
    mapped_count = sum(1 for row in data_rows if len(row) >= 2 and row[1] not in {"", "TBD", "N/A"})
    assert mapped_count >= 7, "At least 7 of 9 MVP rows must map to specific JD bullets"


def test_adr_set_exists_and_each_file_is_concise_with_required_sections() -> None:
    for filename in EXPECTED_ADRS:
        path = ADR_DIR / filename
        assert path.exists(), f"Missing ADR file: {path}"
        lines = path.read_text(encoding="utf-8").splitlines()
        non_empty = [line for line in lines if line.strip()]
        assert len(non_empty) <= 20, f"{filename} must be <=20 non-empty lines"

        text = "\n".join(non_empty).lower()
        assert "decision context" in text, f"{filename} missing 'Decision Context'"
        assert "options considered" in text, f"{filename} missing 'Options Considered'"
        assert "rationale" in text, f"{filename} missing 'Rationale'"


def test_alex_profile_demonstrates_full_userprofile_schema() -> None:
    data = json.loads(ALEX_PROFILE_PATH.read_text(encoding="utf-8"))
    assert isinstance(data, dict), "alex.json must contain a JSON object"

    assert isinstance(data.get("user_id"), str) and data["user_id"], "user_id is required"
    assert isinstance(data.get("past_trips"), list) and data["past_trips"], (
        "past_trips list is required"
    )
    assert isinstance(data.get("preferences"), dict) and data["preferences"], (
        "preferences object is required"
    )
    assert isinstance(data.get("cached_research"), dict), (
        "cached_research session context is required"
    )
    assert isinstance(data.get("_demo_notes"), dict) and data["_demo_notes"], (
        "_demo_notes is required"
    )


def test_memory_readme_is_3_to_5_lines_and_explains_read_write_model() -> None:
    lines = [
        line.strip()
        for line in MEMORY_README_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert 3 <= len(lines) <= 5, "memory/README.md must be 3-5 non-empty lines"

    normalized = " ".join(lines).lower()
    for required_term in ("profile", "session", "read", "write"):
        assert required_term in normalized, f"memory/README.md must mention '{required_term}'"
