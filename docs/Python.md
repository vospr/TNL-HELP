# Python Coding Standards for AI Agents
# Compatible with: Claude Code, Cursor, GitHub Copilot, Gemini CLI, Windsurf
# Save this file as: CLAUDE.md | AGENTS.md | .cursorrules | .github/copilot-instructions.md

---

## 0. Your Core Mission

All code you write MUST be:
- **Correct** — works as intended, handles edge cases
- **Readable** — the next human (or AI) can understand it without asking
- **Maintainable** — easy to change, test, and extend
- **Secure** — no hardcoded secrets, no injection risks, no unsafe eval

If code does not meet these four criteria, do another pass before handing off.

---

## 1. Environment & Tooling

### Package Management
- Use `uv` as the default package manager (faster than pip, modern resolver)
- Always use a virtual environment: `uv venv .venv && source .venv/bin/activate`
- Pin all dependencies in `pyproject.toml`, not `requirements.txt` (unless legacy)
- Never run `pip install` globally — always into `.venv`

```bash
# Setup
uv venv .venv
source .venv/bin/activate      # Linux/macOS
.venv\Scripts\activate         # Windows

# Install deps
uv pip install -e ".[dev]"

# Add dependency
uv add pandas numpy
uv add --dev pytest ruff mypy
```

### Required Dev Tools
| Tool | Purpose | Config file |
|------|---------|-------------|
| `ruff` | Linting + formatting (replaces black, flake8, isort) | `pyproject.toml` |
| `mypy` | Static type checking | `pyproject.toml` |
| `pytest` | Testing | `pyproject.toml` |
| `pre-commit` | Git hooks for auto-checks | `.pre-commit-config.yaml` |

### Automation Commands
```bash
# Format and lint
ruff format .
ruff check . --fix

# Type check
mypy src/

# Test
pytest tests/ -v --tb=short

# All checks before commit
ruff format . && ruff check . && mypy src/ && pytest tests/
```

---

## 2. Project Structure

```
project-root/
├── src/
│   └── package_name/          # Core source — ALWAYS use src/ layout
│       ├── __init__.py
│       ├── core/              # Core business logic
│       ├── utils/             # Shared utilities
│       ├── models/            # Data models / schemas
│       └── config.py          # Configuration (not secrets)
├── tests/
│   ├── unit/                  # Fast, isolated tests
│   ├── integration/           # Tests with external dependencies
│   └── conftest.py            # Shared fixtures
├── notebooks/                 # Jupyter notebooks (exploration only, not production)
│   └── eda/
├── scripts/                   # One-off scripts, not importable modules
├── docs/
├── pyproject.toml             # Single source of truth for all tool config
├── .env.example               # Template for env vars (no real values)
├── .gitignore
├── CLAUDE.md                  # This file
└── README.md
```

**Key rules:**
- Always use `src/` layout — prevents accidental imports of local files over installed packages
- `notebooks/` is for exploration only — never import from notebooks into `src/`
- Keep `scripts/` for CLI entrypoints, not business logic

---

## 3. Naming Conventions

| Element | Convention | Example |
|---------|-----------|---------|
| Variables | `snake_case` | `user_count`, `file_path` |
| Functions | `snake_case` | `calculate_total()`, `get_user()` |
| Classes | `PascalCase` | `DataProcessor`, `UserProfile` |
| Constants | `UPPER_SNAKE_CASE` | `MAX_RETRIES = 3`, `DEFAULT_TIMEOUT = 30` |
| Private | `_leading_underscore` | `_internal_helper()` |
| Modules/files | `snake_case` | `data_loader.py`, `feature_engineering.py` |
| Type aliases | `PascalCase` | `DataFrameType = pd.DataFrame` |

**Name things by what they ARE or DO, not how they work:**
```python
# BAD
def calc(x, y, z):  # cryptic
    tmp = x + y
    return tmp * z

# GOOD
def calculate_weighted_score(base_score: float, bonus: float, weight: float) -> float:
    total = base_score + bonus
    return total * weight
```

---

## 4. Type Hints (MANDATORY)

Use Python 3.10+ union syntax. Type hints are required on ALL function signatures.

```python
# CORRECT — Python 3.10+ style
def process_records(
    records: list[dict[str, str]],
    max_count: int | None = None,
    strict: bool = False,
) -> list[str]:
    ...

# WRONG — old style
def process_records(records: List[Dict[str, str]], max_count: Optional[int] = None) -> List[str]:
    ...
```

**When to use `Any`:** Almost never. If you must, add a comment explaining why.

```python
# Acceptable only when interfacing with untyped third-party code
result: Any  # third-party lib returns unspecified structure — see issue #42
```

---

## 5. Docstrings

Use **NumPy-style** docstrings for all public functions, classes, and modules.

```python
def train_model(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.01,
    epochs: int = 100,
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Train a model using gradient descent.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,).
    learning_rate : float, optional
        Step size for gradient update. Default is 0.01.
    epochs : int, optional
        Number of training iterations. Default is 100.

    Returns
    -------
    weights : np.ndarray
        Trained model weights of shape (n_features,).
    metrics : dict[str, float]
        Training metrics including 'loss' and 'accuracy'.

    Raises
    ------
    ValueError
        If X and y have incompatible shapes.

    Examples
    --------
    >>> X = np.random.randn(100, 5)
    >>> y = np.random.randint(0, 2, 100)
    >>> weights, metrics = train_model(X, y, learning_rate=0.001)
    """
```

**Docstring rules:**
- One-liner for trivial helpers: `"""Return the square of x."""`
- Full NumPy docstring for any public API function
- Describe WHAT and WHY in comments, not HOW (the code shows how)
- Keep docstrings evergreen — no "recently refactored" or "new in v2"

---

## 6. Error Handling

```python
# BAD — catches everything silently
try:
    result = process(data)
except:
    pass

# BAD — too broad, hides bugs
try:
    result = process(data)
except Exception:
    return None

# GOOD — specific, informative, logged
import logging
logger = logging.getLogger(__name__)

try:
    result = process(data)
except ValueError as e:
    logger.error("Invalid data format in process(): %s", e)
    raise  # re-raise to let caller decide what to do
except FileNotFoundError as e:
    logger.error("Data file missing: %s", e)
    raise RuntimeError(f"Cannot process data — file not found: {e}") from e
```

**Error message rules:**
- Be specific: `"Expected positive integer, got -5"` not `"Invalid input"`
- Include context: `f"Failed to process {source_file}: {e}"`  
- Suggest fixes when possible: `"Use ISO format: YYYY-MM-DD"`
- Always use `raise ... from e` to preserve the original traceback

---

## 7. Function Design

```python
# BAD — mixed concerns, side effects, no return
def process_user_data(df):
    df['age'] = df['age'].fillna(0)
    df.to_csv('output.csv')  # side effect!
    print("Done")             # side effect!
    
# GOOD — pure, single responsibility, explicit I/O
def fill_missing_ages(df: pd.DataFrame, fill_value: int = 0) -> pd.DataFrame:
    """Return a copy of df with missing ages filled."""
    return df.assign(age=df['age'].fillna(fill_value))
```

**Rules:**
- Functions do ONE thing
- Avoid modifying inputs — return new objects (especially for DataFrames)
- Separate computation from I/O — don't mix data processing with file writes
- Max ~50 lines per function; if longer, split it
- Avoid positional-only booleans: use keyword args `process(data, strict=True)` not `process(data, True)`
- No magic numbers — name your constants

---

## 8. Class Design

```python
from dataclasses import dataclass, field

@dataclass
class ModelConfig:
    """Configuration for ML model training."""
    
    learning_rate: float = 0.01
    epochs: int = 100
    batch_size: int = 32
    feature_columns: list[str] = field(default_factory=list)
    
    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
```

**Class rules:**
- Prefer `@dataclass` or `pydantic.BaseModel` over plain classes for data containers
- Use `@property` for computed attributes, not getter methods
- Keep `__init__` simple — heavy setup belongs in a `@classmethod` factory
- No more than one level of inheritance by default (composition over inheritance)
- Validate in `__post_init__` or model validators, not scattered across methods

---

## 9. Imports

```python
# Order: stdlib → third-party → local (ruff/isort enforces this)
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from package_name.core.processor import DataProcessor
from package_name.utils.helpers import format_output
```

**Rules:**
- Never use `from module import *` — always explicit imports
- Prefer absolute imports over relative ones in application code
- Group and separate with blank lines: stdlib, third-party, local
- Put expensive imports inside functions only if import time matters

---

## 10. Testing (TDD Approach)

Follow the Red → Green → Refactor cycle:

1. Write a FAILING test that describes expected behavior
2. Write minimum code to make it pass
3. Refactor while keeping tests green

```python
# tests/unit/test_processor.py
import pytest
from package_name.core.processor import DataProcessor

class TestDataProcessor:
    """Tests for DataProcessor.process() method."""

    def test_process_returns_nonempty_result_for_valid_input(self):
        processor = DataProcessor(config={"threshold": 0.5})
        result = processor.process([1, 2, 3])
        assert len(result) > 0

    def test_process_raises_value_error_for_empty_input(self):
        processor = DataProcessor(config={"threshold": 0.5})
        with pytest.raises(ValueError, match="Input cannot be empty"):
            processor.process([])

    def test_process_filters_below_threshold(self):
        processor = DataProcessor(config={"threshold": 2.0})
        result = processor.process([1.0, 2.5, 3.0])
        assert all(v >= 2.0 for v in result)

    @pytest.mark.parametrize("values,expected", [
        ([1, 2, 3], 3),
        ([0], 1),
        ([5, 5, 5], 3),
    ])
    def test_process_count(self, values, expected):
        processor = DataProcessor(config={"threshold": 0.0})
        assert len(processor.process(values)) == expected
```

**Testing rules:**
- 1 test = 1 behavior (not 1 test = 1 function)
- Test the behavior from the outside, not internal implementation
- Use `pytest.mark.parametrize` for multiple input cases
- Use fixtures in `conftest.py` for shared setup
- Name tests: `test_<what>_<condition>_<expected>`
- Aim for high coverage on `src/` — especially edge cases and error paths

---

## 11. Security

```python
# NEVER hardcode secrets
API_KEY = "sk-abc123..."  # ❌ NEVER

# ALWAYS load from environment
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.environ["API_KEY"]  # raises KeyError if missing — good!
# or with default: os.getenv("API_KEY", "")  # silent — use carefully
```

**Security rules:**
- `.env` files MUST be in `.gitignore` — always. Provide `.env.example` instead
- Never use `eval()`, `exec()`, or `pickle.loads()` on untrusted input
- Parameterize SQL queries — never string-format SQL: `cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))`
- Validate and sanitize all external inputs (API, user, file)
- Use `subprocess` with a list, not a string: `subprocess.run(["git", "status"])` not `subprocess.run("git status", shell=True)`

---

## 12. Data Science & MLOps Specifics

### Pandas / Polars
```python
# PREFER Polars for large datasets (faster, more explicit)
import polars as pl

# If using Pandas, always use .copy() to avoid SettingWithCopyWarning
subset = df[df["age"] > 18].copy()
subset["category"] = "adult"  # safe

# Use method chaining for readability
result = (
    df
    .filter(pl.col("age") > 18)
    .select(["name", "age", "score"])
    .sort("score", descending=True)
)
```

### Notebooks
- Notebooks in `notebooks/` are for EDA and exploration ONLY
- Extract any reusable logic from notebooks into `src/` functions
- Use clear markdown cells to explain each section
- Restart and run all before committing — ensure clean top-to-bottom execution

### ML Models
```python
# Always log experiment parameters and metrics
import mlflow

with mlflow.start_run():
    mlflow.log_params({"lr": 0.01, "epochs": 100})
    # ... train ...
    mlflow.log_metrics({"accuracy": 0.94, "loss": 0.12})
    mlflow.sklearn.log_model(model, "model")
```

### Random Seeds
```python
# Always set seeds for reproducibility
import random
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
# torch.manual_seed(SEED)  # if using PyTorch
```

---

## 13. Python Best Practices

```python
# Use context managers for resources
with open("data.csv", "r") as f:
    data = f.read()

# Use enumerate() instead of range(len())
for i, item in enumerate(items):  # ✅
    ...
for i in range(len(items)):        # ❌

# Use f-strings (not .format() or %)
name = "world"
msg = f"Hello, {name}!"  # ✅
msg = "Hello, %s!" % name  # ❌

# Compare to None with `is`, not `==`
if result is None:  # ✅
    ...
if result == None:  # ❌
    ...

# Use Path instead of os.path
from pathlib import Path
config_path = Path("config") / "settings.yaml"  # ✅
config_path = os.path.join("config", "settings.yaml")  # ❌

# Use list comprehensions, not map/filter with lambda
squares = [x**2 for x in range(10)]  # ✅
squares = list(map(lambda x: x**2, range(10)))  # ❌
```

---

## 14. Anchor Comments (for AI Navigation)

Use these standard anchor comments so AI agents and humans can quickly find important spots:

```python
# AIDEV-NOTE: Why this non-obvious decision was made
# AIDEV-TODO: What needs to be done before production
# AIDEV-FIXME: Known bug or broken behavior
# AIDEV-QUESTION: Uncertainty that needs human review
# AIDEV-PERF: Performance-critical section — think before changing
```

Example:
```python
def load_embeddings(path: Path) -> np.ndarray:
    # AIDEV-NOTE: Using memmap instead of np.load to avoid OOM on large files
    # AIDEV-PERF: This is called once at startup — latency here is acceptable
    return np.load(path, mmap_mode="r")
```

---

## 15. Before Every Commit

Run this checklist. All must pass:

```bash
# 1. Format
ruff format .

# 2. Lint
ruff check . --fix

# 3. Type check
mypy src/

# 4. Tests
pytest tests/ -v

# 5. Security scan (optional but recommended)
bandit -r src/ -ll
```

Also verify:
- [ ] No `.env` or secrets committed (`git diff --cached | grep -i "api_key\|password\|secret"`)
- [ ] No `print()` left in production code (use `logger.debug()` instead)
- [ ] Docstrings on all public functions
- [ ] New features have tests

---

## 16. pyproject.toml Reference

```toml
[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "W", "F", "I", "B", "C4", "UP", "N", "SIM"]
ignore = ["E501"]  # line length handled by formatter

[tool.mypy]
python_version = "3.10"
strict = true
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short --strict-markers"

[tool.coverage.run]
source = ["src"]
omit = ["tests/*", "notebooks/*"]
```
