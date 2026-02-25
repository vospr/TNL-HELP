"""
Integration test conftest — autouse mock Anthropic client fixture.
Real API calls are never made in pytest; use tests/manual/ for live API tests.
"""
from unittest.mock import MagicMock, patch
import pytest


@pytest.fixture(autouse=True)
def mock_anthropic_client(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """
    Patches anthropic.Anthropic for all integration tests.
    Real API calls will fail here with a clear message — use tests/manual/ for live tests.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-sk-fake-do-not-use")
    mock = MagicMock()
    with patch("anthropic.Anthropic", return_value=mock):
        yield mock
