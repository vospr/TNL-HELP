from __future__ import annotations

import sys
import types
from pathlib import Path

import yaml

from concierge.agents.dispatcher import DispatcherAgent
from concierge.state import initialize_state


def _install_fake_anthropic(monkeypatch, response_text: str) -> dict[str, object]:
    recorder: dict[str, object] = {}

    class _FakeMessages:
        def create(self, **kwargs):
            recorder["kwargs"] = kwargs
            return types.SimpleNamespace(
                content=[types.SimpleNamespace(text=response_text)]
            )

    class _FakeClient:
        def __init__(self) -> None:
            self.messages = _FakeMessages()

    fake_module = types.SimpleNamespace(Anthropic=lambda: _FakeClient())
    monkeypatch.setitem(sys.modules, "anthropic", fake_module)
    return recorder


def test_stage2_uses_llm_model_and_routes_when_confidence_high(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("FAST_MODE", raising=False)
    recorder = _install_fake_anthropic(
        monkeypatch,
        '{"intent":"destination_research","confidence":0.84}',
    )

    traces: list[tuple[str, dict[str, object]]] = []

    def _capture_trace(node_name: str, outcome: str | None = None, **fields: object) -> None:
        del outcome
        traces.append((node_name, fields))

    monkeypatch.setattr("concierge.agents.dispatcher.trace", _capture_trace)

    state = initialize_state(
        user_id="alex",
        session_id="session-3-4-high",
        current_input="I'm thinking Southeast Asia but not sure where",
        turn_id=1,
    )
    update = DispatcherAgent().run(state)

    create_kwargs = recorder["kwargs"]
    assert create_kwargs["model"] == "claude-opus-4-6"
    assert create_kwargs["max_tokens"] == 128
    assert update["intent"] == "destination_research"
    assert update["confidence"] == 0.84
    assert update["route"] == "research"
    assert traces[-1] == (
        "dispatcher",
        {
            "intent": "destination_research",
            "confidence": 0.84,
            "route": "research",
            "stage": "llm_escalation",
        },
    )


def test_stage2_low_confidence_routes_to_fallback(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    _install_fake_anthropic(
        monkeypatch,
        '{"intent":"destination_research","confidence":0.61}',
    )

    traces: list[tuple[str, dict[str, object]]] = []

    def _capture_trace(node_name: str, outcome: str | None = None, **fields: object) -> None:
        del outcome
        traces.append((node_name, fields))

    monkeypatch.setattr("concierge.agents.dispatcher.trace", _capture_trace)

    state = initialize_state(
        user_id="alex",
        session_id="session-3-4-low",
        current_input="something else",
        turn_id=2,
    )
    update = DispatcherAgent().run(state)

    assert update["intent"] == "destination_research"
    assert update["confidence"] == 0.61
    assert update["route"] == "fallback"
    assert traces[-1][1]["stage"] == "llm_escalation"


def test_dispatcher_prompt_contract_requires_json_intent_confidence_only() -> None:
    prompt_path = Path(__file__).resolve().parents[2] / "prompts" / "dispatcher" / "v1.yaml"
    with prompt_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    assert "output_format" in data
    output_format = str(data["output_format"]).lower()
    assert "intent" in output_format and "confidence" in output_format
    assert "route" not in output_format
    assert "rationale" not in output_format
