from __future__ import annotations

import importlib

import pytest

from concierge.state import initialize_state


def test_dispatcher_node_catches_timeout_with_actionable_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from concierge.nodes import dispatcher_node as module

    traces: list[tuple[str, dict[str, object]]] = []

    def _capture_trace(node_name: str, outcome: str | None = None, **fields: object) -> None:
        del outcome
        traces.append((node_name, dict(fields)))

    def _boom(self, state):  # noqa: ANN001
        del self, state
        raise TimeoutError("request timed out")

    monkeypatch.setattr(module, "trace", _capture_trace)
    monkeypatch.setattr(module.DispatcherAgent, "run", _boom)

    state = initialize_state("alex", "session-3-6-timeout", "hello", turn_id=3)
    update = module.dispatcher_node(state)

    assert update["error"] == "LLM connection timeout. Please try again."
    assert update["human_handoff"] is True
    assert traces == [("dispatcher", {"turn_id": 3, "status": "start"})]


def test_dispatcher_node_catches_invalid_key_with_exact_fix_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from concierge.nodes import dispatcher_node as module

    def _boom(self, state):  # noqa: ANN001
        del self, state
        raise RuntimeError("ANTHROPIC_API_KEY not set or invalid")

    monkeypatch.setattr(module.DispatcherAgent, "run", _boom)
    monkeypatch.setattr(module, "trace", lambda *args, **kwargs: None)

    state = initialize_state("alex", "session-3-6-key", "hello", turn_id=4)
    update = module.dispatcher_node(state)

    assert update["error"] == "ANTHROPIC_API_KEY not set or invalid"
    assert update["human_handoff"] is True


@pytest.mark.parametrize(
    ("module_name", "node_fn_name", "agent_cls_name"),
    [
        ("concierge.nodes.dispatcher_node", "dispatcher_node", "DispatcherAgent"),
        ("concierge.nodes.rag_node", "rag_node", "RAGAgent"),
        ("concierge.nodes.research_node", "research_node", "ResearchAgent"),
        ("concierge.nodes.booking_node", "booking_node", "BookingAgent"),
        ("concierge.nodes.guardrail_node", "guardrail_node", "GuardrailAgent"),
        ("concierge.nodes.synthesis_node", "synthesis_node", "ResponseSynthesisAgent"),
        ("concierge.nodes.followup_node", "followup_node", "FollowUpAgent"),
    ],
)
def test_every_node_catches_exceptions_and_returns_error_update(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    node_fn_name: str,
    agent_cls_name: str,
) -> None:
    module = importlib.import_module(module_name)
    node_fn = getattr(module, node_fn_name)
    agent_cls = getattr(module, agent_cls_name)

    def _boom(self, state):  # noqa: ANN001
        del self, state
        raise RuntimeError("simulated node failure")

    monkeypatch.setattr(agent_cls, "run", _boom)
    monkeypatch.setattr(module, "trace", lambda *args, **kwargs: None)

    state = initialize_state("alex", f"session-3-6-{node_fn_name}", "hello", turn_id=5)
    update = node_fn(state)

    assert update["human_handoff"] is True
    assert update["error"]


def test_graph_continues_after_dispatcher_error_and_terminal_response_handles_handoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from concierge.agents.dispatcher import DispatcherAgent
    from concierge.graph import compiled_graph

    def _boom(self, state):  # noqa: ANN001
        del self, state
        raise TimeoutError("simulated timeout")

    monkeypatch.setattr(DispatcherAgent, "run", _boom)

    state = initialize_state("alex", "session-3-6-graph", "hello", turn_id=6)
    state["route"] = None

    result = compiled_graph.invoke(state)

    assert result["error"] == "LLM connection timeout. Please try again."
    assert result["human_handoff"] is True
    assert result["current_response"].endswith(
        "A human travel specialist can help - please wait"
    )
    assert result.get("_executed_nodes") == [
        "dispatcher",
        "guardrail",
        "synthesis",
    ]
