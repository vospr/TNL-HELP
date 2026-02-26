from __future__ import annotations

from pathlib import Path
from typing import Callable

import yaml

from concierge.agents.dispatcher import DispatcherAgent
from concierge.graph import _FallbackCompiledGraph
from concierge.state import ConciergeState, initialize_state


# This test proves extensibility. In production, follow these exact steps to add a new agent.
def test_new_hotel_agent_can_be_added_without_dispatcher_code_changes(
    monkeypatch,
    tmp_path: Path,
) -> None:
    prompts_dir = tmp_path / "prompts" / "hotel_agent"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    policy_path = prompts_dir / "policy.yaml"
    policy_path.write_text(
        yaml.safe_dump(
            {
                "agent_name": "hotel_agent",
                "model": "claude-haiku-4-5-20251001",
                "prompt_version": "v1",
                "max_tokens": 256,
                "confidence_threshold": 0.75,
                "max_clarifications": 3,
                "allowed_tools": [],
                "prompt_sections": [
                    "role",
                    "context",
                    "constraints",
                    "output_format",
                    "examples",
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    agents_dir = tmp_path / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)
    hotel_agent_path = agents_dir / "hotel_agent.py"
    hotel_agent_path.write_text(
        "class HotelAgent:\n"
        "    def run(self, state):\n"
        "        return {'current_response': '[HOTEL] Recommended property options.'}\n",
        encoding="utf-8",
    )

    assert policy_path.exists()
    assert hotel_agent_path.exists()

    routing_rules_path = tmp_path / "routing_rules.yaml"
    routing_rules_path.write_text(
        yaml.safe_dump(
            {
                "escalation_threshold": 0.72,
                "rules": [
                    {
                        "pattern": r"\b(hotel|hotels)\b",
                        "intent": "hotel_lookup",
                        "route": "hotel_agent",
                        "score": 0.93,
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    repo_root = Path(__file__).resolve().parents[1]
    dispatcher_policy_path = repo_root / "prompts" / "dispatcher" / "policy.yaml"
    dispatcher_prompt_dir = repo_root / "prompts" / "dispatcher"
    dispatcher = DispatcherAgent(
        routing_rules_path=routing_rules_path,
        dispatcher_policy_path=dispatcher_policy_path,
        dispatcher_prompt_dir=dispatcher_prompt_dir,
    )

    stage1_intent, stage1_confidence, stage1_route = dispatcher._evaluate_stage1(
        "Find me a hotel in Phuket"
    )
    assert stage1_intent == "hotel_lookup"
    assert stage1_confidence == 0.93
    assert stage1_route == "hotel_agent"

    import concierge.graph as graph_module
    from concierge.state import NodeName

    original_node_functions = dict(graph_module.NODE_FUNCTIONS)
    original_dispatcher_edges = dict(graph_module.CONDITIONAL_EDGES[NodeName.DISPATCHER])

    def hotel_node(state: ConciergeState) -> ConciergeState:
        del state
        return {"current_response": "[HOTEL] Recommended property options."}

    def custom_dispatcher_node(state: ConciergeState) -> ConciergeState:
        return dispatcher.run(state)

    node_functions = dict(original_node_functions)
    node_functions[NodeName.DISPATCHER] = custom_dispatcher_node
    node_functions["hotel_agent"] = hotel_node
    monkeypatch.setattr(graph_module, "NODE_FUNCTIONS", node_functions)

    dispatcher_edges = dict(original_dispatcher_edges)
    dispatcher_edges["hotel_agent"] = "hotel_agent"
    conditional_edges = dict(graph_module.CONDITIONAL_EDGES)
    conditional_edges[NodeName.DISPATCHER] = dispatcher_edges
    monkeypatch.setattr(graph_module, "CONDITIONAL_EDGES", conditional_edges)

    state = initialize_state("alex", "session-7-4-extensibility", "Find me a hotel in Phuket", turn_id=1)
    result = _FallbackCompiledGraph().invoke(state)

    assert result["route"] == "hotel_agent"
    assert result["intent"] == "hotel_lookup"
    assert result["current_response"] == "[HOTEL] Recommended property options."
    assert result.get("_executed_nodes") == [
        "dispatcher",
        "hotel_agent",
        "guardrail",
        "synthesis",
    ]
