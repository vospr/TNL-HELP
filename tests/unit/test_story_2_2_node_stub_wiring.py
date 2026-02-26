from __future__ import annotations

import ast
from copy import deepcopy
from pathlib import Path

from concierge.state import initialize_state


REPO_ROOT = Path(__file__).resolve().parents[2]
NODES_DIR = REPO_ROOT / "src" / "concierge" / "nodes"

NODE_FILES = [
    "dispatcher_node.py",
    "rag_node.py",
    "research_node.py",
    "booking_node.py",
    "guardrail_node.py",
    "synthesis_node.py",
    "followup_node.py",
]


def test_nodes_package_contains_all_7_node_modules() -> None:
    for filename in NODE_FILES:
        assert (NODES_DIR / filename).exists(), f"Missing node module: {filename}"


def test_each_node_module_has_docstring_and_single_delegate_function() -> None:
    for filename in NODE_FILES:
        path = NODES_DIR / filename
        module = ast.parse(path.read_text(encoding="utf-8"))
        assert ast.get_docstring(module), f"{filename} must include a module docstring"

        functions = [node for node in module.body if isinstance(node, ast.FunctionDef)]
        assert len(functions) == 1, f"{filename} must define exactly one function"

        function_def = functions[0]
        assert any(isinstance(node, ast.Try) for node in function_def.body), (
            f"{filename} function must use node-boundary try/except handling"
        )

        run_calls = 0
        for node in ast.walk(function_def):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "run"
            ):
                run_calls += 1

        assert run_calls >= 1, f"{filename} must delegate to agent.run(state)"


def test_compiled_graph_exports_without_errors() -> None:
    from concierge.graph import compiled_graph

    assert compiled_graph is not None
    assert hasattr(compiled_graph, "invoke")


def test_stub_graph_sequence_for_rag_route_and_turn_id_preserved() -> None:
    from concierge.graph import compiled_graph

    state = initialize_state(
        user_id="alex",
        session_id="session-2-2-rag",
        current_input="hello",
        turn_id=7,
    )
    state["route"] = "rag"
    before = deepcopy(state)

    result = compiled_graph.invoke(state)

    assert state == before, "input state must not be mutated in place"
    assert result["turn_id"] == 7, "stub path must preserve turn_id"
    assert result["route"] == "rag"
    assert result.get("_executed_nodes") == [
        "dispatcher",
        "rag",
        "guardrail",
        "synthesis",
    ]


def test_stub_graph_sequence_for_research_and_booking_routes() -> None:
    from concierge.graph import compiled_graph

    research = initialize_state("alex", "session-2-2-research", "hello", turn_id=11)
    research["route"] = "research"
    research_result = compiled_graph.invoke(research)
    assert research_result.get("_executed_nodes") == [
        "dispatcher",
        "research",
        "guardrail",
        "synthesis",
        "followup",
    ]
    assert research_result["turn_id"] == 11

    booking = initialize_state("alex", "session-2-2-booking", "hello", turn_id=13)
    booking["route"] = "booking_stub"
    booking_result = compiled_graph.invoke(booking)
    assert booking_result.get("_executed_nodes") == [
        "dispatcher",
        "booking_stub",
        "guardrail",
        "synthesis",
    ]
    assert booking_result["turn_id"] == 13


def test_stub_agents_keep_core_state_fields_unchanged() -> None:
    from concierge.graph import compiled_graph

    state = initialize_state("alex", "session-2-2-unchanged", "hello", turn_id=2)
    state["route"] = "research"
    state["memory_profile"] = {"preferred_name": "Alex"}
    state["conversation_history"] = [{"role": "user", "content": "hello"}]

    result = compiled_graph.invoke(state)

    assert result["user_id"] == state["user_id"]
    assert result["session_id"] == state["session_id"]
    assert result["memory_profile"] == state["memory_profile"]
    assert result["conversation_history"] == state["conversation_history"]
    assert result["current_input"] == state["current_input"]
    assert result["rag_results"] == state["rag_results"]
    assert result["research_results"] is None or isinstance(result["research_results"], list)
