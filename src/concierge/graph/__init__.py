from __future__ import annotations

from copy import deepcopy
from typing import Any

from concierge.nodes.booking_node import booking_node
from concierge.nodes.dispatcher_node import dispatcher_node
from concierge.nodes.followup_node import followup_node
from concierge.nodes.guardrail_node import guardrail_node
from concierge.nodes.rag_node import rag_node
from concierge.nodes.research_node import research_node
from concierge.nodes.synthesis_node import synthesis_node
from concierge.state import ConciergeState, NodeName


NODE_FUNCTIONS = {
    NodeName.DISPATCHER: dispatcher_node,
    NodeName.RAG: rag_node,
    NodeName.RESEARCH: research_node,
    NodeName.BOOKING: booking_node,
    NodeName.GUARDRAIL: guardrail_node,
    NodeName.SYNTHESIS: synthesis_node,
    NodeName.FOLLOWUP: followup_node,
}

GRAPH_NODES = tuple(NODE_FUNCTIONS.keys())

GRAPH_EDGES = (
    (NodeName.DISPATCHER, NodeName.RAG),
    (NodeName.DISPATCHER, NodeName.RESEARCH),
    (NodeName.DISPATCHER, NodeName.BOOKING),
    (NodeName.DISPATCHER, NodeName.GUARDRAIL),
    (NodeName.RAG, NodeName.SYNTHESIS),
    (NodeName.RESEARCH, NodeName.SYNTHESIS),
    (NodeName.RAG, NodeName.GUARDRAIL),
    (NodeName.RESEARCH, NodeName.GUARDRAIL),
    (NodeName.BOOKING, NodeName.GUARDRAIL),
    (NodeName.GUARDRAIL, NodeName.SYNTHESIS),
    (NodeName.SYNTHESIS, "__end__"),
    (NodeName.SYNTHESIS, NodeName.FOLLOWUP),
    (NodeName.FOLLOWUP, "__end__"),
)

CONDITIONAL_EDGES = {
    NodeName.DISPATCHER: {
        "rag": NodeName.RAG,
        "research": NodeName.RESEARCH,
        "booking_stub": NodeName.BOOKING,
        "fallback": NodeName.GUARDRAIL,
    },
    NodeName.SYNTHESIS: {
        "followup": NodeName.FOLLOWUP,
        "__end__": "__end__",
    },
}


def _specialist_for_route(route: str) -> str:
    return CONDITIONAL_EDGES[NodeName.DISPATCHER].get(route, NodeName.GUARDRAIL)


def should_run_followup(state: dict[str, Any]) -> bool:
    # Follow-Up is intentionally conditioned on research route only to demonstrate
    # a meaningful conditional edge and avoid generic suggestions on non-research paths.
    return str(state.get("route") or "").strip() == "research"


def _sequence_for_route(route: str) -> list[str]:
    specialist = _specialist_for_route(route)
    if specialist == NodeName.GUARDRAIL:
        sequence = [NodeName.DISPATCHER, NodeName.GUARDRAIL, NodeName.SYNTHESIS]
    else:
        sequence = [NodeName.DISPATCHER, specialist, NodeName.GUARDRAIL, NodeName.SYNTHESIS]
    if route == "research":
        sequence.append(NodeName.FOLLOWUP)
    return sequence


class _FallbackCompiledGraph:
    def invoke(self, state: ConciergeState) -> ConciergeState:
        result = deepcopy(state)

        executed: list[str] = []
        dispatcher_update = NODE_FUNCTIONS[NodeName.DISPATCHER](result)
        result.update(dispatcher_update)
        executed.append(NodeName.DISPATCHER)

        route = str(result.get("route") or "fallback")
        for node_name in _sequence_for_route(route)[1:]:
            update = NODE_FUNCTIONS[node_name](result)
            result.update(update)
            executed.append(node_name)

        result["_executed_nodes"] = executed
        return result


class _CompiledGraphAdapter:
    def __init__(self, compiled: Any) -> None:
        self._compiled = compiled

    def invoke(self, state: ConciergeState) -> ConciergeState:
        return self._compiled.invoke(deepcopy(state))


def _track_node_execution(node_name: str, fn: Any) -> Any:
    def _wrapped(state: dict[str, Any]) -> dict[str, Any]:
        update = fn(state)
        executed = list(state.get("_executed_nodes") or [])
        executed.append(node_name)

        if isinstance(update, dict):
            merged = dict(update)
            merged["_executed_nodes"] = executed
            return merged

        return {"_executed_nodes": executed}

    return _wrapped


def _build_langgraph_compiled() -> Any:
    from langgraph.graph import END, START, StateGraph  # pragma: no cover - optional dependency

    graph = StateGraph(ConciergeState)
    for node_name, fn in NODE_FUNCTIONS.items():
        graph.add_node(node_name, _track_node_execution(node_name, fn))

    graph.add_edge(START, NodeName.DISPATCHER)
    graph.add_conditional_edges(
        NodeName.DISPATCHER,
        lambda s: str(s.get("route") or "fallback"),
        CONDITIONAL_EDGES[NodeName.DISPATCHER],
    )
    graph.add_edge(NodeName.RAG, NodeName.GUARDRAIL)
    graph.add_edge(NodeName.RESEARCH, NodeName.GUARDRAIL)
    graph.add_edge(NodeName.BOOKING, NodeName.GUARDRAIL)
    graph.add_edge(NodeName.GUARDRAIL, NodeName.SYNTHESIS)
    graph.add_conditional_edges(
        NodeName.SYNTHESIS,
        lambda s: "followup" if should_run_followup(s) else "__end__",
        CONDITIONAL_EDGES[NodeName.SYNTHESIS],
    )
    graph.add_edge(NodeName.FOLLOWUP, END)
    return _CompiledGraphAdapter(graph.compile())


def _is_missing_langgraph(exc: Exception) -> bool:
    if isinstance(exc, ModuleNotFoundError):
        module_name = str(getattr(exc, "name", "") or "")
        return module_name.startswith("langgraph")
    if isinstance(exc, ImportError):
        return "langgraph" in str(exc).lower()
    return False


def build_graph() -> Any:
    try:
        return _build_langgraph_compiled()
    except Exception as exc:
        if _is_missing_langgraph(exc):
            return _FallbackCompiledGraph()
        raise


compiled_graph = build_graph()
# Standard LangGraph Platform export name — langgraph.json points here as "concierge.graph:graph"
graph = compiled_graph
