# ADR-005: LangGraph Framework Selection
Decision Context: The orchestration layer must support explicit state and conditional edges.
Options Considered:
- Custom orchestration framework.
- LangGraph 1.0 StateGraph with typed state contracts.
Rationale: LangGraph provides native graph composition, clear node boundaries, and maintainable control flow for multi-agent routing.
Status: Accepted.
