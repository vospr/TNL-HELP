# ADR-001: Context Window Ownership
Decision Context: Conversation history must remain coherent across routing decisions.
Options Considered:
- Shared mutable history across all agents.
- Dispatcher-owned history with scoped inputs to specialists.
Rationale: Dispatcher ownership centralizes turn state, avoids stale cross-agent bleed, and keeps routing deterministic.
Status: Accepted.
