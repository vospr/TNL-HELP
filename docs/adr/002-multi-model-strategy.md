# ADR-002: Multi-Model Strategy
Decision Context: Different nodes have different latency and quality requirements.
Options Considered:
- Single model for all agents.
- Per-agent model policy with optional FAST_MODE override.
Rationale: Per-agent model selection balances cost, speed, and output quality while preserving one fallback path for development.
Status: Accepted.
