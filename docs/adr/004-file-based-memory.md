# ADR-004: File-Based Memory
Decision Context: The demo needs inspectable memory without cloud dependencies.
Options Considered:
- Immediate database-backed memory service.
- Local JSON profiles and session files with clear swap points.
Rationale: File-based memory makes behavior transparent for reviewers and can be upgraded later without changing contracts.
Status: Accepted.
