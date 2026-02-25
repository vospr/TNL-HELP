# Memory

This directory holds the file-based memory layer for the TNL-HELP concierge.

## Structure

- `profiles/{user_id}.json` — User profile: past trips, preferences, cached research. Version-controlled. `alex.json` is pre-seeded as the demo persona.
- `sessions/{session_id}.json` — Per-session conversation history. Runtime-generated, **never committed** (covered by `.gitignore`).

## Demo Reset

To restore the demo to its original state:
```bash
git checkout memory/profiles/alex.json
git clean -f memory/sessions/
```

## Production Swap Point

`memory/profiles/` and `memory/sessions/` are backed by local JSON in MVP.
See `spec/concierge-spec.md §ProductionPath` PP3 for the DynamoDB replacement path.
