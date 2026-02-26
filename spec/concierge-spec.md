# TNL-HELP Concierge Spec - 2026-02-25 written before implementation
<!-- v1.0 | 2026-02-25 | Written before implementation -->
<!-- Author: Andrey | Status: Authoritative — code must not contradict this file silently -->

This spec is the contract. All agent implementations, graph edges, and node return types
are derived from the schemas and conditions defined here. If code and spec conflict, fix the code.

---

## §State — ConciergeState

```python
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class ConciergeState(TypedDict):
    # Session identity — set once at main.py startup, never mutated by nodes
    user_id: str
    session_id: str
    turn_id: int                                   # incremented by Dispatcher at turn start

    # Conversation — Dispatcher-owned; full session history
    # MUST use add_messages reducer — direct assignment overwrites on every node invocation
    conversation_history: Annotated[list, add_messages]
    current_input: str                             # raw user input this turn
    current_response: str                          # final assembled response this turn

    # Routing — set by Dispatcher, reset to None by reset_turn_state() each turn
    intent: str | None
    confidence: float | None
    route: Literal["rag", "research", "booking_stub", "fallback"] | None

    # Sub-agent outputs — reset to None by Dispatcher at turn start; never carry across turns
    rag_results: list | None
    research_results: list | None
    source_attribution: list[str]

    # Memory — loaded once per session by MemoryService; not reset between turns
    memory_profile: dict | None

    # Governance
    guardrail_passed: bool
    proactive_suggestion: str | None
    clarification_needed: bool
    clarification_question: str | None
    human_handoff: bool

    # Error propagation — set by node on exception; reset each turn
    error: str | None
```

### Per-Node Update TypedDicts

Nodes return these — never `dict` or full `ConciergeState`.
`total=False` makes all fields optional in update TypedDicts.

```python
class DispatcherUpdate(TypedDict, total=False):
    intent: str | None
    confidence: float | None
    route: str | None
    turn_id: int
    conversation_history: list
    error: str | None
    human_handoff: bool

class RAGUpdate(TypedDict, total=False):
    rag_results: list | None
    source_attribution: list[str]
    error: str | None
    human_handoff: bool

class ResearchUpdate(TypedDict, total=False):
    research_results: list | None
    source_attribution: list[str]
    error: str | None
    human_handoff: bool

class SynthesisUpdate(TypedDict, total=False):
    current_response: str
    error: str | None
    human_handoff: bool

class GuardrailUpdate(TypedDict, total=False):
    guardrail_passed: bool
    clarification_needed: bool
    clarification_question: str | None
    human_handoff: bool
    error: str | None

class FollowUpUpdate(TypedDict, total=False):
    proactive_suggestion: str | None
    current_response: str
    error: str | None

class TurnResetUpdate(TypedDict):
    # ALL fields required — not total=False. mypy enforces completeness.
    intent: None
    confidence: None
    route: None
    rag_results: None
    research_results: None
    source_attribution: list
    proactive_suggestion: None
    clarification_needed: bool
    clarification_question: None
    human_handoff: bool
    error: None
```

### NodeName Constants

```python
class NodeName:
    DISPATCHER  = "dispatcher"
    RAG         = "rag"
    RESEARCH    = "research"
    SYNTHESIS   = "synthesis"
    GUARDRAIL   = "guardrail"
    FOLLOWUP    = "followup"
    BOOKING     = "booking_stub"
```

---

## §AgentPolicy

```python
from pydantic import BaseModel, Field, model_validator
from typing import Literal
import os
from pathlib import Path
import yaml

class AgentPolicy(BaseModel):
    agent_name: str
    model: str                    # e.g. "claude-opus-4-6"
    prompt_version: str           # e.g. "v1" — must match prompts/{agent}/v1.yaml
    max_tokens: int               # enforced before LLM call
    confidence_threshold: float = Field(ge=0.0, le=1.0)
    max_clarifications: int = 3   # consecutive below-threshold turns before human handoff
    allowed_tools: list[str]      # declarative allowlist — runtime enforcement post-MVP
    prompt_sections: list[str]    # must contain all 5: role, context, constraints, output_format, examples

    @property
    def effective_model(self) -> str:
        """Returns haiku override in FAST_MODE; else declared model."""
        if os.environ.get("FAST_MODE") == "1":
            return "claude-haiku-4-5-20251001"
        return self.model

    @classmethod
    def from_yaml(cls, agent_name: str, prompts_root: Path | None = None) -> "AgentPolicy":
        root = prompts_root or Path(__file__).parent.parent / "prompts"
        path = root / agent_name / "policy.yaml"
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    @model_validator(mode="after")
    def check_prompt_sections(self) -> "AgentPolicy":
        required = {"role", "context", "constraints", "output_format", "examples"}
        missing = required - set(self.prompt_sections)
        if missing:
            raise ValueError(f"Missing required prompt sections: {missing}")
        return self
```

**Multi-model assignment:**

| Agent | Model | Rationale |
|-------|-------|-----------|
| Dispatcher | `claude-opus-4-6` | Complex intent classification; full conversation context |
| Research Agent | `claude-sonnet-4-6` | Multi-step reasoning over web results |
| Response Synthesis | `claude-sonnet-4-6` | Blended source attribution requires coherent reasoning |
| RAG Agent | `claude-haiku-4-5-20251001` | Fast retrieval formatting from structured KB |
| Guardrail | `claude-haiku-4-5-20251001` | Pattern-match + confidence threshold only |
| Follow-Up | `claude-haiku-4-5-20251001` | Template-driven proactive suggestion |

*Alternatives considered: single model for all agents (rejected — cost signal and routing quality
signal both degraded); Sonnet for Dispatcher (rejected — Opus routing quality is the
architectural statement for this role application).*

---

## §NodeContracts

### Dispatcher

**Inputs read:** `current_input`, `conversation_history`, `memory_profile`, `turn_id`
**Outputs written:** `DispatcherUpdate` — `intent`, `confidence`, `route`, `turn_id` (incremented),
`conversation_history` (appended)
**Side effects:** calls `reset_turn_state()` before classification; emits
`[DISPATCHER] intent=X confidence=0.NN → route` trace
**On error:** returns `DispatcherUpdate(error="[DISPATCHER] ...", human_handoff=True)`

### RAG Agent

**Inputs read:** `current_input`, `intent`
**Outputs written:** `RAGUpdate` — `rag_results`, `source_attribution`
**Contract:** reads `MockKnowledgeBase` (see §ProductionPath); returns up to 3 matching entries;
returns empty list (not `None`) on no-match; never raises on empty KB
**On error:** returns `RAGUpdate(rag_results=[], error="[RAG] ...")`

### Research Agent

**Inputs read:** `current_input`, `intent`
**Outputs written:** `ResearchUpdate` — `research_results`, `source_attribution`
**Contract:** DuckDuckGo `DDGS().text(query, max_results=5)`; on DDG failure returns
`ResearchUpdate(research_results=[], source_attribution=["[WEB SEARCH UNAVAILABLE]"])`; never raises
**On error:** returns `ResearchUpdate(research_results=[], error="[RESEARCH] ...")`

### Response Synthesis

**Inputs read:** `rag_results`, `research_results`, `source_attribution`, `memory_profile`,
`conversation_history` (for context)
**Outputs written:** `SynthesisUpdate` — `current_response`
**Contract:** filters `role == "system_summary"` from history before LLM call;
inline attribution format: `[RAG]` and `[Web]` labels in response text
**On error:** returns `SynthesisUpdate(current_response="[ERROR] Unable to synthesize response — please try again.")`

### Guardrail

**Inputs read:** `current_response`, `intent`, `confidence`, `current_input`
**Outputs written:** `GuardrailUpdate` — `guardrail_passed`, `clarification_needed`,
`clarification_question`, `human_handoff`
**Contract:**
- `confidence ≥ dispatcher.confidence_threshold`: pass through (`guardrail_passed=True`)
- `confidence < threshold, clarification_count < max_clarifications`: one clarifying question
- `confidence < threshold, clarification_count ≥ max_clarifications`: human handoff
- Out-of-domain intent: human handoff with deflection message
**On error:** returns `GuardrailUpdate(guardrail_passed=False, human_handoff=True)`

### BookingAgent Stub

**Inputs read:** `intent`, `current_input`
**Outputs written:** `SynthesisUpdate` — `current_response` (stub message only)
**Contract:** returns `BookingAgentResponse` Pydantic model; never fakes a booking
**Stub message:** `"Booking is not available in this version. A human travel specialist can assist — shall I connect you?"`

```python
class BookingAgentResponse(BaseModel):
    status: Literal["stub"]
    message: str
    integration_point: str  # "BookingAPI v2 — requires BOOKING_API_KEY env var"
    required_env_vars: list[str]  # ["BOOKING_API_KEY", "BOOKING_API_URL"]
    # AIDEV-TODO: PRODUCTION SWAP POINT — replace stub body with real BookingAPI call
```

### Follow-Up Node

**Inputs read:** `current_response`, `confidence`, `clarification_needed`, `turn_id`,
`memory_profile`
**Outputs written:** `FollowUpUpdate` — `proactive_suggestion`, `current_response` (appended)
**Contract:** see §FollowUpCondition below
**On error:** returns `FollowUpUpdate(proactive_suggestion=None)` — follow-up is non-critical; fail silently

---

## §FollowUpCondition

The Follow-Up node demonstrates **conditional graph edges** — the primary LangGraph capability
distinguishing this system from a linear chain. This section defines when synthesis routes to
the Follow-Up node vs. when it skips directly to output.

### Edge Condition

Follow-Up node is invoked **only when ALL of the following are true:**

```python
def should_run_followup(state: ConciergeState) -> str:
    """
    Conditional edge: synthesis → followup | END.

    AIDEV-NOTE: This conditional edge is the primary LangGraph demonstration.
    A linear chain cannot implement per-turn proactive suggestions based on
    state — this requires a stateful graph with conditional routing.
    """
    if (
        state.get("confidence", 0.0) >= 0.8          # high-confidence routing turn
        and not state.get("clarification_needed")    # no pending clarification
        and (state.get("turn_id") or 0) > 1          # not the greeting turn
        and not state.get("human_handoff")           # no error/handoff path
        and state.get("route") in ("rag", "research") # knowledge-domain turns only
    ):
        return NodeName.FOLLOWUP
    return "__end__"
```

### Follow-Up Suggestion Logic

When invoked, the Follow-Up node generates one proactive suggestion using
`claude-haiku-4-5-20251001` based on:
- `memory_profile.past_trips` — trips that relate to the current intent
- `current_response` context — suggest depth, not breadth
- Rule: ONE suggestion only — never a list

**Output format:** appended to `current_response` as a new paragraph:
```
---
💡 Based on your past trip to Bali, you might also want to ask about...
```

### Alternatives Considered

- **Always run Follow-Up** (rejected — spam after every turn degrades demo experience)
- **Run Follow-Up only on Research turns** (rejected — too narrow; misses RAG turns with
  high confidence where proactive suggestion adds value)
- **Turn_id > 2** (rejected — the second turn is often the most relevant for a follow-up;
  turn_id > 1 correctly skips only the session-opening greeting turn)

---

## §DemoScript

The demo script defines the expected routing path for each query — not just the expected response.
Stage 1 queries are pre-validated against `routing_rules.yaml` scores before Day 5.

### Demo Persona

User: `python main.py --user alex`
Session opens with proactive greeting (no user input required):
```
[MEMORY] Loaded profile: alex — 2 past trips (Bali, Tokyo), 1 cached research session
Welcome back, Alex. Based on your March trip to Bali, you might be interested in upcoming
deals to Southeast Asia — I can look into current options if you'd like.
```

### Five Demo Queries

See `memory/profiles/alex.json` `_demo_notes.routing_showcase_queries` for the exact queries
and their expected routes. Summary:

| # | Query Type | Expected Stage | Expected Route | Demonstrates |
|---|-----------|---------------|----------------|-------------|
| Q1 | Wyndham property by name | Stage 1 keyword hit | `rag` | Deterministic KB routing |
| Q2 | Specific KB destination | Stage 1 keyword hit | `rag` | High-confidence Stage 1 |
| Q3 | Ambiguous destination intent | Stage 2 LLM escalation | `research` | Hybrid dispatcher |
| Q4 | Current trends / "right now" | Stage 2 LLM escalation | `research` | LLM classification |
| Q5 | Out-of-domain (crypto) | Guardrail catch | `fallback` | Guardrail boundary |

**Stage 1 validation test** (no LLM required):
```python
# tests/unit/test_routing_rules.py
def test_q1_routes_to_rag_at_stage1():
    rules = load_routing_rules("config/routing_rules.yaml")
    confidence = rules.score("What Wyndham properties are available in Bali?")
    assert confidence >= rules["escalation_threshold"]
    assert rules.intent("What Wyndham properties are available in Bali?") == "rag"
```

---

## §ProductionPath

Every production swap point is indexed here. All swap-point comments in source use format:
`# AIDEV-TODO: PRODUCTION SWAP POINT — replace X with Y (requires ENV_VAR)`

| # | File | Current (MVP) | Production Replacement | Required Env Var |
|---|------|--------------|----------------------|-----------------|
| PP1 | `agents/rag_agent.py` | `MockKnowledgeBase` (JSON file) | `BedrockKnowledgeBase(kb_id=...)` | `BEDROCK_KB_ID` |
| PP2 | `agents/booking_agent.py` | `BookingAgentStub` | `BookingAPIClient(url=..., key=...)` | `BOOKING_API_KEY`, `BOOKING_API_URL` |
| PP3 | `agents/memory_service.py` | File-based JSON in `memory/` | DynamoDB or RDS with TTL | `MEMORY_TABLE_NAME`, `AWS_REGION` |
| PP4 | `token_budget.py` | `TokenBudgetManager` stub | Full summarization via `claude-haiku-4-5-20251001` | None (uses existing `ANTHROPIC_API_KEY`) |
| PP5 | `trace.py` | stdout `_trace_writer` | LangSmith `RunTree` client | `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT` |

---

## §TokenBudgetManager

```python
# AIDEV-TODO: Activation threshold: summarize when turn_count > [TBD].
# See architecture.md §G1 — threshold deliberately deferred.
# Candidate value: summarize when len(conversation_history) > 20 messages.
# Production signal only — never fires in a standard 5–8 turn demo.
class TokenBudgetManager:
    def check_and_summarize(self, history: list) -> list:
        """
        [TBD] Activation threshold not yet defined.
        When implemented: summarize oldest turns into role='system_summary' message
        using claude-haiku-4-5-20251001. Response Synthesis filters system_summary
        before generation.
        """
        return history  # stub: pass-through
```

*Alternatives considered: fixed 10-turn window (rejected — arbitrary; production threshold
depends on actual token counts, not turn counts); always summarize (rejected — kills hiring
manager demo readability).*

---

## §KnowledgeBase

Mock KB seeded with 6 travel destinations. RAG agent matches against these.
Format: `list[KBEntry]` loaded from `data/knowledge_base.json` at RAG agent `__init__`.

```python
class KBEntry(BaseModel):
    id: str
    destination: str
    property_name: str | None    # Wyndham property name if applicable
    description: str
    highlights: list[str]
    best_season: str
    tags: list[str]              # used for keyword matching
```

Seeded destinations: Bali (Wyndham), Phuket (Wyndham), Tokyo, Koh Samui, Sydney, Maldives.

[TBD] — exact KB content and tag vocabulary defined on Day 2 when RAG agent is implemented.

---

## §MemorySchema

```python
class UserProfile(TypedDict):
    user_id: str
    preferred_name: str
    past_trips: list[dict]        # {destination, dates, property, notes}
    preferences: dict             # {travel_style, room_type, activities, diet}
    cached_research: dict | None  # last research session context
    _demo_notes: dict[str, str]   # demo metadata — not used by agents at runtime
```

```python
class SessionRecord(TypedDict):
    session_id: str
    user_id: str
    started_at: str               # ISO 8601
    turns: list[dict]             # {turn_id, input, response, route, intent, confidence}
    last_intent: str | None
    last_route: str | None
```

Session files written to `memory/sessions/{session_id}.json` at session close.
Profile updated to `memory/profiles/{user_id}.json` on meaningful preference signals.
Both paths resolved relative to `Path(__file__).parent` — never `os.getcwd()`.

---

## §ValidateConfig

10 checks executed by `validate_config.run_checks()` before graph import.
Ordered cheapest → most expensive. Early-exit on first failure with count of remaining skipped.

1. Python version ≥ 3.11 — `sys.version_info`
2. `ANTHROPIC_API_KEY` present in environment
3. All `prompts/*/policy.yaml` parse as valid `AgentPolicy` (incl. 5-section check)
4. All declared model names in allow-list
5. Dispatcher `max_tokens` ≤ 256 (routing output only)
6. `guardrail.confidence_threshold < dispatcher.confidence_threshold` (strict less-than)
7. `config/routing_rules.yaml` parses with `escalation_threshold` as float in [0.0, 1.0]
8. `langgraph.__version__` starts with `"1."` — prints tested version in error
9. `memory/profiles/alex.json` parses as valid `UserProfile`
10. Anthropic API reachable — 5-second timeout; on timeout: `"API probe timed out — check network. Key format appears valid."`

`SKIP_API_PROBE=1` env var skips check 10 for offline development.

---

## §GraphTopology

```
START
  └─► dispatcher_node
        ├─► (route == "rag")           → rag_node → synthesis_node
        ├─► (route == "research")      → research_node → synthesis_node
        ├─► (route == "booking_stub")  → booking_node → END
        ├─► (route == "fallback")      → guardrail_node → END
        └─► (human_handoff == True)    → END

  synthesis_node
        └─► should_run_followup()
              ├─► (conditions met)     → followup_node → END
              └─► (conditions not met) → END
```

Edge function locations:
- Dispatcher conditional edge: `edges.route_from_dispatcher(state)` in `src/concierge/edges.py`
- Synthesis conditional edge: `edges.should_run_followup(state)` in `src/concierge/edges.py`

All edge functions read state fields only — no threshold re-computation, no business logic.
