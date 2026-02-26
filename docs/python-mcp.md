# Python + MCP + Agentic AI Quick Memory

## Python core

### Function
Reusable logic for one task.

```python
def classify_intent(text: str) -> str:
    if "refund" in text.lower():
        return "billing_refund"
    return "general_support"
```

### Class
Blueprint that keeps state + behavior together.

```python
class Session:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.history = []

    def add_turn(self, role: str, text: str) -> None:
        self.history.append({"role": role, "text": text})
```

Use function for isolated actions. Use class when state must persist.

---

## Asyncio and multiflow

### When to use `asyncio`
Use for I/O-bound work:
- LLM calls
- MCP tool/API calls
- DB/network waits
- streaming responses

```python
import asyncio

async def get_profile(user_id: str) -> dict:
    await asyncio.sleep(0.1)
    return {"tier": "gold"}

async def get_orders(user_id: str) -> list[dict]:
    await asyncio.sleep(0.1)
    return [{"order_id": "A123"}]

async def build_context(user_id: str) -> dict:
    profile, orders = await asyncio.gather(get_profile(user_id), get_orders(user_id))
    return {"profile": profile, "orders": orders}
```

### Multiflow (parallel paths)
Use when independent steps can run in parallel (retrieval + policy check + profile load), especially in LangGraph node graphs.

```python
async def retrieve_docs(q: str) -> list[str]:
    await asyncio.sleep(0.05)
    return ["doc1", "doc2"]

async def policy_ok(q: str) -> bool:
    await asyncio.sleep(0.02)
    return "forbidden" not in q.lower()

async def answer(q: str) -> str:
    docs, allowed = await asyncio.gather(retrieve_docs(q), policy_ok(q))
    return "Blocked." if not allowed else f"Grounded with {len(docs)} docs"
```

---

## Refactoring

Refactoring = improving code structure without changing behavior.

Why:
- cleaner code
- easier tests
- safer changes

---

## Deterministic flow

Deterministic flow = fixed rules, predictable output.
Use for compliance, approvals, and high-risk decisions.

```python
def refund_decision(amount: float, has_receipt: bool) -> str:
    if not has_receipt:
        return "DENY"
    if amount <= 100:
        return "AUTO_APPROVE"
    return "MANUAL_REVIEW"
```

---

## MCP: why we need it

MCP gives a standard interface between agents and tools/data.

Benefits:
- reusable integrations
- faster onboarding
- clearer security boundaries

Client vs server:
- MCP client: discovers/calls tools from app/agent side
- MCP server: exposes tools and executes operations safely

---

## How we define what MCP agents can call

Use:
- allowlist per role
- strict input schema
- deny by default
- logs/audit

```python
ALLOWED_TOOLS = {
    "support_agent": {"search_kb", "get_order", "create_ticket"},
    "finance_agent": {"get_invoice"},
}

def can_call(role: str, tool: str) -> bool:
    return tool in ALLOWED_TOOLS.get(role, set())
```

---

## What if agents do not use RAG

Without RAG:
- relies on model memory + prompt only
- higher hallucination risk
- weaker citations and traceability
- harder to stay current

With RAG:
- grounded on company data
- better factual reliability
- citation/audit support

---

## Explain LLM to a child

An LLM is like a very smart sentence finisher.
It has read lots of text and guesses the next best words.
Sometimes it guesses wrong, so we check important answers.

---

## Mini end-to-end example (deterministic + async + tool policy)

```python
import asyncio

ALLOWED = {"assistant": {"search_kb"}}

def guardrail(intent: str) -> bool:
    return intent not in {"illegal_advice", "self_harm"}

async def search_kb(query: str) -> list[str]:
    await asyncio.sleep(0.05)
    return ["policy_doc", "faq_doc"]

async def run_agent(user_text: str) -> str:
    intent = "illegal_advice" if "hack" in user_text.lower() else "general"
    if not guardrail(intent):
        return "I cannot help with that."
    if "search_kb" not in ALLOWED["assistant"]:
        return "Tool not allowed."
    docs = await search_kb(user_text)
    return f"Answer grounded by {docs}"

if __name__ == "__main__":
    print(asyncio.run(run_agent("How do I reset my password?")))
```

