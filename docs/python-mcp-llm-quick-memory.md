# Python + MCP + Agentic AI Quick Memory

## 1) Python basics

### What is a function?
A function is a reusable block of logic.

```python
def classify_intent(text: str) -> str:
    if "refund" in text.lower():
        return "billing_refund"
    return "general_support"
```

### What is a class?
A class is a blueprint that groups data and behavior together.

```python
class ConversationSession:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.turns = []

    def add_turn(self, role: str, text: str) -> None:
        self.turns.append({"role": role, "text": text})
```

When to use:
- `function`: one clear task
- `class`: state + multiple related methods

---

## 2) Asyncio: when and why

Use `asyncio` for I/O-bound work:
- LLM API calls
- MCP tool calls
- database/network requests
- streaming operations

```python
import asyncio


async def fetch_profile(user_id: str) -> dict:
    await asyncio.sleep(0.1)  # simulate network
    return {"tier": "gold"}


async def fetch_orders(user_id: str) -> list[dict]:
    await asyncio.sleep(0.1)
    return [{"order_id": "A123"}]


async def load_context(user_id: str) -> dict:
    profile, orders = await asyncio.gather(
        fetch_profile(user_id),
        fetch_orders(user_id),
    )
    return {"profile": profile, "orders": orders}
```

Do not use `asyncio` for CPU-heavy tasks (use workers/processes instead).

---

## 3) Multiflow (parallel paths): when and why

Use multiflow when independent steps can run in parallel to reduce latency.

Business-case example (concierge):
- retrieve docs
- check compliance policy
- load customer profile
- then combine results

```python
import asyncio


async def retrieve_docs(query: str) -> list[str]:
    await asyncio.sleep(0.1)
    return ["doc1", "doc2"]


async def check_policy(query: str) -> bool:
    await asyncio.sleep(0.05)
    return True


async def build_response(query: str) -> str:
    docs_task = retrieve_docs(query)
    policy_task = check_policy(query)
    docs, policy_ok = await asyncio.gather(docs_task, policy_task)
    if not policy_ok:
        return "I cannot answer that request."
    return f"Answer based on {len(docs)} docs."
```

LangGraph relation:
- this pattern maps to parallel nodes in a graph, then a merge node.

---

## 4) Refactoring

Refactoring = improving code structure without changing behavior.

Goals:
- readability
- testability
- maintainability
- safer change velocity

Example:
- move tool-call logic out of one huge function into smaller functions/classes.

---

## 5) Deterministic flow

Deterministic flow means predictable steps and outputs for the same input/rules.

Use for:
- compliance-heavy logic
- billing/refund approvals
- regulated workflows

```python
def refund_decision(amount: float, has_receipt: bool) -> str:
    if not has_receipt:
        return "DENY"
    if amount <= 100:
        return "AUTO_APPROVE"
    return "MANUAL_REVIEW"
```

In agent systems:
- keep risky actions deterministic
- use LLMs for language/reasoning around the deterministic core

---

## 6) MCP: why we need it

MCP provides a standard contract between agents and tools/data.

Benefits:
- reusable integrations
- faster onboarding of tools
- clearer governance/security boundaries
- lower custom glue code

Client vs server:
- MCP client: discovers and calls tools
- MCP server: exposes tools and executes operations safely

---

## 7) How to define what MCP agents can call

Use:
- explicit allowlist
- strict JSON schemas
- role-based permissions
- audit logs
- deny-by-default policy

```python
ALLOWED_TOOLS = {
    "agent_support": {"search_kb", "get_order", "create_ticket"},
    "agent_finance": {"search_kb", "get_invoice"},
}


def is_tool_allowed(agent_role: str, tool_name: str) -> bool:
    return tool_name in ALLOWED_TOOLS.get(agent_role, set())


def call_tool(agent_role: str, tool_name: str, args: dict) -> dict:
    if not is_tool_allowed(agent_role, tool_name):
        raise PermissionError(f"{agent_role} cannot call {tool_name}")
    # validate args against schema here
    return {"status": "ok", "tool": tool_name, "result": "..."}
```

---

## 8) What if agents do not use RAG?

Without RAG:
- answers rely on model pretraining + prompt context only
- higher hallucination risk for domain facts
- weaker citation/traceability
- harder to stay up-to-date

With RAG:
- grounded on current enterprise data
- better factual reliability
- easier source citation and audit

---

## 9) Child-friendly LLM explanation

An LLM is like a very smart sentence finisher.
It has read lots of text and guesses the next best words.
It can be very helpful, but sometimes it guesses wrong, so we check important answers.

---

## 10) Tiny end-to-end mini example (deterministic + async + tool policy)

```python
import asyncio

ALLOWED = {"assistant": {"search_kb"}}


async def search_kb(query: str) -> list[str]:
    await asyncio.sleep(0.05)
    return ["policy_doc", "faq_doc"]


def deterministic_guardrail(intent: str) -> bool:
    blocked = {"illegal_advice", "self_harm"}
    return intent not in blocked


async def answer(user_text: str) -> str:
    intent = "general" if "hack" not in user_text.lower() else "illegal_advice"
    if not deterministic_guardrail(intent):
        return "I cannot help with that."

    if "search_kb" not in ALLOWED["assistant"]:
        return "Tool not allowed."

    docs = await search_kb(user_text)
    return f"Grounded answer using {', '.join(docs)}"


if __name__ == "__main__":
    print(asyncio.run(answer("How to reset my account password?")))
```

