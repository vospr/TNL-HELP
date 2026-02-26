CH# LangChain / LangGraph Python — AI Agent Coding Standards
# Based on: LangChain team's own research (Sept 2025), official LangGraph patterns,
# community best practices from awesome-LangGraph, GPT-Researcher, and LangGraph templates
#
# Compatible with: Claude Code, Cursor (.cursorrules), GitHub Copilot, Gemini CLI, Windsurf
# Save as: CLAUDE.md | AGENTS.md | .cursorrules

---

## 0. LangGraph/LangChain Documentation Access (MCP)

Before writing any LangGraph or LangChain code, use the mcpdoc MCP server for up-to-date docs.
LangChain APIs change frequently — never rely on training data alone.

**mcpdoc setup (add to mcp.json in Cursor / Claude Code):**
```json
{
  "mcpServers": {
    "langgraph-docs-mcp": {
      "command": "uvx",
      "args": [
        "--from", "mcpdoc", "mcpdoc",
        "--urls",
        "LangGraph:https://langchain-ai.github.io/langgraph/llms.txt LangChain:https://python.langchain.com/llms.txt",
        "--transport", "stdio"
      ]
    }
  }
}
```

**Rule:** For ANY question about LangGraph or LangChain APIs, call `list_doc_sources` then `fetch_docs`
on the relevant URL before generating code. Outdated patterns cause runtime errors.

---

## 1. Environment & Tooling

```bash
# Package manager: uv (mandatory)
uv venv .venv
source .venv/bin/activate

# Core dependencies
uv add langchain langchain-core langchain-community
uv add langgraph langgraph-checkpoint-sqlite   # or checkpoint-postgres for production
uv add langchain-anthropic                      # or langchain-openai
uv add pydantic python-dotenv langsmith

# Dev tools
uv add --dev pytest pytest-asyncio ruff mypy pre-commit
```

**Required tools:**
| Tool | Purpose |
|------|---------|
| `ruff` | Format + lint (replaces black, flake8, isort) |
| `mypy` | Type checking — critical for catching LangGraph state bugs |
| `pytest-asyncio` | Testing async nodes and graphs |
| `langsmith` | Tracing and debugging — enable from day one |

```bash
# Run all checks before commit
ruff format . && ruff check . && mypy src/ && pytest tests/
```

---

## 2. Project Structure

```
project-root/
├── src/
│   └── agent_name/
│       ├── __init__.py
│       ├── graph.py          # StateGraph definition — the entry point
│       ├── state.py          # State schema (TypedDict or Pydantic)
│       ├── nodes.py          # Individual node implementations
│       ├── edges.py          # Conditional edge logic / routing functions
│       ├── tools.py          # Tool definitions (@tool decorated functions)
│       ├── prompts.py        # System prompts and prompt templates
│       └── config.py         # RunnableConfig, configurable fields
├── tests/
│   ├── unit/
│   │   ├── test_nodes.py     # Test each node independently
│   │   └── test_tools.py
│   ├── integration/
│   │   └── test_graph.py     # Test full graph execution
│   └── conftest.py
├── notebooks/                # Exploration only, not production
├── langgraph.json            # LangGraph Platform deployment config
├── pyproject.toml
├── .env.example              # Template (never commit .env)
└── CLAUDE.md                 # This file
```

**Key rule:** One file per concern. `graph.py` should only assemble the graph — no business logic.
Node logic goes in `nodes.py`, routing goes in `edges.py`.

---

## 3. State Design (Most Critical Pattern)

State is the backbone of LangGraph. Get this right first.

```python
# state.py

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

# PREFERRED: Use MessagesState for chat agents (inherits messages with add_messages reducer)
from langgraph.graph import MessagesState

class AgentState(MessagesState):
    """Extended state for agents that need more than messages."""
    # Add custom fields on top of MessagesState
    user_id: str
    retrieved_docs: list[str]
    iteration_count: int

# For non-chat workflows: define your own TypedDict
from typing import TypedDict

class SQLAgentState(TypedDict):
    query: str                                     # User's natural language query
    sql: str                                       # Generated SQL
    sql_result: str                                # Query execution result
    messages: Annotated[list, add_messages]        # ALWAYS use add_messages reducer
    error: str | None                              # Track errors across nodes
```

**State rules:**
- ALWAYS use `Annotated[list, add_messages]` for messages — never plain `list`
- `add_messages` is a reducer that appends; without it, messages get overwritten
- Use `MessagesState` as base for most agents — it handles message list correctly
- Keep state flat — avoid deeply nested dicts
- Every field should have a clear type annotation
- Add `error: str | None = None` to track failures across nodes

### Common Mistake — Wrong State Update Pattern

```python
# ❌ WRONG — overwrites messages list instead of appending
def my_node(state: AgentState) -> AgentState:
    return {"messages": [new_message]}  # loses history!

# ✅ CORRECT — add_messages reducer handles appending
def my_node(state: AgentState) -> dict:
    return {"messages": [new_message]}  # add_messages merges, not overwrites
```

---

## 4. Node Design

```python
# nodes.py

from langchain_anthropic import ChatAnthropic
from langgraph.graph import MessagesState

llm = ChatAnthropic(model="claude-sonnet-4-5", temperature=0)

# SIMPLE NODE — pure function, takes state, returns partial state update
def generate_sql(state: SQLAgentState) -> dict:
    """Generate SQL from natural language query."""
    prompt = f"Convert to SQL: {state['query']}"
    response = llm.invoke([{"role": "user", "content": prompt}])
    return {"sql": response.content}

# TOOL-CALLING NODE — bind tools to LLM
from langchain_core.tools import tool

@tool
def search_database(query: str) -> str:
    """Execute a SQL query and return results."""
    # implementation
    return results

tools = [search_database]
llm_with_tools = llm.bind_tools(tools)

def call_llm(state: MessagesState) -> dict:
    """Main reasoning node."""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}
```

**Node rules:**
- Nodes are pure functions: `(state) -> dict` — return only the fields that change
- Never mutate state directly — always return a new dict
- One node = one responsibility (generate SQL, execute SQL, format response are 3 separate nodes)
- Nodes should be independently testable without the full graph

---

## 5. Graph Assembly

```python
# graph.py

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from .state import MessagesState
from .nodes import call_llm, tools

def build_graph() -> StateGraph:
    """Assemble and compile the agent graph."""
    
    graph = StateGraph(MessagesState)
    
    # Add nodes
    graph.add_node("agent", call_llm)
    graph.add_node("tools", ToolNode(tools))
    
    # Add edges
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        tools_condition,         # built-in: routes to "tools" or END based on tool calls
    )
    graph.add_edge("tools", "agent")  # loop back after tool execution
    
    return graph.compile()

# Export compiled graph (required for LangGraph Platform)
graph = build_graph()
```

**Graph rules:**
- Compile once at module level: `graph = build_graph()`
- Use `ToolNode` from `langgraph.prebuilt` for tool execution — don't write your own
- Use `tools_condition` from `langgraph.prebuilt` for routing — don't write your own
- Search the existing codebase for patterns before creating new nodes or edges

---

## 6. Common Patterns (Use These, Don't Reinvent)

### ReAct Agent (tool-calling loop)
```python
from langgraph.prebuilt import create_react_agent

# Most agents should start here — only drop down to manual graph when needed
graph = create_react_agent(
    model=llm,
    tools=tools,
    state_modifier="You are a helpful assistant.",  # system prompt
)
```

### Supervisor (multi-agent coordination)
```python
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent

researcher = create_react_agent(llm, [search_tool], name="researcher")
writer = create_react_agent(llm, [write_tool], name="writer")

# Supervisor routes between sub-agents
supervisor = create_supervisor(
    llm,
    agents=[researcher, writer],
    prompt="Coordinate research and writing tasks.",
)
graph = supervisor.compile()
```

### Checkpointing (persistence across turns)
```python
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver  # for production

# Development: SQLite
async with AsyncSqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
    graph = build_graph().compile(checkpointer=checkpointer)
    
    # Thread ID enables multi-turn conversations
    config = {"configurable": {"thread_id": "user_session_123"}}
    result = await graph.ainvoke({"messages": [user_msg]}, config=config)
```

### Human-in-the-Loop
```python
from langgraph.types import interrupt, Command

def review_node(state: AgentState) -> Command:
    """Pause execution for human review."""
    human_feedback = interrupt({
        "question": "Approve this SQL query?",
        "sql": state["sql"],
    })
    
    if human_feedback["approved"]:
        return Command(goto="execute_sql")
    else:
        return Command(goto="regenerate_sql", update={"feedback": human_feedback["notes"]})
```

**CRITICAL — Common `interrupt()` Mistake:**
```python
# ❌ WRONG — interrupt() must be called from within a node, not in edges
def route(state):
    if needs_review(state):
        interrupt("review needed")  # This crashes!

# ✅ CORRECT — interrupt() only inside node functions
def review_node(state):
    result = interrupt({"data": state["output"]})  # Pauses graph here
    return {"approved": result}
```

---

## 7. Streaming

LangGraph supports streaming. Always implement streaming for user-facing agents.

```python
# Stream events (recommended for UI integration)
async for event in graph.astream_events(
    {"messages": [user_message]},
    config=config,
    version="v2",
):
    if event["event"] == "on_chat_model_stream":
        chunk = event["data"]["chunk"]
        print(chunk.content, end="", flush=True)

# Stream state updates (good for debugging)
async for state_update in graph.astream(
    {"messages": [user_message]},
    config=config,
):
    print(state_update)  # dict of state changes per node
```

**Streaming + Streamlit (common asyncio pitfall):**
```python
# ❌ WRONG — Streamlit runs its own event loop, asyncio.run() conflicts
async def run_agent():
    result = await graph.ainvoke(...)

asyncio.run(run_agent())  # Crashes in Streamlit!

# ✅ CORRECT — use nest_asyncio or run sync version
import nest_asyncio
nest_asyncio.apply()

# Or use the sync graph interface in Streamlit:
result = graph.invoke({"messages": [user_message]}, config=config)
```

---

## 8. Type Hints & Structured Output

```python
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage

# ALWAYS use Pydantic for structured LLM output
class SQLOutput(BaseModel):
    """Structured output for SQL generation."""
    sql_query: str = Field(description="The SQL query to execute")
    explanation: str = Field(description="Why this query was generated")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score 0-1")

# Bind structured output to the model
llm_structured = llm.with_structured_output(SQLOutput)

def generate_sql_node(state: SQLAgentState) -> dict:
    result: SQLOutput = llm_structured.invoke(state["messages"])
    return {
        "sql": result.sql_query,
        "messages": [AIMessage(content=result.explanation)],
    }
```

**Structured output rules:**
- Use `BaseModel` over `TypedDict` for LLM outputs — better validation and error messages
- Always add `Field(description=...)` — it's included in the JSON schema sent to the LLM
- Validate structured output BEFORE using it downstream — LLMs occasionally miss fields

---

## 9. Error Handling in Graphs

```python
import logging
from langchain_core.messages import AIMessage

logger = logging.getLogger(__name__)

def execute_sql_node(state: SQLAgentState) -> dict:
    """Execute generated SQL with proper error handling."""
    try:
        result = db.execute(state["sql"])
        return {"sql_result": str(result), "error": None}
    except ValueError as e:
        # Non-fatal: bad SQL syntax — let agent retry
        logger.warning("SQL syntax error: %s", e)
        return {
            "error": f"SQL error: {e}. Please regenerate.",
            "messages": [AIMessage(content=f"SQL failed: {e}. Retrying...")],
        }
    except Exception as e:
        # Fatal: unexpected error — surface it
        logger.error("Unexpected database error: %s", e, exc_info=True)
        raise  # Let LangGraph's error handling deal with it

# In graph, add retry logic via conditional edge
def should_retry(state: SQLAgentState) -> str:
    if state.get("error") and state.get("iteration_count", 0) < 3:
        return "generate_sql"   # retry
    elif state.get("error"):
        return "error_node"     # max retries exceeded
    return "format_response"    # success
```

---

## 10. LangSmith Observability (Enable from Day One)

```python
# .env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_api_key
LANGCHAIN_PROJECT=my-agent-project  # groups traces by project

# No code changes needed — all LangChain/LangGraph calls are auto-traced
# View traces at: https://smith.langchain.com
```

**LangSmith rules:**
- Enable tracing in development — it's free and invaluable for debugging
- Add `run_name` to config for readable trace names:
  ```python
  config = {"run_name": "sql_generation", "configurable": {"thread_id": "..."}}
  ```
- Use LangSmith for evaluations before shipping — test agent trajectories, not just final output

---

## 11. Testing LangGraph Agents

```python
# tests/unit/test_nodes.py
import pytest
from unittest.mock import MagicMock, patch
from src.agent_name.nodes import generate_sql_node
from src.agent_name.state import SQLAgentState

def test_generate_sql_returns_valid_sql():
    """Node should return SQL for a valid query."""
    state = SQLAgentState(
        query="Show all users older than 30",
        sql="",
        sql_result="",
        messages=[],
        error=None,
    )
    
    with patch("src.agent_name.nodes.llm_structured") as mock_llm:
        mock_llm.invoke.return_value = MagicMock(
            sql_query="SELECT * FROM users WHERE age > 30",
            explanation="Filter users by age",
        )
        result = generate_sql_node(state)
    
    assert "sql" in result
    assert "SELECT" in result["sql"]

# tests/integration/test_graph.py
import pytest
from src.agent_name.graph import graph

@pytest.mark.asyncio
async def test_full_graph_completes():
    """Full graph should return a result for a simple query."""
    result = await graph.ainvoke({
        "messages": [{"role": "user", "content": "How many users are there?"}]
    })
    assert result["messages"]
    assert len(result["messages"]) > 1  # at least user + AI response
```

**Testing rules:**
- Test each node as a pure function — no graph needed
- Mock the LLM in unit tests — test logic, not LLM behavior
- Write one integration test per major agent workflow
- Use `pytest-asyncio` for all async tests
- Add `@pytest.mark.asyncio` to every async test function

---

## 12. Common Anti-Patterns (LangChain team's documented failure modes)

These are mistakes AI agents frequently make with LangGraph:

```python
# ❌ WRONG — Overcomplicating: building custom tool execution instead of using ToolNode
class CustomToolExecutor:
    def execute(self, tool_calls):
        ...  # 50 lines of code that ToolNode already does

# ✅ CORRECT
from langgraph.prebuilt import ToolNode
tool_node = ToolNode(tools)

# ❌ WRONG — Type assumption errors: assuming message types without checking
def get_last_ai_message(state):
    return state["messages"][-1].content  # Crashes if last message is ToolMessage

# ✅ CORRECT
from langchain_core.messages import AIMessage
def get_last_ai_message(state):
    ai_messages = [m for m in state["messages"] if isinstance(m, AIMessage)]
    return ai_messages[-1].content if ai_messages else ""

# ❌ WRONG — Hardcoding schema instead of reading it at runtime
SCHEMA = """CREATE TABLE users (id INT, name TEXT, age INT)"""  # stale!
prompt = f"Use this schema: {SCHEMA}"

# ✅ CORRECT — Read schema dynamically
def get_schema_tool() -> str:
    """Get current database schema."""
    return db.get_schema()  # always fresh

# ❌ WRONG — Deprecated: chaining with | operator for agents (LangChain v0.1 style)
chain = prompt | llm | StrOutputParser()

# ✅ CORRECT for agents: use LangGraph StateGraph
# (chains are still fine for simple, non-agentic pipelines)

# ❌ WRONG — Missing export for LangGraph Platform deployment
# langgraph.json points to "agent_name.graph:graph" but graph isn't exported

# ✅ CORRECT — always export compiled graph at module level
# graph.py
graph = build_graph()  # module-level export required by LangGraph Platform
```

---

## 13. LangGraph Platform Deployment Config

```json
// langgraph.json
{
  "dependencies": ["."],
  "graphs": {
    "agent": "./src/agent_name/graph.py:graph"
  },
  "env": ".env"
}
```

```bash
# Test locally before deploying
langgraph dev   # starts local server with Studio UI

# Deploy to LangGraph Cloud
langgraph deploy
```

---

## 14. Key Documentation URLs

Include these in CLAUDE.md so the agent can fetch them on demand:

- LangGraph docs: https://langchain-ai.github.io/langgraph/llms.txt
- LangChain Python: https://python.langchain.com/llms.txt
- LangGraph conceptual guides: https://langchain-ai.github.io/langgraph/concepts/
- Human-in-the-loop: https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/
- Multi-agent patterns: https://langchain-ai.github.io/langgraph/concepts/multi_agent/
- Streaming: https://langchain-ai.github.io/langgraph/how-tos/streaming/
- LangSmith tracing: https://docs.smith.langchain.com/

---

## 15. Before Every Commit

```bash
ruff format .
ruff check . --fix
mypy src/
pytest tests/ -v --asyncio-mode=auto
```

Also check:
- [ ] `graph` is exported at module level in `graph.py`
- [ ] All nodes return `dict`, not full state
- [ ] `add_messages` reducer used for message fields
- [ ] `.env` not committed (only `.env.example`)
- [ ] LangSmith tracing enabled in `.env`
- [ ] `langgraph.json` points to correct graph path
