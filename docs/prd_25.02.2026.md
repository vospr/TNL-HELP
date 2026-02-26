---
stepsCompleted: [step-01-init, step-02-discovery, step-02b-vision, step-02c-executive-summary, step-03-success, step-04-journeys, step-05-domain, step-06-innovation, step-07-project-type, step-08-scoping, step-09-functional, step-10-nonfunctional, step-11-polish]
inputDocuments:
  - C:\Users\AndreyPopov\Documents\EPAM\Project_Temp_automated\_bmad-output\brainstorming\brainstorming-session-2026-02-24.md
workflowType: 'prd'
documentCounts:
  briefs: 0
  research: 0
  brainstorming: 1
  projectDocs: 0
  projectContext: 0
classification:
  projectType: staged_learning_implementation
  domain: ai_agent_orchestration
  complexity: progressive_low_to_high
  projectContext: behavioral_reference
  primarySuccess: learning_first_stages_1_4_function_first_stages_5_7
  stageIsolation: standalone_test_graph_with_mocks
  stateSchema: core_spine_stage_1_plus_per_stage_extensions
  coreArtifacts: stage_report_per_stage_plus_primitives_coverage_map
vision:
  summary: Universal multi-agent project delivery engine on LangGraph
  differentiator: Runtime programmable orchestration — CE Template behavior made deployable and scalable
  coreInsight: Reliability from protocols and agent specialization, not larger models
  users: [developers_seeking_acceleration, non_technical_users, operator_any_business_case]
  qualityEdge: [better_architecture, fewer_bugs, full_traceability]
demoScenario: rag_chatbot
---

# Product Requirements Document - Project_Temp_automated

**Author:** Andrey
**Date:** 2026-02-24

## Executive Summary

This product is a universal multi-agent project delivery engine built on LangGraph. It translates the proven orchestration model of the Context Engineering Template — stateless dispatch, 6 specialized agents, spec-driven workflows, governance gates — from markdown conventions into a programmable Python runtime with real state management, checkpointing, and human-in-the-loop control.

The system takes any business case (a RAG chatbot, a macOS translator app, an API backend) and orchestrates the full delivery lifecycle: planning, architecture, implementation, review, and testing. It serves three user classes equally: developers seeking acceleration, non-technical users wanting autonomous delivery, and operators pointing it at arbitrary business cases.

### What Makes This Special

Programmable orchestration with quality guarantees. Where raw Claude Code relies on a single agent's judgment, this system enforces structure through specialized agents, governance gates, and evidence-based verification. The result: better architecture (architect agent shapes it), fewer bugs (reviewer agent catches them), and full traceability (specs, assertions, and evidence link every output to intent).

The core insight: reliability comes from protocols and agent specialization, not larger models. LangGraph makes this programmable — the CE Template's behavioral model becomes a deployable, scalable, reproducible runtime.

## Project Classification

- **Project Type:** Staged learning implementation
- **Domain:** AI agent orchestration
- **Complexity:** Progressive: low → high
- **Context:** Behavioral reference — CE Template = WHAT, LangGraph = HOW
- **Primary Success:** Learning-first (Stages 1-4), function-first (Stages 5-7)
- **Stage Isolation:** Each stage has standalone test graph with mocks
- **State Schema:** Core spine in Stage 1 + per-stage extensions
- **Core Artifacts:** Stage Report per stage + Primitives Coverage Map (project-level)

## Success Criteria

### User Success
- Users can describe a business case (e.g., "build a RAG chatbot") and the system orchestrates planning, architecture, implementation, review, and testing without manual intervention beyond decision points.
- Users can intervene at key decision points (architecture choices, scope questions, review feedback) and the system respects and incorporates their input.
- Output quality is measurably better than raw Claude Code: structured architecture, fewer bugs caught by reviewer, full traceability from intent to implementation.

### Business Success
- 3-month win: one end-to-end project (RAG chatbot) delivered through the system with full artifact trail — specs, architecture decisions, implementation evidence, review reports.
- 12-month win: multiple business cases (different domains, different complexity levels) successfully delivered, proving universality of the orchestration model.

### Technical Success
- LangGraph graph executes the full dispatch loop: task selection → agent routing → execution → result processing → next task.
- Worker-reviewer loop enforces quality with circuit breaker (max 3 cycles).
- State schema supports the full lifecycle without mid-project redesign.
- Each build stage is independently testable with standalone test graphs.
- Checkpointing enables pause/resume without state loss.

### Measurable Outcomes
- 1 complete RAG chatbot delivered through the system as proof-of-concept demo.
- All 6 agent roles (researcher, planner, architect, implementer, reviewer, tester) exercised in the demo project.
- 0 tasks requiring manual intervention outside of designated human-in-the-loop points.
- Primitives Coverage Map shows all target LangGraph features exercised across 7 stages.
- Stage Reports exist for every completed stage.

## User Journeys

### 1) Developer — Alex (Success Path)

**Opening Scene:** Alex is a mid-level Python developer at a startup. The CEO wants a customer support chatbot with RAG over the company's knowledge base. Alex has built simple chatbots before but never architected a production RAG system — retrieval strategy, chunking, embedding pipeline, conversation memory, evaluation. He's staring at 15 open browser tabs of conflicting LangChain tutorials.

**Rising Action:** Alex describes his goal to the system: "Build a customer support chatbot with RAG over our documentation." The dispatcher routes to the planner, which breaks the project into specs — embedding pipeline, retrieval service, conversation chain, evaluation harness. Alex reviews the plan at an interrupt point, adjusts the chunking strategy from fixed-size to semantic, and approves. The implementer builds each component while the reviewer catches a missing error handler on the vector store connection and an overly broad retrieval query that would leak internal docs.

**Climax:** The reviewer flags that the chatbot has no fallback when retrieval returns zero results — it would hallucinate. The implementer adds a confidence threshold with a graceful "I don't have information about that" response. Alex didn't think of this. The system caught it because the reviewer agent has a structured checklist, not because Alex is inexperienced.

**Resolution:** Alex has a working RAG chatbot with proper error handling, retrieval guardrails, and test coverage. He also has architecture decisions documented, a spec trail he can show his CTO, and a codebase he understands because every decision was surfaced at interrupt points. He ships in 3 days instead of 3 weeks.

### 2) Developer — Alex (Edge Case / Recovery)

**Opening Scene:** Same Alex, same project. But this time the knowledge base is 50,000 PDFs with mixed languages (English and Japanese), and the CEO also wants real-time streaming responses.

**Rising Action:** The planner classifies this as COMPLEX tier — multilingual embedding, streaming architecture, and large-scale ingestion are beyond the SIMPLE spec the system initially generated. The dispatcher escalates with an ESCALATED flag. The architect agent is invoked to make technology choices: which multilingual embedding model, streaming protocol (SSE vs WebSocket), and batch ingestion strategy.

**Climax:** The architect proposes a design but the worker-reviewer loop hits 3 cycles on the streaming implementation — the reviewer keeps finding race conditions in the async streaming + retrieval pipeline. Circuit breaker fires. The system interrupts Alex with a structured summary: "3 review cycles exhausted on streaming retrieval. Core issue: async generator lifetime management. Recommendation: simplify to SSE with synchronous retrieval, defer true async streaming to v2."

**Resolution:** Alex accepts the recommendation. The system re-specs the streaming component, implementer builds the simpler version, reviewer approves on first pass. Alex has a working system with a clear documented decision about what was descoped and why. When the CEO asks about "true streaming," Alex has the ADR to explain the trade-off.

### 3) Non-technical User — Maria (Autonomous Path)

**Opening Scene:** Maria runs customer success at a mid-size SaaS company. Support tickets are growing 40% quarter-over-quarter. She's heard AI chatbots can help but has zero coding experience. Her IT team is backlogged for 6 months.

**Rising Action:** Maria describes what she needs in plain language: "I want a chatbot that answers customer questions using our help center articles. It should know when it doesn't know something and offer to connect to a human agent." The system operates autonomously — planner breaks it down, architect selects technologies, implementer builds, reviewer validates. Maria doesn't see any of this. She sees progress updates: "Planning complete. Building retrieval pipeline. Building conversation interface. Running quality checks."

**Climax:** The system completes the chatbot and presents Maria with a test interface. She asks it three questions she knows the answers to — it gets two right and correctly says "I'm not sure about that, let me connect you with our support team" on the third. She asks a question about billing that's not in the help center — it doesn't hallucinate, it escalates.

**Resolution:** Maria has a working chatbot without writing code, without waiting for IT, without understanding what RAG means. She has a deployment package and a one-page summary of what was built and how it works. When IT eventually reviews it, the full artifact trail (specs, architecture, review reports) gives them confidence to deploy to production.

### 4) Operator — Andrey (System Configuration)

**Opening Scene:** Andrey has built the delivery engine. A new client needs a RAG chatbot but with strict compliance requirements — the chatbot must never generate responses from outside the approved knowledge base, all responses must be logged with retrieval sources, and the system must support audit trails.

**Rising Action:** Andrey configures the system for the compliance use case: enables governance gates for file scope enforcement, adds an assertion that every response includes source citations, configures the reviewer agent's prompt to check for compliance-specific patterns. He doesn't rewrite code — he adjusts agent prompts, enables/disables gates, and defines spec constraints.

**Climax:** The system runs the project. The governance gate catches that the implementer's initial chatbot design allows the LLM to supplement retrieval results with its own knowledge — a compliance violation. The gate blocks the task and routes it back with a specific violation report. The implementer rebuilds with strict retrieval-only mode.

**Resolution:** Andrey delivers a compliant chatbot without custom development. The governance gates — which he built as part of Stage 6 — proved their value by catching exactly the kind of issue that manual review misses. The audit trail is automatic because the spec-driven workflow requires evidence for every assertion.

### 5) System Maintainer — Kenji (Extension Path)

**Opening Scene:** Kenji is a senior engineer who joined Andrey's team. The delivery engine works well for web applications but a client wants a mobile-first RAG chatbot with push notifications and offline FAQ caching. The current agent toolset doesn't include mobile-specific tools.

**Rising Action:** Kenji adds a new tool to the implementer's toolset for React Native scaffolding. He creates a new agent prompt variant for mobile architecture patterns. He doesn't touch the graph structure, the state schema, or the dispatch loop — he extends the system at the designated extension points (tool allowlists, agent prompts, spec templates).

**Climax:** The system dispatches the mobile RAG chatbot project using Kenji's extensions. The architect agent selects the mobile-optimized architecture. The reviewer agent, using the updated prompt, catches that the offline caching strategy doesn't handle embedding updates — stale vectors would persist until the user reconnects.

**Resolution:** Kenji's extensions work without breaking existing workflows. The system successfully delivers a mobile RAG chatbot. Kenji documents his extensions in a Stage Report. The Primitives Coverage Map doesn't change — this was an extension, not a new LangGraph feature. The system's universality is proven: same orchestration, different domain, different platform.

### Journey Requirements Summary

- Dispatch and routing for task classification, agent matching, and complexity-based model selection.
- Human-in-the-loop interrupts for decision points, circuit breaker escalation, and approval gates.
- Autonomous execution mode with progress reporting and zero-intervention delivery.
- Governance gates for compliance enforcement, file scope auditing, and assertion validation.
- Agent extensibility via tool allowlists, prompt configuration, and spec templates — no graph modification needed.
- Worker-reviewer loop with circuit breaker for quality enforcement and structured escalation.
- Artifact trail with specs, ADRs, review reports, and evidence for auditability.
- Configurable system — operator adjusts behavior through configuration, not code changes.

## Developer Tool Specific Requirements

### Project-Type Overview
Python-first multi-agent delivery engine on LangGraph. The system itself is Python, but the projects it produces are language-agnostic — agents can generate code in any language the LLM supports. Output quality depends on the LLM's proficiency in the target language. Python/TypeScript/JavaScript are primary targets; other languages are supported but not validated in MVP. User interaction is through Claude Code CLI.

### Technical Architecture Considerations
- **Runtime:** Python 3.11+ with LangGraph as the core framework.
- **LLM interface:** MVP uses Claude API via Anthropic SDK. Agent node architecture uses LangGraph's model-configurable pattern, making provider swap possible but not a design goal.
- **State management:** LangGraph StateGraph with TypedDict state schema + SqliteSaver checkpointer.
- **File I/O:** Agents read/write project files on the local filesystem — same model as Claude Code.

### Language Matrix
- **System language:** Python (LangGraph requirement).
- **Output languages:** Language-agnostic — agents generate whatever the business case requires (Python, TypeScript, JavaScript are primary; others supported but unvalidated in MVP).
- **No language-specific assumptions** in the orchestration layer — dispatch, routing, review, and governance operate on task descriptions and file contents, not language ASTs.

### Installation Methods
- Clone the repository.
- `pip install` dependencies (LangGraph, Anthropic SDK, Pydantic).
- Requires: Python 3.11+, Anthropic API key in environment (`ANTHROPIC_API_KEY`). Recommended: virtual environment via `uv` or `venv`.
- No Docker, no hosted service, no package distribution for MVP.

### API Surface

#### Interface Architecture (ADR-001)
- **Graph as API:** `graph.invoke(state)` / `graph.stream(state)` is the primary interface. CLI is a presentation adapter.
- **No I/O inside nodes:** All user interaction flows through LangGraph's `interrupt()` / `Command(resume=)` pattern. Nodes never call `input()` or write to stdout directly.
- **CLI adapter for MVP:** `python -m langgraph_ce_template "<goal>" --output <dir> --mode interactive|autonomous`
- **Adapter-agnostic by design:** Following LangGraph's interrupt pattern means web UI adapter requires zero graph changes when needed.

#### CLI Interaction Modes
- **Interactive:** pauses at decision points (architecture, scope, review), presents options, accepts user input.
- **Autonomous:** auto-resolves interrupts using the system's default recommendation. COMPLEX-tier architecture decisions still interrupt even in autonomous mode — some decisions require human judgment regardless of mode.

#### MVP User Scope
- MVP primary users: developers (interactive mode) and operators (configuration mode).
- Non-technical users are supported through Claude Code's conversational interface in autonomous mode — viable but not the primary validation target for MVP.

### Output Project Management
- System creates output projects in a designated target directory (`--output`), separate from the orchestration engine codebase.
- Output includes: source code, dependency manifest, configuration files, run instructions, and test suite.
- Each output project is self-contained and runnable without the orchestration engine.
- Minimal output contract: `README.md` (run instructions), `requirements.txt` or equivalent (dependencies), `tests/` (test suite), `docs/adr/` (architecture decisions). Internal source structure is architect-determined per project type.

### Project Context Input
- `--context <path>` flag provides project-specific input (document directories, reference code, existing configs).
- Context is a local directory path. The system scans it for relevant files during planning.
- If no context provided, planner asks interactively. Context is additive — flag provides initial context, planner can request more.

### Architecture Decision Persistence
- Architect agent writes ADR files (`docs/adr/`) in the output project directory, not just graph state.
- Implementer reads ADRs as constraints. Reviewer validates against them.
- User can always inspect decisions by reading files, not debugging graph state.

### Project Delivery Completeness
- Every output project includes: dependency management, configuration, run/build instructions, and passing tests.
- Architecture decisions propagate as constraints to downstream agents — implementer cannot contradict architect.
- Final integration test validates the complete output project runs end-to-end.
- Output test framework matches the output project's language (pytest for Python, Jest for TypeScript, etc.). Tester agent selects framework based on architect's technology choices.

### Operator Configuration
- Operator profiles are a Stage 6+ feature. MVP uses hardcoded defaults.
- Profile system is designed in Stage 6 when governance gates exist.
- Profiles override: agent prompts, tool allowlists, gate enable/disable settings.
- `--profile <name>` flag selects configuration at runtime.

### Extension Points
- **Agent tools** — add capabilities without graph changes.
- **Agent prompts** — modify behavior without code changes.
- **Governance gates** — add validation rules.
- **Output templates** — add project scaffolding patterns.
- All extensions are additive — they don't modify existing system files.

### Test Domain Separation
- **Engine tests:** validate orchestration (dispatch, routing, review loop, governance gates) using mock agents and controlled scenarios.
- **Output tests:** validate the delivered project (generated by agents) runs correctly. These are part of the output project, written by the tester agent.

### Code Examples
- Stage Reports serve as living documentation with code examples per stage.
- RAG chatbot demo project serves as the end-to-end reference example.

### Implementation Considerations
- Keep the orchestration layer thin — LangGraph does the heavy lifting.
- Agent prompts and tool allowlists are the primary extension points — no graph modification needed to add new capabilities.
- State schema core spine is designed once in Stage 1; per-stage extensions add fields without breaking existing stages.

## Project Scope & Phased Development

### MVP Strategy & Philosophy
**MVP Approach:** Platform MVP — prove the orchestration engine works end-to-end by delivering one real project (RAG chatbot) through it.
**Resource Requirements:** 1 developer (Andrey), part-time. No team needed for MVP.

### MVP Feature Set (Phase 1 — Stages 1-4)
**Core Journeys Supported:** Alex (developer, interactive mode) — Journey 1 only. Other journeys require Stages 5-7.

**Must-Have Capabilities:**
- **Stage 1:** Minimal dispatch loop (task selector → implementer → result processor) with core state spine.
- **Stage 2:** Multi-agent routing (all 6 agents as create_react_agent subgraphs with role-specific tools/prompts).
- **Stage 3:** Worker-reviewer loop with circuit breaker (flat conditional loop, max 3 cycles).
- **Stage 4:** Parallel fan-out via map-reduce (parallel planner → Send() → parallel collector).
- Output project generation in target directory.
- Core state spine supporting all 7 stages.
- Engine test suite per stage.
- **Exit criterion per stage:** code works + tests pass + Stage Report written.
- **MVP demo:** RAG chatbot project orchestrated through Stages 1-4.

**Explicitly NOT in MVP:**
- Human-in-the-loop interrupts (Stage 5) — MVP runs autonomously with hardcoded decisions
- Governance gates (Stage 6) — MVP relies on reviewer agent, not automated gates
- Persistence/checkpointing (Stage 7) — MVP runs in single session, no pause/resume
- Operator profiles — MVP uses defaults
- Maria's journey (non-technical autonomous) — MVP validates developer path first

### Post-MVP Features

**Phase 2 (Growth — Stages 5-6):**
- Human-in-the-loop with interrupt/resume (enables Alex Journey 2 and Maria Journey 3)
- Governance gates with Pydantic validation (enables Andrey Journey 4)
- Operator profile system
- Autonomous mode with smart default resolution

**Phase 3 (Expansion — Stage 7+):**
- SqliteSaver checkpointing for pause/resume across sessions
- File sync and token management
- Kenji's extension path (Journey 5) fully validated
- Web UI adapter
- Multi-project orchestration

### Risk Mitigation Strategy

**Technical Risks:**
- *State schema breaks across stages* → Mitigated: core spine designed in Stage 1 for all 7 stages, extensions per stage.
- *LangGraph API limitations* → Mitigated: brainstorming confirmed all required primitives exist (StateGraph, create_react_agent, Send, interrupt, checkpointer).
- *Agent quality varies by output language* → Mitigated: MVP validates Python output only (RAG chatbot). Other languages post-MVP.

**Market Risks:**
- *"Why not just use Claude Code directly?"* → Mitigated: RAG chatbot demo must show concrete quality improvements (bugs caught by reviewer, architecture shaped by architect) that raw Claude Code misses.
- *Perceived overhead of orchestration* → Mitigated: one-command entry point, autonomous mode for simple cases.

**Resource Risks:**
- *Single developer, limited bandwidth* → Mitigated: 7-stage ladder keeps each increment small. Any stage can be the stopping point with a working system at that level.
- *Context loss between sessions* → Mitigated: Stage Reports and Primitives Coverage Map serve as session-independent state. Git history as fault tolerance.

## Functional Requirements

*74 FRs across 10 capability areas. Each tagged: MVP (Stages 1-4), Growth (Stages 5-6), or Vision (Stage 7+). Risk flags on critical FRs.*

### Dispatch & Task Routing
- FR1: [MVP] System can accept a natural language project goal and initiate the delivery lifecycle.
- FR2: [MVP] Dispatcher can select the lowest-priority unblocked task from the task queue.
- FR3: [MVP] Dispatcher can classify task complexity (TRIVIAL/SIMPLE/MODERATE/COMPLEX).
- FR4: [MVP] Dispatcher can match tasks to the appropriate agent role based on task type.
- FR5: [MVP] Dispatcher can select the appropriate model tier based on task complexity.
- FR6: [MVP] Dispatcher can assign a Trace ID to every dispatched task.
- FR7: [MVP] Result processor can update task status (completed/blocked) after agent execution.
- FR8: [Growth] Result processor can re-dispatch tasks when ESCALATED complexity is signaled.
- FR9: [Growth] Result processor can trigger re-spec when NEEDS_RESPEC is signaled.
- FR50: [MVP] Planner can request clarification from the user when a project goal is too ambiguous to decompose into tasks.
- FR51: [Growth] System can enforce a maximum of one complexity escalation per task, blocking on second escalation.

### Agent Execution
- FR10: [MVP] System can route tasks to 6 specialized agent roles (researcher, planner, architect, implementer, reviewer, tester).
- FR11: [MVP] Each agent can operate as an independent subgraph with role-specific tools and system prompt.
- FR12: [MVP] Implementer can create, edit, and write files in the output project directory.
- FR13: [MVP] Reviewer can read and analyze code without write access to project files.
- FR14: [MVP] **HIGH RISK** Planner can break a project goal into structured task specs with assertions. *Quality of everything downstream depends on planner decomposition. Mitigated by human review of plan in interactive mode.*
- FR15: [MVP] Architect can make technology selections and write ADR files in the output project.
- FR16: [MVP] Tester can execute tests and report pass/fail results with evidence.
- FR17: [Growth] Researcher can perform web search and document analysis for background research.
- FR52: [MVP] System can restrict implementer file operations to the designated output project directory only.
- FR74: [MVP] Implementer can read architecture decisions and treat them as binding constraints during implementation.

### Worker-Reviewer Quality Loop
- FR18: [MVP] System can route implementation tasks through a worker-reviewer cycle.
- FR19: [MVP] Reviewer can return structured feedback (APPROVED/NEEDS_CHANGES/BLOCKED) with numbered issues.
- FR20: [MVP] System can loop implementer back with reviewer feedback when NEEDS_CHANGES is returned.
- FR21: [MVP] System can enforce a circuit breaker after 3 review cycles and escalate to blocked status.
- FR54: [Growth] Reviewer can return structured issues with file path, line reference, severity (CRITICAL/MAJOR/MINOR), and specific fix guidance.
- FR55: [Growth] System can record failure patterns from circuit breaker events for future reference during planning and review.

### Parallel Execution
- FR22: [MVP] **HIGH RISK** System can identify independent tasks that do not share file scope. *Conservative approach for MVP: only parallelize when planner explicitly marks tasks as parallelizable.*
- FR23: [MVP] System can dispatch multiple agents in parallel for non-overlapping tasks.
- FR24: [MVP] System can collect parallel results and merge them into unified state.
- FR56: [MVP] Planner can annotate tasks with explicit dependency declarations beyond file scope.

### Human-in-the-Loop
- FR25: [Growth] System can interrupt execution at decision points requiring human judgment.
- FR25b: [MVP] System can interrupt once after planning phase for user to review and approve the task plan before execution begins.
- FR26: [Growth] System can present structured options to the user at interrupt points.
- FR27: [Growth] System can resume execution with user's decision applied to state.
- FR28: [Growth] System can auto-resolve simple interrupts in autonomous mode using default recommendations.
- FR29: [Growth] System can force interrupts for COMPLEX-tier architecture decisions regardless of mode.
- FR57: [Growth] System can log all auto-resolved decisions in autonomous mode with the recommendation used and alternatives considered.
- FR65: [MVP] User can cancel the current task and revert file changes made by the in-progress agent. *HIGH RISK — mitigated by git-based rollback: commit before each task, revert on cancel.*
- FR66: [Growth] System can present interrupt decisions with plain-language explanations and recommendations.

### Governance & Validation
- FR30: [Growth] System can validate spec packets against controlled vocabulary (MUST/SHOULD/MAY/MUST NOT).
- FR31: [Growth] System can audit that implementer's file changes stay within declared file scope.
- FR32: [Growth] System can verify that every assertion has PASS/FAIL evidence with file:line references.
- FR33: [Growth] System can block tasks that fail governance checks and route back with violation reports.
- FR58: [Growth] Implementer can request file scope expansion with justification, subject to reviewer approval.

### Output Project Management
- FR34: [MVP] System can create a self-contained output project in a user-specified target directory.
- FR35: [MVP] System can generate a dependency manifest in the output project.
- FR36: [MVP] System can generate a test suite in the output project appropriate to the project's language/framework.
- FR37: [MVP] System can generate ADR documentation in the output project for all architecture decisions.
- FR38: [MVP] System can validate that generated tests in the output project pass. *Full end-to-end integration validation is manual for MVP; automated in Growth.*
- FR59: [Growth] System can generate environment setup instructions or scripts in the output project for required services.
- FR60: [Growth] System can detect and flag dependency version conflicts across tasks within the same output project.
- FR67: [Growth] System can generate a non-technical project summary describing what was built, how to run it, and what it does.

### State & Persistence
- FR39: [MVP] System can maintain project state across the full task lifecycle via state channels.
- FR40: [Vision] System can checkpoint state for pause/resume across sessions.
- FR41: [Vision] System can recover from interrupted sessions using checkpointed state.
- FR42: [MVP] **HIGH RISK** System can maintain a state schema that accommodates all stage extensions without breaking changes. *The single most important Stage 1 design decision. Mitigated by TypedDict with Optional fields + per-stage extensions.*
- FR61: [Vision] System can validate checkpoint integrity on resume and fall back to last known good state if corruption is detected.

### Progress & User Experience
- FR63: [MVP] System can display real-time progress to the user (current task, active agent, tasks remaining, tasks completed).
- FR64: [Vision] System can analyze an existing codebase in the output directory and plan tasks that extend rather than overwrite it.

### Observability & Diagnostics
- FR47: [MVP] Artifacts can include Trace IDs for cross-agent debugging.
- FR48: [MVP] System can produce Stage Reports documenting what was built, primitives used, and key decisions.
- FR49: [MVP] System can maintain a Primitives Coverage Map tracking framework primitives exercised.
- FR62: [MVP] System can validate that each stage's target framework primitives are exercised before marking stage complete.
- FR68: [Growth] System can maintain a project delivery log recording each completed project, its configuration, and outcome summary.
- FR69: [Vision] System can aggregate quality metrics across projects (circuit breaker frequency, reviewer severity distribution, task completion rates).
- FR70: [Vision] Operator can version control profile configurations and trace output quality to specific prompt/configuration versions.
- FR71: [MVP] Maintainer can test individual agent subgraphs in isolation with mock tasks and controlled inputs.
- FR72: [Growth] System can validate that registered extensions meet the expected interface contract before runtime execution.
- FR73: [MVP] System can expose agent-level execution traces (messages, tool calls, decisions) for debugging individual agent behavior.

### System Integrity
- FR75: [MVP] System can produce a functional, demonstrable system at the completion of any individual stage.

## Non-Functional Requirements

### Performance
- NFR1: Dispatch loop overhead (task selection + agent routing + result processing) completes within 5 seconds per cycle, excluding agent execution time.
- NFR2: Agent execution time is bounded by the LLM provider's API latency, not by the orchestration engine. The engine adds no unnecessary file reads or state operations between dispatch and agent invocation.
- NFR3: Progress updates (FR63) are displayed within 1 second of state changes.

### Security
- NFR4: API keys are read from environment variables only, never stored in configuration files or state.
- NFR5: The system never writes API keys, credentials, or secrets to output project files, state checkpoints, or logs.
- NFR6: File operations are sandboxed to the output directory (FR52); the engine's own source files are never modified by agents.

### Reliability & Continuity
- NFR7: If an agent invocation fails (API timeout, malformed response), the system retries once, then marks the task as blocked with the error details — never silently drops a task.
- NFR8: If the system is interrupted mid-session (process killed, terminal closed), no output project files are left in a corrupted state. Git-based micro-commits before each task ensure rollback is possible.
- NFR9: If a state channel receives unexpected data (wrong type, missing field), the system logs the violation and continues with a safe default rather than crashing.

### Developer Experience
- NFR10: A new user can go from `git clone` to a running dispatch loop in under 5 minutes (install + configure API key + run first command).
- NFR11: Error messages include actionable context: which agent failed, which task, what the error was, and suggested next steps.
- NFR12: Agent execution traces (FR73) are human-readable without specialized tooling — plain text or structured JSON viewable in any editor.
