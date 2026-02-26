# Intentional Placeholder Skills: Design Documentation

## Executive Summary

The Claude Code Context Engineering Template includes **four intentional placeholder skills** that are deliberately left unpopulated:

1. **`coding-standards.md`** — Project-specific code conventions and style rules
2. **`review-checklist.md`** — Code review severity thresholds and automated checks
3. **`testing-strategy.md`** — Test framework and CI/CD integration strategy
4. **`architecture-principles.md`** — System design constraints and quality attributes

These are **not incomplete artifacts** waiting to be filled. They are **architectural placeholders** by design, requiring each project to customize them for its own stack, conventions, and standards. This document explains the intentionality, provides customization guidance, and validates this design choice against the template's architecture.

---

## Part 1: Design Intent

### Why Skills Are Placeholders

The Context Engineering Template is **project-agnostic** by definition. The README states:

> "This template encodes Context Engineering: controlling what enters an LLM context, when, in what form, and what stays externalized."

The template intentionally provides:
- **Fixed orchestration** (CLAUDE.md dispatch loop)
- **Fixed agent definitions** (.claude/agents/)
- **Fixed governance protocol** (spec-protocol.md)
- **Customizable project knowledge** (skills)

The four skills are customizable because they capture **project-specific operating standards** that vary completely between projects:

- A TypeScript/Node.js project has different `coding-standards` than a Python/FastAPI project
- A microservices architecture has different `architecture-principles` than a monolith
- A project using Jest/TypeScript has different `testing-strategy` than one using Pytest/Go
- A fintech project has different `review-checklist` severity thresholds than a prototype

**This is intentional.** The template would be less useful if it baked in assumptions about your stack, framework, or conventions. Instead, it provides the **structure** (what each skill contains) and asks you to provide the **content** (values specific to your project).

---

## Part 2: Per-Skill Analysis

### Skill 1: `coding-standards.md`

#### Current Placeholder Structure

```markdown
# Coding Standards

> **[PLACEHOLDER]** — Customize this file for your project's language, framework, and conventions.

## Language & Framework
- **Language:** {e.g., TypeScript, Python, Go, Rust}
- **Framework:** {e.g., React, FastAPI, Express, None}
- **Runtime:** {e.g., Node.js 20, Python 3.12, Go 1.22}

## Naming Conventions
- **Files:** {e.g., kebab-case.ts, snake_case.py}
- **Functions:** {e.g., camelCase, snake_case}
- **Classes:** {e.g., PascalCase}
- **Constants:** {e.g., UPPER_SNAKE_CASE}
- **Variables:** {e.g., camelCase, snake_case}

## Code Style
- **Linter:** {e.g., ESLint, Ruff, golangci-lint}
- **Formatter:** {e.g., Prettier, Black, gofmt}
- **Lint command:** {e.g., `npm run lint`, `ruff check .`}
- **Format command:** {e.g., `npm run format`, `black .`}

## Patterns to Follow
1. {e.g., Dependency injection for services}
2. {e.g., Repository pattern for data access}
3. {e.g., Error boundaries for React components}

## Patterns to Avoid
1. {e.g., No global mutable state}
2. {e.g., No any types in TypeScript}
3. {e.g., No raw SQL — use query builder}

## File Organization
```
src/
  {your structure here}
```

## Customization Instructions
1. Replace all `{placeholder}` values with your project specifics
2. Remove sections that don't apply
3. Add project-specific rules as needed
4. Keep total under 80 lines — this loads into agent context
```

#### Fields to Customize

| Field | User Customizes | Example |
|-------|-----------------|---------|
| Language | Stack choice | TypeScript, Python, Rust |
| Framework | Framework choice | React, FastAPI, Express, Spring |
| Runtime | Runtime environment | Node.js 20, Python 3.12, Go 1.22 |
| Naming Conventions | Project standards | camelCase functions, PascalCase classes |
| Linter | CI/CD tooling | ESLint, Ruff, golangci-lint |
| Formatter | Code formatting | Prettier, Black, gofmt |
| Patterns to Follow | Best practices | DI, repository pattern, error handling |
| Patterns to Avoid | Anti-patterns | Global state, raw SQL, weak types |
| File Organization | Directory structure | src/api, src/services, src/models |

#### Example Customization: TodoAPI (Node.js + Express)

```markdown
# Coding Standards

## Language & Framework
- **Language:** TypeScript
- **Framework:** Express.js
- **Runtime:** Node.js 20.x

## Naming Conventions
- **Files:** kebab-case.ts (e.g., user-service.ts, auth-controller.ts)
- **Functions:** camelCase (e.g., createUser, validateInput)
- **Classes:** PascalCase (e.g., UserService, AuthController)
- **Constants:** UPPER_SNAKE_CASE (e.g., MAX_RETRY_ATTEMPTS, API_TIMEOUT_MS)
- **Variables:** camelCase (e.g., userId, isValid)

## Code Style
- **Linter:** ESLint with @typescript-eslint/parser
- **Formatter:** Prettier (80-char line width)
- **Lint command:** `npm run lint`
- **Format command:** `npm run format`

## Patterns to Follow
1. Dependency injection for all services (use tsyringe decorators)
2. Repository pattern for data access (UserRepository, TodoRepository)
3. Error handling: custom AppError class with code + message

## Patterns to Avoid
1. No global mutable state — use DI instead
2. No `any` types in TypeScript — explicit typing required
3. No direct database queries — use query builder only

## File Organization
```
src/
  controllers/     # Express request handlers
  services/        # Business logic
  repositories/    # Data access
  middleware/      # Authentication, logging
  types/           # TypeScript interfaces
  models/          # Database models
  utils/           # Shared utilities
```
```

#### How the Skill Is Used

- **Implementer** reads this when writing code to match project conventions
- **Reviewer** uses this during code review (CLAUDE.md Step 5: Process Result → dispatch reviewer)
- **Planner** may reference this when authoring constraints in spec packets

---

### Skill 2: `review-checklist.md`

#### Current Placeholder Structure

```markdown
# Code Review Checklist

> **[PLACEHOLDER]** — Customize severity thresholds and automated checks for your project.

## Automated Checks (Run Before Manual Review)
- [ ] Linter passes: `{e.g., npm run lint}`
- [ ] Type checker passes: `{e.g., npx tsc --noEmit}`
- [ ] Tests pass: `{e.g., npm test}`
- [ ] Build succeeds: `{e.g., npm run build}`

## Code Quality Checklist

### Correctness
- [ ] Code does what the task description requires
- [ ] Edge cases handled (null, empty, boundary values)
- [ ] Error handling is appropriate (not swallowed, not excessive)
- [ ] No obvious bugs or logic errors

### Readability
- [ ] Names are clear and descriptive
- [ ] Functions are focused (single responsibility)
- [ ] No unnecessary complexity or over-engineering
- [ ] Comments explain WHY, not WHAT (if present)

### Architecture
- [ ] Follows established patterns from ADRs
- [ ] No new dependencies without justification
- [ ] Respects component boundaries
- [ ] Changes are within the scope of the task

### Testing
- [ ] New code has corresponding tests
- [ ] Tests cover the happy path and key edge cases
- [ ] Tests are deterministic (no flaky tests)
- [ ] Test names describe the behavior being verified

### Security
- [ ] No hardcoded secrets, tokens, or credentials
- [ ] User input is validated/sanitized
- [ ] No SQL injection, XSS, or command injection vectors
- [ ] Authentication/authorization checks present where needed

### Performance
- [ ] No obvious N+1 queries or unbounded loops
- [ ] Large data sets handled with pagination/streaming
- [ ] No blocking operations in async contexts

## Severity Guidelines
- **CRITICAL**: Security vulnerabilities, data corruption, crashes, incorrect business logic
- **MAJOR**: Missing error handling, missing tests, architecture violations, performance issues
- **MINOR**: Naming improvements, minor refactoring opportunities, documentation gaps

## Customization Instructions
1. Replace automated check commands with your project's actual commands
2. Add project-specific checklist items
3. Adjust severity definitions for your risk tolerance
4. Keep total under 80 lines
```

#### Fields to Customize

| Field | User Customizes | Example |
|-------|-----------------|---------|
| Automated Checks | CI/CD commands | lint, typecheck, test, build |
| Quality Criteria | Review focus areas | Add fintech-specific checks, remove N/A items |
| Severity Definitions | Risk tolerance | Adjust what counts as CRITICAL vs MAJOR |
| Project-Specific Rules | Domain rules | Security, compliance, performance targets |

#### Example Customization: TodoAPI (Node.js + Express)

```markdown
# Code Review Checklist

## Automated Checks (Run Before Manual Review)
- [ ] Linter passes: `npm run lint`
- [ ] Type checker passes: `npx tsc --noEmit`
- [ ] Tests pass: `npm test`
- [ ] Build succeeds: `npm run build`

## Code Quality Checklist

### Correctness
- [ ] Code matches task assertions (spec-protocol.md Section 11)
- [ ] Edge cases handled: null checks, empty lists, invalid IDs
- [ ] Error handling uses AppError class with proper codes
- [ ] No console.log in production code (use logger)

### Readability
- [ ] Names follow kebab-case files, camelCase functions
- [ ] Functions <= 30 lines (single responsibility)
- [ ] No arrow-hell nesting (max 3 levels)
- [ ] Comments explain WHY changes, not WHAT code does

### Architecture
- [ ] Follows DI pattern (services injected, not global)
- [ ] Uses Repository pattern for data access
- [ ] No circular dependencies between modules
- [ ] Changes confined to declared file_scope

### Testing
- [ ] Unit tests present (jest suite or mocha)
- [ ] Happy path + 2 edge cases minimum per function
- [ ] Tests use mocked repos/services
- [ ] Test names describe behavior: "should return 400 when email is empty"

### Security
- [ ] No .env values, API keys, or tokens in code
- [ ] User input validated before processing (input validation middleware)
- [ ] No raw SQL — query builder only
- [ ] Auth checks present on protected routes

### Performance
- [ ] No N+1 queries (batch load or use joins)
- [ ] Large lists paginated (default limit 50)
- [ ] Async operations never block event loop
- [ ] Database indexes added for WHERE/JOIN columns

## Severity Guidelines
- **CRITICAL**: Security leak, unhandled auth, crash on null, missing assertions
- **MAJOR**: Missing tests, missing error handling, architecture violation
- **MINOR**: Naming clarity, comment improvements, formatting

## Customization Instructions
1. Adjust commands for your CI/CD (GitHub Actions, GitLab, etc.)
2. Add TodoAPI-specific checks (e.g., pagination validation)
3. Severity reflects production expectations (high risk = CRITICAL)
```

#### How the Skill Is Used

- **Reviewer** reads this when conducting code review (CLAUDE.md mentions reviewer agent, Section 3)
- **Dispatch loop** references this for quality gate enforcement
- **Circuit breaker** uses severity guidelines to determine escalation level (CLAUDE.md Step 6)

---

### Skill 3: `testing-strategy.md`

#### Current Placeholder Structure

```markdown
# Testing Strategy

> **[PLACEHOLDER]** — Customize this file for your project's test framework and conventions.

## Test Framework
- **Framework:** {e.g., Jest, Pytest, Go testing, Cargo test}
- **Assertion library:** {e.g., built-in, Chai, assertpy}
- **Mocking:** {e.g., jest.mock, unittest.mock, testify/mock}

## Test Commands
```bash
# Unit tests
{e.g., npm test, pytest tests/unit, go test ./...}

# Integration tests
{e.g., npm run test:integration, pytest tests/integration}

# End-to-end tests
{e.g., npm run test:e2e, playwright test}

# Coverage report
{e.g., npm run test:coverage, pytest --cov=src}

# Single test file
{e.g., npx jest path/to/test, pytest path/to/test.py}
```

## Test Organization
```
tests/
  unit/          # Fast, isolated, no external dependencies
  integration/   # Tests with databases, APIs, file system
  e2e/           # Full user flow tests
  fixtures/      # Shared test data
```

## Naming Conventions
- **Test files:** `{e.g., *.test.ts, test_*.py, *_test.go}`
- **Test names:** `{e.g., "should {behavior} when {condition}"}`
- **Describe blocks:** `{e.g., describe('{ComponentName}', ...)}`

## Mocking Strategy
- Mock external services (APIs, databases) in unit tests
- Use real implementations in integration tests
- Never mock the system under test
- Prefer dependency injection over monkey-patching

## Coverage Requirements
- **Minimum coverage:** {e.g., 80%}
- **Critical paths:** {e.g., 100% for auth, payment}
- **New code:** {e.g., must include tests}

## CI/CD Integration
- Tests run on: {e.g., every push, PR only}
- Pipeline: {e.g., lint → typecheck → unit → integration → e2e}
- Blocking: {e.g., any failure blocks merge}

## Customization Instructions
1. Replace all `{placeholder}` values with your project specifics
2. Add project-specific test patterns or utilities
3. Remove test types you don't use
4. Keep total under 80 lines
```

#### Fields to Customize

| Field | User Customizes | Example |
|-------|-----------------|---------|
| Framework | Test runner | Jest, Pytest, Go testing |
| Assertion Library | Assert utility | Chai, assertpy, built-in |
| Mocking | Mock library | jest.mock, unittest.mock, testify |
| Commands | Test invocations | npm test, pytest tests/unit |
| Organization | Directory structure | tests/unit, tests/integration, tests/e2e |
| Naming | Test file/case naming | *.test.ts, should_X_when_Y |
| Coverage Requirements | Target percentages | 80% minimum, 100% for auth |
| CI/CD Pipeline | Automation | GitHub Actions, when blocking |

#### Example Customization: TodoAPI (Node.js + Jest)

```markdown
# Testing Strategy

## Test Framework
- **Framework:** Jest 29.x
- **Assertion library:** Jest built-in expect()
- **Mocking:** jest.mock() with jest.spyOn()

## Test Commands
```bash
# Unit tests
npm test

# Unit tests watch mode (development)
npm test -- --watch

# Integration tests
npm run test:integration

# End-to-end tests
npm run test:e2e

# Coverage report
npm run test:coverage

# Single test file
npx jest src/services/user-service.test.ts
```

## Test Organization
```
src/
  {source files}
tests/
  unit/                          # Jest tests, no DB
    services/
      user-service.test.ts
      todo-service.test.ts
    middleware/
      auth.test.ts
  integration/                   # Real database
    api/
      users.integration.test.ts
      todos.integration.test.ts
  e2e/                          # Full flow
    user-workflow.e2e.test.ts
  fixtures/
    user-fixtures.ts
    db-fixtures.ts
```

## Naming Conventions
- **Test files:** `*.test.ts` for unit, `*.integration.test.ts` for integration
- **Test names:** `should {behavior} when {condition}` (e.g., "should return 400 when email is empty")
- **Describe blocks:** `describe('{ServiceName}', ...)` (e.g., `describe('UserService', ...)`)

## Mocking Strategy
- Unit tests: mock UserRepository, use jest.mock()
- Integration tests: real in-memory SQLite database
- Never mock the service under test
- Use dependency injection for testability

## Coverage Requirements
- **Minimum coverage:** 80% across entire project
- **Critical paths:** 100% for auth middleware, payment routes
- **New code:** MUST include tests before merge

## CI/CD Integration
- Tests run on: Every push to any branch AND every PR
- Pipeline: lint → typecheck → unit → integration → e2e
- Blocking: Any test failure blocks merge to main
- Coverage: Report to GitHub (codecov integration)
```

#### How the Skill Is Used

- **Implementer** reads this when writing tests (part of coding standard)
- **Tester** uses this to execute test suite and generate reports (CLAUDE.md Step 3 tester agent)
- **Reviewer** checks that tests follow this strategy (review-checklist.md)

---

### Skill 4: `architecture-principles.md`

#### Current Placeholder Structure

```markdown
# Architecture Principles

> **[PLACEHOLDER]** — Customize this file for your project's architectural style and constraints.

## System Type
- **Type:** {e.g., Web application, CLI tool, API service, Mobile app}
- **Architecture:** {e.g., Monolith, Microservices, Serverless, Modular monolith}
- **Deployment:** {e.g., Docker/K8s, Vercel, AWS Lambda, bare metal}

## Technology Constraints
- **Must use:** {e.g., PostgreSQL for persistence, Redis for caching}
- **Must avoid:** {e.g., No ORM — use raw SQL, No GraphQL}
- **Compatibility:** {e.g., Must support Node.js 18+, browsers last 2 versions}

## Design Principles (Priority Order)
1. {e.g., Simplicity — prefer boring technology}
2. {e.g., Correctness — no silent failures}
3. {e.g., Maintainability — new dev productive in 1 day}
4. {e.g., Performance — p99 latency <200ms}
5. {e.g., Security — zero trust, validate everything}

## Component Boundaries
```
{component-1}/    # {responsibility}
{component-2}/    # {responsibility}
{component-3}/    # {responsibility}
shared/           # {shared utilities, types}
```

### Communication Rules
- {e.g., Components communicate through defined interfaces only}
- {e.g., No circular dependencies between components}
- {e.g., Shared types in shared/ — no type duplication}

## Quality Attributes
| Attribute | Priority | Target |
|-----------|----------|--------|
| Scalability | {HIGH/MED/LOW} | {e.g., 1000 concurrent users} |
| Availability | {HIGH/MED/LOW} | {e.g., 99.9% uptime} |
| Security | {HIGH/MED/LOW} | {e.g., OWASP top 10 compliance} |
| Performance | {HIGH/MED/LOW} | {e.g., <200ms API response} |
| Observability | {HIGH/MED/LOW} | {e.g., structured logging, metrics} |

## Decision Record
All architectural decisions MUST be documented as ADRs in `planning-artifacts/`.
Reference existing ADRs before proposing new architecture.

## Customization Instructions
1. Replace all `{placeholder}` values with your project specifics
2. Reorder design principles by your actual priorities
3. Define your real component boundaries
4. Set realistic quality attribute targets
5. Keep total under 80 lines
```

#### Fields to Customize

| Field | User Customizes | Example |
|-------|-----------------|---------|
| System Type | Application category | Web app, API service, CLI tool |
| Architecture Style | Structural pattern | Monolith, microservices, serverless |
| Deployment | Infrastructure | Docker/K8s, Vercel, bare metal |
| Must Use | Hard dependencies | PostgreSQL, Redis, TypeScript |
| Must Avoid | Prohibited patterns | No ORM, no GraphQL, no global state |
| Design Principles | Priority ranking | Simplicity > Correctness > Performance |
| Component Boundaries | Module structure | controllers/, services/, repositories/ |
| Quality Attributes | Non-functional requirements | Scalability, performance, security targets |

#### Example Customization: TodoAPI (Monolithic Node.js Service)

```markdown
# Architecture Principles

## System Type
- **Type:** Web API service (REST)
- **Architecture:** Monolithic (all logic in single Node.js process)
- **Deployment:** Docker container on Kubernetes

## Technology Constraints
- **Must use:** Express.js for routing, PostgreSQL for persistence, TypeScript
- **Must avoid:** No ORM — use pg query builder, No GraphQL, No global state
- **Compatibility:** Node.js 20.x LTS, clients support HTTP/1.1 minimum

## Design Principles (Priority Order)
1. Simplicity — prefer Express + raw SQL over complex frameworks
2. Correctness — explicit error handling, no silent failures
3. Maintainability — new dev can understand code in 1 day
4. Performance — p99 API latency <200ms
5. Security — input validation, SQL injection prevention, OWASP compliance

## Component Boundaries
```
src/
  controllers/       # HTTP request/response handlers
  services/          # Business logic (UserService, TodoService)
  repositories/      # Database access (UserRepository, TodoRepository)
  middleware/        # Auth, logging, error handling
  types/             # TypeScript interfaces
  models/            # Database schema, migrations
  utils/             # Shared helpers
shared/             # Constants, error codes
```

### Communication Rules
- Services receive repositories via dependency injection
- No direct database queries outside repositories
- No circular dependencies (use interfaces when needed)
- Shared error codes defined in utils/error-codes.ts
- Controllers call services, services call repositories only

## Quality Attributes
| Attribute | Priority | Target |
|-----------|----------|--------|
| Scalability | HIGH | 1000 concurrent users |
| Availability | HIGH | 99.9% uptime |
| Security | HIGH | OWASP top 10 compliance |
| Performance | MEDIUM | p99 latency <200ms |
| Observability | MEDIUM | Structured JSON logging |

## Decision Record
All architectural decisions documented as ADRs in `planning-artifacts/`.
See `planning-artifacts/adr-*.md` for technology choices and major changes.
```

#### How the Skill Is Used

- **Architect** reads this when making design decisions (CLAUDE.md Step 3: Match Agent → architect for complex tasks)
- **Planner** references this when defining feature scope and constraints
- **Implementer** uses this to understand allowed patterns and boundaries
- **Reviewer** checks conformance to these principles

---

## Part 3: Evidence from CLAUDE.md and Architecture

### CLAUDE.md Supports Placeholder Design

From CLAUDE.md Line 6:

> **"Customization placeholder skills (Required)"**
> "Customize these files for your stack:"
> ```bash
> .claude/skills/coding-standards.md
> .claude/skills/review-checklist.md
> .claude/skills/testing-strategy.md
> .claude/skills/architecture-principles.md
> ```

This is explicit: the template expects users to customize these four skills.

### README.md Validates Placeholder Design

From README.md Section "How It Works" (lines 214-230):

> "The template provides a practical operating model where:
> - A **Main Agent** (`CLAUDE.md`) acts as a stateless dispatcher
> - **Specialized subagents** handle research, planning, architecture, implementation, review, and testing
> - **Skills** provide reusable knowledge loaded on demand"

And from Quick Start Section (lines 97-106):

> "### 2. Customize placeholder skills (Required)"
> "Customize these files for your stack:"
>
> This template is **project-agnostic**. It does not bake in assumptions about:
> - Programming language or framework
> - Code style or naming conventions
> - Test framework or CI/CD pipeline
> - Architectural style or deployment model

### Skill Loading Mechanism

From CLAUDE.md, the dispatch loop reads skills on demand:

```markdown
## Dispatch Loop

### 3. Match Agent
- Compare task against agent descriptions in .claude/agents/
- Agents list required skills in `setting_sources` field
```

Each agent definition (researcher.md, planner.md, etc.) references which skills it loads. This means:

1. The implementer agent loads `coding-standards.md` when writing code
2. The reviewer agent loads `review-checklist.md` during code review
3. The tester agent loads `testing-strategy.md` when running tests
4. The architect agent loads `architecture-principles.md` when making decisions

The skills are **loaded at runtime** per agent's needs, not baked into the template.

---

## Part 4: Governance Cascade (spec-protocol.md Section 5)

From spec-protocol.md Section 5 ("Constraints & Governance"), the Governance Cascade is defined:

```markdown
### Governance Cascade

When conflicts arise between governance layers, higher-numbered layers lose:

1. **constitution.md** (highest authority — optional, Slice 3)
2. **spec-protocol.md** (this file)
3. **Spec templates** (`.claude/spec-templates/` — optional, Slice 4)
4. **Skills** (coding-standards.md, testing-strategy.md, etc.)
5. **Agent freedom** (lowest — anything not constrained above)
```

The **four placeholder skills sit at Layer 4** in the governance cascade. This means:

- Constitution.md (if present) can override any skill
- spec-protocol.md (mandatory) overrides skills
- Spec templates can guide skill choices
- **Skills inform agent behavior but don't constrain it beyond what spec-protocol.md requires**

This is by design. The skills layer is meant to be **customizable and project-specific**, while higher layers provide **immutable governance** for all projects.

---

## Part 5: Customization Guidance for Users

### How to Customize These Skills for Your Project

#### Step 1: Identify Your Stack

Before opening any skill file, gather basic facts about your project:

```
1. What programming language(s) do you use?
   → Answer: TypeScript, Python, Go, Rust, Java, etc.

2. What framework(s) are you using?
   → Answer: Express, FastAPI, Spring, Django, etc.

3. What is your project type?
   → Answer: Web app, API service, CLI tool, mobile backend, etc.

4. What test framework do you use?
   → Answer: Jest, Pytest, Go testing, Mocha, etc.

5. What is your deployment model?
   → Answer: Docker/K8s, Vercel, AWS Lambda, bare metal, etc.

6. What are your architectural constraints?
   → Answer: Monolith, microservices, serverless, CQRS, etc.
```

#### Step 2: Customize Each Skill File

**File: `.claude/skills/coding-standards.md`**

1. Replace the "Language & Framework" section with your actual stack
2. Update "Naming Conventions" with your project's rules (camelCase vs snake_case, etc.)
3. Replace linter/formatter commands with your actual tools
4. List 3-5 design patterns you actually use (not just examples)
5. List 3-5 anti-patterns you want to prevent
6. Draw your actual file structure under "File Organization"

**File: `.claude/skills/testing-strategy.md`**

1. Replace "Test Framework" with your actual testing tool and library
2. Update all test commands to match your project's scripts
3. Create your actual directory structure under "Test Organization"
4. Update naming conventions to match your test files (e.g., `*.test.ts` vs `test_*.py`)
5. Set realistic coverage thresholds for your project
6. Document your actual CI/CD pipeline

**File: `.claude/skills/review-checklist.md`**

1. Replace automated check commands with the exact commands you run locally
2. Add or remove checklist items based on your project's risk profile
3. Adjust severity definitions to match your team's risk tolerance
4. Add project-specific checks (e.g., database migration validation, security headers)

**File: `.claude/skills/architecture-principles.md`**

1. Describe your system's actual type and architecture
2. List your hard constraints (database choice, framework, etc.)
3. Reorder design principles by your actual project priorities
4. Draw your actual component boundaries
5. Set realistic targets for quality attributes (scalability, performance, security)

#### Step 3: Validate

After customization, verify that:

```bash
# Check that all placeholders are replaced
grep -r "{" .claude/skills/

# Should return NO matches. If matches exist, fill them in.
```

#### Step 4: Commit to Git

Once customized, commit these skills to version control:

```bash
git add .claude/skills/
git commit -m "customize: skills for [your project name]"
```

These files are now part of your project's durable state and will survive context windows.

---

## Part 6: Complete Example: TodoAPI Project

To illustrate, here's a complete example of how a hypothetical **TodoAPI** project would customize all four skills.

### TodoAPI Stack
- **Language:** TypeScript
- **Framework:** Express.js
- **Database:** PostgreSQL
- **Testing:** Jest
- **Deployment:** Docker on Kubernetes

### TodoAPI Customized Skills

#### 1. `coding-standards.md` (TodoAPI)

```markdown
# Coding Standards

## Language & Framework
- **Language:** TypeScript 5.x
- **Framework:** Express.js
- **Runtime:** Node.js 20.x

## Naming Conventions
- **Files:** kebab-case.ts (user-controller.ts, todo-service.ts)
- **Functions:** camelCase (createUser, validateTodoText)
- **Classes:** PascalCase (UserController, TodoService)
- **Constants:** UPPER_SNAKE_CASE (MAX_TODOS_PER_USER, API_TIMEOUT_MS)
- **Variables:** camelCase (userId, isActive)

## Code Style
- **Linter:** ESLint + @typescript-eslint
- **Formatter:** Prettier (80 chars, single quotes)
- **Lint command:** `npm run lint`
- **Format command:** `npm run format`

## Patterns to Follow
1. Dependency injection (tsyringe decorators)
2. Repository pattern for data access
3. AppError class for all error handling

## Patterns to Avoid
1. No global mutable state
2. No `any` types
3. No raw SQL queries

## File Organization
```
src/
  controllers/    # HTTP handlers
  services/       # Business logic
  repositories/   # Data access
  middleware/     # Auth, logging
  types/          # Interfaces
```
```

#### 2. `testing-strategy.md` (TodoAPI)

```markdown
# Testing Strategy

## Test Framework
- **Framework:** Jest 29.x
- **Assertion library:** Jest built-in expect()
- **Mocking:** jest.mock()

## Test Commands
```bash
npm test                          # All tests
npm test -- --watch             # Watch mode
npm run test:coverage           # Coverage report
npx jest src/services/           # Single directory
```

## Test Organization
```
tests/
  unit/
    services/
      user-service.test.ts
      todo-service.test.ts
  integration/
    api/
      todos.integration.test.ts
  fixtures/
    test-data.ts
```

## Naming Conventions
- **Test files:** `*.test.ts`
- **Test names:** "should {behavior} when {condition}"
- **Describe blocks:** describe('{ServiceName}', ...)

## Coverage Requirements
- **Minimum:** 80% overall
- **Critical:** 100% for auth middleware
- **New code:** Must have tests before merge

## CI/CD Integration
- Tests run on every push and PR
- Failures block merge to main
```

#### 3. `review-checklist.md` (TodoAPI)

```markdown
# Code Review Checklist

## Automated Checks
- [ ] Linter passes: `npm run lint`
- [ ] TypeScript: `npx tsc --noEmit`
- [ ] Tests pass: `npm test`
- [ ] Build succeeds: `npm run build`

## Code Quality

### Correctness
- [ ] Code matches task assertions
- [ ] Null/empty checks present
- [ ] Uses AppError with proper codes
- [ ] No console.log (use logger)

### Readability
- [ ] Kebab-case files, camelCase functions
- [ ] Functions <= 30 lines
- [ ] Comments explain WHY

### Architecture
- [ ] Uses DI pattern (no global services)
- [ ] Uses Repository pattern
- [ ] Changes in declared file_scope

### Testing
- [ ] Unit tests present
- [ ] Happy path + 2 edge cases per function
- [ ] Test names describe behavior

### Security
- [ ] No .env, API keys, or tokens
- [ ] Input validated via middleware
- [ ] No raw SQL
- [ ] Auth checks on protected routes

### Performance
- [ ] No N+1 queries
- [ ] Lists paginated (default 50)
- [ ] Async never blocks event loop

## Severity Guidelines
- **CRITICAL:** Security leak, auth bypass, crash on null
- **MAJOR:** Missing tests, missing error handling
- **MINOR:** Naming clarity, comments
```

#### 4. `architecture-principles.md` (TodoAPI)

```markdown
# Architecture Principles

## System Type
- **Type:** REST API service
- **Architecture:** Monolith (single Node.js process)
- **Deployment:** Docker on Kubernetes

## Technology Constraints
- **Must use:** PostgreSQL, Express.js, TypeScript, pg query builder
- **Must avoid:** No ORM, no GraphQL, no global state
- **Compatibility:** Node.js 20.x LTS

## Design Principles (Priority Order)
1. Simplicity — Express + raw SQL, not complex framework
2. Correctness — explicit error handling, no silent failures
3. Maintainability — new dev productive in 1 day
4. Performance — p99 latency <200ms
5. Security — OWASP top 10

## Component Boundaries
```
src/
  controllers/    # HTTP layer
  services/       # Business logic
  repositories/   # Database layer
  middleware/     # Cross-cutting concerns
  types/          # Shared types
```

## Quality Attributes
| Attribute | Priority | Target |
|-----------|----------|--------|
| Scalability | HIGH | 1000 concurrent users |
| Availability | HIGH | 99.9% uptime |
| Security | HIGH | OWASP compliance |
| Performance | MEDIUM | p99 <200ms |
| Observability | MEDIUM | Structured logging |
```

---

## Part 7: Differences from Complete Examples

### Not Placeholders: Spec Templates

The template includes **reusable spec templates** in `.claude/spec-templates/`:

- `rest-crud-endpoint.yaml` — Complete example for REST endpoints
- `auth-flow.yaml` — Complete example for authentication
- `data-pipeline.yaml` — Complete example for data processing

These are **NOT placeholders**. They are:
- **Fully populated examples** with concrete assertion patterns
- **Optional reusable patterns** (planner may use them as starting points)
- **Reference implementations** showing how to write good specs

The spec templates exist in `.claude/spec-templates/` (not `.claude/skills/`) because they serve a different purpose: providing reference patterns, not project-specific rules.

### Not Placeholders: spec-protocol.md

The file `.claude/skills/spec-protocol.md` is **NOT a placeholder**. It is:
- A **complete, immutable specification** of how SDD works
- Defines the format for all specs across all projects
- Not project-specific — applies the same way to every project using SDD
- Over 1400 lines of governance rules and examples

The only "customization" for spec-protocol.md is **reading it to understand how to use SDD**. No user modifies it.

---

## Part 8: Why This Design Was Chosen

### Principle: Project Agnosticism

From README.md, Design Principle #1:

> "**Stateless Dispatcher** - Main Agent stores no durable state in memory"

This principle extends to project knowledge. The template provides:
- **Fixed orchestration** (CLAUDE.md — works the same for all projects)
- **Fixed governance** (spec-protocol.md — works the same for all projects)
- **Customizable context** (skills — different for each project)

If the template hard-coded coding standards for TypeScript/Express, it would be useless for Python/FastAPI projects. Instead, it provides **structure** (what information to include) and asks users to provide **content** (values for your specific project).

### Principle: Token Efficiency

From CLAUDE.md, Line 9:

> "**Token discipline:** Compact at 80k tokens; target <128k total from start to finish"

Each skill is kept **under 80 lines** specifically so it fits into agent context when loaded. If the template tried to include standards for 10 different languages/frameworks, the skill would be 800 lines — unusable.

Instead, each project **loads only the standards relevant to their stack**, keeping context efficient.

### Principle: Immediate Usefulness

A project-agnostic template is only useful if it's customized for your actual project. The template **requires customization** to be useful, so it flags the four skills as `[PLACEHOLDER]` and includes explicit instructions to customize them.

This is better than:
- Shipping with generic/useless standards (ignored by users)
- Shipping with TypeScript/Express examples (confuses users with other stacks)
- Shipping with no standards at all (leaves agents without guidance)

---

## Part 9: Summary: Why These Are Intentional Placeholders

| Reason | Explanation |
|--------|-------------|
| **Project Agnosticism** | Template works for any language/framework/architecture, so these skills cannot be universal |
| **Token Efficiency** | Skills are kept <80 lines for efficient context loading; can't include all possible standards |
| **Immediate Usefulness** | Project must customize skills for template to work; placeholder status forces this |
| **Governance Cascade** | Skills sit at Layer 4 in governance cascade; lower than constitution/spec-protocol, can be project-specific |
| **CLAUDE.md Requirement** | README explicitly marks them as required customization |
| **Agent Loading** | Each agent loads only the skills it needs; customization lets agents read relevant guidance |
| **Durable State** | Skills are version-controlled files, surviving context windows; customization anchors project identity |

---

## Validation Checklist

After customizing your skills, validate:

```markdown
## Customization Validation

- [ ] All `{placeholders}` in coding-standards.md are replaced with actual values
- [ ] All `{placeholders}` in testing-strategy.md are replaced with actual values
- [ ] All `{placeholders}` in review-checklist.md are replaced with actual values
- [ ] All `{placeholders}` in architecture-principles.md are replaced with actual values

- [ ] coding-standards.md matches your actual language/framework/runtime
- [ ] testing-strategy.md commands work locally (e.g., `npm test`, `pytest`)
- [ ] review-checklist.md reflects your project's quality standards
- [ ] architecture-principles.md describes your actual system design

- [ ] Each skill file is under 80 lines (context efficiency)
- [ ] Each skill file is committed to git
- [ ] No skill file has unresolved `{placeholder}` text
```

---

## References

1. **CLAUDE.md** — Main orchestration kernel, dispatches agents, loads skills
2. **README.md** — Project intent, design principles, quick start with customization requirement
3. **spec-protocol.md Section 5** — Governance Cascade showing skills at Layer 4
4. **`.claude/agents/_agent-template.md`** — Agent definition showing `setting_sources` field (skills loaded per agent)
5. **`.claude/spec-templates/`** — Reference examples (not placeholders) of reusable spec patterns

---

## Conclusion

The four placeholder skills are **intentional design artifacts**. They are:

1. **Required** for the template to be useful (project must define its own standards)
2. **Project-specific** (cannot be universal across all possible stacks)
3. **Token-efficient** (kept short to fit in agent context)
4. **Governance-positioned** (Layer 4 in cascade, below immutable protocol)
5. **Durable** (version-controlled files surviving context windows)
6. **Customizable** (explicit instructions and examples provided)

They are **not incomplete**. They are **intentionally unpopulated**, awaiting customization by each project using the template.

This design ensures the template remains project-agnostic while requiring projects to define their own operating standards — a deliberate balance between template generality and project specificity.
