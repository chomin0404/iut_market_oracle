# CLAUDE.md

See @README.md for project overview and usage, @pyproject.toml for package/tool settings, @Makefile for canonical commands, and @configs/dependency_edges.yaml for editable dependency structure.

## Purpose
- Maintain a research-grade end-to-end pipeline: **math spec (YAML) → verify → skeleton code → trace → audit → report → CI**.
- Every model must exist first as a registry YAML; code without a verified YAML spec is prohibited.
- Prefer minimal diffs, deterministic behavior, and reproducible outputs.

## Rules
- Explore first, then plan, then edit.
- Keep changes as small as possible.
- Update tests when behavior, counts, scoring, CLI defaults, or report output changes.
- Prefer YAML/TOML/Makefile/workflow edits before Python rewrites when possible.
- Use Python 3.11+ compatible code.

## Verify
- `make lint`
- `make test-cov`
- `make report`
- `make ci`

## Done
- Relevant tests updated.
- Lint, coverage gate, and report generation all pass.
- User-facing command or config changes are reflected in docs.

---

## ModelForge Governance

### Single source of truth
`configs/model_registry/<id>.yaml` is the canonical math spec.
No model exists in code unless its YAML passes `/modelforge-verify`.

### Traceability chain (immutable)
```
YAML spec → VerificationReport → impl_skeleton.py → TraceNode → AuditEntry
```
Every artifact carries its parent's SHA-256 hash.
Stored in `artifacts/modelforge/trace.jsonl` (append-only).

### Audit log
All forge events are recorded in `.claude/audit/modelforge.jsonl`.
Scope: registry edits, forge runs, verification runs.
Rotation policy: same as config-changes log (1 MB, 10 archives, gzip).

### Automation (Cloud Functions pattern)
`src/models/automation.py` exposes event handlers:
- `registry_changed` — fired automatically by hook on YAML edit
- `forge_requested`  — invoked by `/modelforge-run`
- `verify_requested` — invoked by `/modelforge-verify`

### Rules
- Never bypass verification: do not write to `artifacts/modelforge/` manually.
- Never edit `trace.jsonl` retroactively (append-only invariant).
- Adding a new model = add YAML first, run `/modelforge-run <id>`, then write code.
- A model's YAML `id` must match its filename stem exactly.

### Full pipeline (end-to-end)
```
1. Author  configs/model_registry/<id>.yaml    (math spec — single source of truth)
2. Verify  /modelforge-verify <id>             (7 static checks, no artifacts)
3. Forge   /modelforge-run <id>                (verify + skeleton + trace + audit)
4. Code    src/<module>/<id>.py                (human/LLM implementation, guided by skeleton)
5. Test    uv run pytest tests/                (all tests must pass)
6. Report  /modelforge-report <id>             (markdown report from forge artifacts)
7. CI      make ci                             (lint + test-cov + report)
```

Hook automation:
- Edit/Write `configs/model_registry/*.yaml` → `on-registry-change.sh` auto-runs verify + forge (PostToolUse)
- Edit/Write `*.py` → `post-edit-python-check.sh` auto-runs ruff + targeted pytest (PostToolUse)
- Before stop → `stop-verify.sh` gates on lint + test-cov state markers

### Commands
| Command | Action |
|---|---|
| `/modelforge-run <id\|all>` | Full pipeline: verify → skeleton → trace → audit |
| `/modelforge-verify <id\|all>` | Static verification only, no artifacts |
| `/modelforge-trace <id>` | Show DAG for a model |
| `/modelforge-audit` | Tail audit log |
| `/modelforge-report <id\|all>` | Generate markdown report from forge artifacts |

---

## Project Overview

This repository builds GNSS Resilience Twin as a testable, explainable mathematical MVP.
The system ingests observation logs and returns anomaly analysis with traceable outputs.

Primary goals:
- detect spoofing-related anomalies from observation sequences
- produce explainable outputs, not only binary alarms
- preserve reproducibility and traceability
- support CLI-driven experiments and report generation

---

## Core Workflow

Always follow this order:

1. Explore
2. Plan
3. Implement
4. Verify
5. Summarize

Do not skip directly to implementation unless the user explicitly asks for a tiny local edit.

---

## Explore Rules

Before writing code:
- inspect relevant files
- summarize current behavior
- identify dependencies and data flow
- list unclear assumptions
- check whether similar logic already exists

When the task is ambiguous:
- ask a clarifying question, or
- propose 2-3 interpretations and choose the smallest safe one

---

## Plan Rules

Before coding, provide:
- target files
- intended behavior change
- tests to add or update
- acceptance criteria
- risks and tradeoffs

Planning principles:
- prefer the smallest viable diff
- prefer schema-first design
- prefer config-driven behavior when appropriate
- avoid broad refactors unless explicitly requested
- keep unrelated files unchanged

---

## Implement Rules

When implementing:
- change only files relevant to the approved plan
- keep functions focused and names explicit
- avoid hidden state and magic constants
- preserve traceability fields such as run_id, config, and outputs
- prefer deterministic behavior when possible
- do not invent unsupported claims, metrics, or pseudo-results

For mathematical logic:
- keep formulas and assumptions explicit
- separate model logic from I/O and CLI glue
- prefer modular components over monolithic scripts

---

## Verify Rules

Every meaningful change must include at least one of:
- unit test
- integration test
- reproducible CLI run
- saved output artifact

Verification checklist:
- does it match the requested scope?
- do tests pass?
- does the output schema remain stable?
- are artifacts saved under output/ when appropriate?
- are risks and limitations stated clearly?

If verification fails:
- explain the failure before revising
- do not claim success without evidence

If three implementation attempts fail:
- stop coding
- switch to architecture review mode
- propose a smaller task split

---

## Output and Reporting

When completing a task, always report:
- changed files
- tests run
- outputs generated
- remaining risks
- next recommended step

When generating artifacts:
- save final outputs under output/
- use stable, descriptive filenames
- prefer machine-readable formats such as JSON and CSV

---

## Coding Style

- Prefer readability over cleverness.
- Prefer explicit schemas over ad-hoc dicts.
- Prefer small modules with clear responsibilities.
- Keep comments short and structural.
- Avoid premature abstraction.
- Do not silently change public behavior.

---

## Repository Priorities

Priority order:
1. correctness
2. verifiability
3. traceability
4. explainability
5. extensibility
6. optimization

Do not sacrifice correctness and verification for speed.

---

## GNSS-Specific Guidance

For GNSS anomaly tasks:
- separate genuine and spoofing scenarios
- track false alarm rate and detection delay
- preserve score components and reasons
- record config, seed, and run_id
- prefer explainable alarm logic over opaque heuristics when possible

Typical outputs may include:
- risk_score
- alarm
- suspicious_subset
- score_components
- reasons
- metrics summary

---

## Prompting Pattern

Use this default pattern unless instructed otherwise:

Task:
[one-sentence task]

First do not code.
1. Explore relevant files and summarize current behavior.
2. Propose the smallest implementation plan.
3. List target files, tests, and acceptance criteria.
4. Then implement only the approved scope.
5. Verify with tests and/or reproducible execution.
6. Summarize changed files, outputs, and remaining risks.
