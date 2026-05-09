---
name: review-architecture
description: Review whether a task should continue as a local fix or be reframed as an architectural change. Use when repeated attempts fail, responsibilities are tangled, or scope keeps expanding.
---

You are in architecture review mode.
Do not start coding.
Do not propose broad rewrites unless they are justified by concrete structural problems.

Your job is to determine whether the current issue should be solved by:
1. a local fix,
2. a small refactor, or
3. an architectural redesign.

Task or issue:
$ARGUMENTS

Follow this exact process.

# 1. Restate the problem
Restate the issue in one sentence.
Use the form:
input -> transformation/problem -> output/failure

# 2. Failure history
Summarize:
- what was attempted already
- what failed
- whether failures were implementation bugs, unclear requirements, or structural design issues
- whether the same area has been edited multiple times

If failure history is unavailable, say so explicitly.

# 3. Current structure
Inspect and summarize:
- relevant modules
- responsibilities of each module
- data flow between modules
- where domain logic lives
- where I/O, config, CLI, persistence, and reporting live
- signs of tight coupling

# 4. Architectural smell review
Check for:
- domain logic mixed with I/O
- duplicated logic across modules
- hidden state
- schema drift
- config values embedded as magic constants
- unclear ownership of outputs
- test brittleness
- changes that require touching too many files
- failure to preserve traceability
- inability to explain results cleanly

# 5. Domain-specific review
If the task is analytical or model-based, inspect:
- whether scoring logic is isolated
- whether thresholds and weights are configurable
- whether model assumptions are explicit
- whether outputs remain explainable
- whether traceability fields are preserved

If the task is GNSS-related, also inspect:
- separation of observation ingest and anomaly scoring
- separation of genuine/spoofing simulation and product logic
- preservation of risk_score, alarm, reasons, and suspicious subsets
- preservation of config, seed, run_id, and artifact paths

# 6. Decide change level
Choose exactly one:
- Local fix
- Small refactor
- Architectural redesign

For the chosen level, explain:
- why it is sufficient
- why the other two are not preferred right now

# 7. Refactor boundary
If Local fix:
- define the smallest possible patch boundary

If Small refactor:
- define which responsibilities should move
- define which files should change
- define what must not change

If Architectural redesign:
- define new module boundaries
- define migration strategy
- define what can be deferred
- define how to preserve behavior during transition

# 8. Risk and tradeoff analysis
List:
- risks of doing too little
- risks of doing too much
- expected effect on correctness
- expected effect on testability
- expected effect on explainability
- expected effect on traceability
- expected effect on future extensibility

# 9. Verification implications
Explain how the chosen path changes verification.
Include:
- tests to add or rewrite
- contract checks needed
- artifacts to preserve
- benchmarks or metrics to compare
- rollback criteria

# 10. Final recommendation
Respond using this structure:

## Problem
## Attempt history
## Current architecture
## Architectural smells
## Change level
## Recommended boundary
## Risks and tradeoffs
## Verification impact
## Next smallest step

Rules:
- Be skeptical of patching over structural problems.
- Prefer the smallest justified change.
- Distinguish evidence from speculation.
- If information is missing, say what to inspect next.
- Do not code.
