---
description: Plan a task before coding using problem formulation and smallest-diff discipline
argument-hint: [task description]
---

You are in planning mode only.
Do not write code yet.

Task:
$ARGUMENTS

Follow this exact process.

# 1. Restate
Restate the task in one sentence using input -> transformation -> output form.

# 2. Clarify
If the task is ambiguous, list up to 3 possible interpretations.
Choose the smallest safe interpretation unless the user explicitly asked for a broader one.

# 3. Explore
Inspect the repository and identify:
- relevant files
- existing related logic
- schemas or interfaces affected
- configs or CLI entrypoints affected
- tests that already cover nearby behavior

# 4. Problem formulation
Produce a compact formulation with:
- objective
- inputs
- outputs
- hidden states or assumptions
- constraints
- success condition
- failure condition

# 5. Options
Propose 2-3 implementation options.
For each option, include:
- approach
- benefits
- risks
- expected file changes

# 6. Choose plan
Select the smallest viable plan.
Explain why it is preferred now.

# 7. Execution plan
Provide:
- target files
- non-target files
- tests to add/update
- CLI/manual verification steps
- artifacts to save under output/ if relevant
- acceptance criteria
- stop condition for today

# 8. Risk review
List:
- unknowns
- edge cases
- what could invalidate the plan

# 9. Final output format
Respond using this structure:

## Task
## Current state
## Problem formulation
## Options
## Chosen plan
## Files to change
## Tests and verification
## Risks
## Acceptance criteria

Reminder:
- Prefer smallest diff.
- Prefer schema-first changes.
- Prefer config-driven behavior when appropriate.
- Do not implement yet.
