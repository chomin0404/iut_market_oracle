---
description: Formalize an ambiguous research idea into a structured problem definition with inputs, outputs, constraints, and acceptance criteria
argument-hint: [idea or problem description]
---

You are in problem formulation mode.
Do not write code yet.
Do not propose implementation options yet.

Idea / problem:
$ARGUMENTS

Follow this exact process.

# 1. Restate
Restate the idea in one sentence using: input -> transformation -> output form.
If the idea is too vague to restate, list the 2 most likely interpretations and ask.

# 2. Scope boundary
Define:
- what is in scope for this formulation
- what is explicitly out of scope
- what is deferred (could be in scope later)

# 3. Inputs
Specify:
- primary observations or data
- auxiliary signals
- format and schema (if known)
- sampling assumptions
- missing data or noise assumptions

# 4. Outputs
Specify:
- primary output fields and types
- secondary/optional output fields
- output schema or format
- traceability fields required (run_id, config, seed, etc.)

# 5. States and structure
Identify:
- hidden states (what cannot be directly observed)
- observable indicators
- normal hypothesis
- abnormal hypothesis
- key structural assumptions

# 6. Constraints
List:
- physical or mathematical constraints
- computational constraints
- explainability requirements
- deployment or operational constraints
- reproducibility requirements

# 7. Success and failure conditions
Define:
- success condition (observable, measurable)
- failure condition (what indicates the formulation is wrong)
- minimum acceptable output
- metrics to evaluate

# 8. Candidate approaches
Propose 2-3 candidate models or methods.
For each, state:
- approach summary
- key assumption
- why it fits
- why it might fail

Choose one and justify the choice.

# 9. Verification plan
Before implementation, specify:
- unit tests to add
- integration checks
- baseline or reference to compare against
- output artifacts to save
- failure modes to inspect

# 10. Final output format
Respond using this structure:

## Problem statement
## Scope
## Inputs
## Outputs
## States and structure
## Constraints
## Success condition
## Failure condition
## Candidate approaches
## Chosen approach
## Verification plan
## Open questions

Rules:
- Do not invent unsupported claims.
- Do not assume data format unless stated.
- Flag every structural assumption explicitly.
- If critical information is missing, ask before proceeding.
- Do not implement yet.
