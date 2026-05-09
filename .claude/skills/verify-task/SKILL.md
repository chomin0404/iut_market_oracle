---
description: Verify a completed task against scope, tests, reproducibility, and explainability
argument-hint: [task or diff summary]
---

You are in verification mode.
Assume implementation may be incomplete or incorrect.
Be skeptical and evidence-driven.

Task / change summary:
$ARGUMENTS

Follow this exact process.

# 1. Scope check
Determine:
- intended scope
- what was actually changed
- whether unrelated changes were introduced

# 2. Contract check
Inspect:
- input schema changes
- output schema changes
- CLI/API behavior changes
- backward compatibility risks

# 3. Test check
Review:
- unit tests added or modified
- normal-case coverage
- abnormal-case coverage
- boundary-case coverage
- gaps that remain

# 4. Reproducibility check
Confirm whether the change preserves:
- deterministic execution where expected
- config traceability
- seed traceability
- run_id or experiment traceability
- output artifacts under output/ when relevant

# 5. Metric check
If the task is analytical or model-based, verify:
- metrics are defined
- metrics match the objective
- baseline or prior behavior is available
- no unsupported performance claim is made

# 6. Explainability check
Determine whether the result includes:
- interpretable output
- reasons or score components
- clear failure messaging
- human-readable summary

# 7. GNSS-specific check
If the task touches GNSS anomaly logic, explicitly inspect:
- genuine scenario behavior
- spoofing scenario behavior
- false alarm risk
- detection delay
- suspicious subset or equivalent explanation field
- saved traceability record

# 8. Verdict
Return one of:
- Accept
- Accept with limitations
- Reject and revise

# 9. Final output format
Respond using this structure:

## Scope alignment
## Contract stability
## Test status
## Reproducibility
## Metrics
## Explainability
## Domain-specific checks
## Risks
## Verdict
## Next revision

Rules:
- Do not assume success without evidence.
- If evidence is missing, mark it missing.
- Distinguish tested facts from plausible expectations.
