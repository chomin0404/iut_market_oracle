---
description: Generate a reproducibility report for the current session's changes — changed files, test results, artifacts, risks, and next step
argument-hint: [optional: experiment id or task tag]
---

You are generating an experiment reproducibility report.
Do not start new implementation.
Summarize what was done, what was verified, and what remains.

Session context:
$ARGUMENTS

Follow this exact process.

# 1. Session summary
In 2-3 sentences, describe:
- what task was attempted
- what was implemented
- what was verified

# 2. Changed files
List all files modified, created, or deleted in this session.
For each file, state:
- path
- change type (created / modified / deleted)
- one-line description of what changed

# 3. Test results
Report:
- test command(s) run
- number of tests passed / failed / skipped
- coverage if available
- any test failures and their cause

# 4. Artifacts
List any output artifacts saved in this session:
- path under output/ or artifacts/
- content type (JSON, CSV, PNG, JSONL, etc.)
- what it contains
- whether it is reproducible from config + seed

# 5. Reproducibility check
Confirm:
- random seeds used (if any)
- config files referenced
- run_id or experiment_id assigned
- whether the same result can be reproduced from the same inputs

# 6. Traceability chain
Describe the chain from input to output:
- config / YAML spec
- code path
- output artifact
- audit or trace record (if applicable)

# 7. Risks and limitations
List:
- known gaps or missing tests
- edge cases not yet covered
- assumptions that have not been validated
- anything that could break silently

# 8. Next recommended step
State exactly one next action.
Be specific: file to change, test to write, or question to resolve.

# 9. Final output format
Respond using this structure:

## Session summary
## Changed files
## Test results
## Artifacts
## Reproducibility
## Traceability
## Risks and limitations
## Next step

Rules:
- Do not claim success without test evidence.
- If tests were not run, say so explicitly.
- If an artifact was not saved, say so explicitly.
- Distinguish verified facts from assumptions.
