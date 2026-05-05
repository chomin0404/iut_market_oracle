---
description: ModelForge — generate markdown diagnostic report from forge artifacts
argument-hint: <model_id|all>
---
Generate a markdown report for one model or all forged models.

Model ID: $ARGUMENTS

Steps:
1. Run `uv run python -m src.models report $ARGUMENTS`.
2. The report is written to `reports/modelforge/<model_id>.md`.
3. Display the report path and summarise its contents:
   - Math spec (name, equations, solver)
   - Verification results (per-check table)
   - Skeleton code path
   - Traceability chain (REGISTRY → VERIFICATION → GENERATED_CODE → AUDIT_ENTRY)
4. If no forge artifacts exist for the model, instruct the user to run `/modelforge-run $ARGUMENTS` first.

Do not edit any source file. Only generate the report and display results.
