---
description: ModelForge — static verification of model registry YAML (no artifacts written)
argument-hint: <model_id|all>
---
Verify one or all model registry entries without writing artifacts.

Model ID: $ARGUMENTS

Steps:
1. Run `uv run python -m src.models verify $ARGUMENTS`.
2. Report each check result with status (PASS / WARN / FAIL) and message.
3. If any check is FAIL, list remediation steps based on the check name:
   - schema_valid       → fix YAML syntax or missing required fields
   - equations_present  → add at least one equation to the `equations:` list
   - solver_specified   → fill in the `solver:` field
   - outputs_present    → add at least one output to `outputs:`
   - parameters_documented → remove empty strings from `parameters:`
   - references_present → add a primary reference to `references:`
   - parameter_in_equations → mention each parameter name in an equation string

Do not write or edit any file. Only verify and report.
