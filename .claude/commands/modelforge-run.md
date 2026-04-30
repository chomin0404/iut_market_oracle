---
description: ModelForge — YAML spec → verify → skeleton code → trace → audit
argument-hint: <model_id|all>
---
Run the full ModelForge pipeline for one model or the entire registry.

Model ID: $ARGUMENTS

Steps:
1. Read `configs/model_registry/$ARGUMENTS.yaml` (or all YAMLs if "all").
2. Run `uv run python -m src.models run $ARGUMENTS` to execute:
   - Static verification (7 checks)
   - Skeleton code generation
   - TraceNode append to `artifacts/modelforge/trace.jsonl`
   - Audit entry append to `.claude/audit/modelforge.jsonl`
3. Report verification status and artifact paths.
4. If verification overall = FAIL, list the failing checks and stop.
5. If overall = WARN, list warnings but continue.

Do not edit any source file. Only run the command above and report results.
