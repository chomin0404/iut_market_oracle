---
description: ModelForge — show traceability chain for a model
argument-hint: <model_id>
---
Display the full traceability graph for one model.

Model ID: $ARGUMENTS

Steps:
1. Run `uv run python -m src.models trace $ARGUMENTS`.
2. Display the DAG: for each TraceNode show type, node_id, parent_ids, artifact_path, hash prefix.
3. Explain the chain:
   - REGISTRY → source YAML spec
   - VERIFICATION → static check report
   - GENERATED_CODE → skeleton implementation stub
   - AUDIT_ENTRY → audit log entry
4. If no nodes found, report that the model has not been forged yet and suggest running `/modelforge-run $ARGUMENTS`.

Do not edit any file.
