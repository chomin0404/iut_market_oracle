# CLAUDE.md

See @README.md for project overview and usage, @pyproject.toml for package/tool settings, @Makefile for canonical commands, and @configs/dependency_edges.yaml for editable dependency structure.

## Purpose
- Maintain a research-grade pipeline: classification -> graph scoring -> markdown reporting -> CLI -> CI.
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

### Commands
| Command | Action |
|---|---|
| `/modelforge-run <id\|all>` | Full pipeline: verify → skeleton → trace → audit |
| `/modelforge-verify <id\|all>` | Static verification only, no artifacts |
| `/modelforge-trace <id>` | Show DAG for a model |
| `/modelforge-audit` | Tail audit log |
