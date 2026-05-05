# CLAUDE.md

See @README.md for project overview and usage, @pyproject.toml for package/tool settings, @Makefile for canonical commands, and @configs/dependency_edges.yaml for editable dependency structure.

## Purpose
- Maintain a research-grade end-to-end pipeline: **math spec (YAML) → verify → skeleton code → trace → audit → report → CI**.
- Every model must exist first as a registry YAML; code without a verified YAML spec is prohibited.
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

### Full pipeline (end-to-end)
```
1. Author  configs/model_registry/<id>.yaml    (math spec — single source of truth)
2. Verify  /modelforge-verify <id>             (7 static checks, no artifacts)
3. Forge   /modelforge-run <id>                (verify + skeleton + trace + audit)
4. Code    src/<module>/<id>.py                (human/LLM implementation, guided by skeleton)
5. Test    uv run pytest tests/                (all tests must pass)
6. Report  /modelforge-report <id>             (markdown report from forge artifacts)
7. CI      make ci                             (lint + test-cov + report)
```

Hook automation:
- Edit/Write `configs/model_registry/*.yaml` → `on-registry-change.sh` auto-runs verify + forge (PostToolUse)
- Edit/Write `*.py` → `post-edit-python-check.sh` auto-runs ruff + targeted pytest (PostToolUse)
- Before stop → `stop-verify.sh` gates on lint + test-cov state markers

### Commands
| Command | Action |
|---|---|
| `/modelforge-run <id\|all>` | Full pipeline: verify → skeleton → trace → audit |
| `/modelforge-verify <id\|all>` | Static verification only, no artifacts |
| `/modelforge-trace <id>` | Show DAG for a model |
| `/modelforge-audit` | Tail audit log |
| `/modelforge-report <id\|all>` | Generate markdown report from forge artifacts |
