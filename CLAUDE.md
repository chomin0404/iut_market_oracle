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
