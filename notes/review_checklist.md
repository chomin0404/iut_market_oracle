# Review Checklist

Use this checklist before merging any change into the main branch.

---

## 1. Tests

```bash
uv run pytest -q
```

- [ ] All tests pass (exit code 0)
- [ ] No tests were silently skipped (check `-v` output if uncertain)
- [ ] New logic has corresponding test coverage

---

## 2. Lint

```bash
uv run ruff check .
```

- [ ] No lint errors
- [ ] No unused imports or variables

---

## 3. Format

```bash
uv run ruff format --check .
```

- [ ] All files are consistently formatted (no diff reported)

---

## 4. Type Check

```bash
uv run pyright
```

- [ ] No type errors in `src/`
- [ ] New functions have type annotations

---

## 5. Security

- [ ] No secrets, API keys, or credentials in source files
- [ ] No writes to `data/raw/` (immutable source data)
- [ ] `.env` and `*.pem` / `*secret*` files are in `.gitignore`
- [ ] `git diff --stat` reviewed for accidental binary or large file inclusions

---

## 6. Reproducibility

- [ ] Random seeds recorded in configs or `ExperimentMeta`
- [ ] Any new numerical constants are named (no magic numbers)
- [ ] Assumption tags applied: `heuristic` / `empirical` / `unproven` / `todo`

---

## 7. Task Tracking

- [ ] `TASKS.md` updated: completed subtasks marked `[x]`
- [ ] Experiment output (if any) linked to an `exp-NNN` registry entry

---

## 8. Documentation

- [ ] Mathematical derivation notes updated in `notes/` if model logic changed
- [ ] `SPEC.md` unchanged or updated for scope changes

---

## Full Sequence (copy-paste)

```bash
uv run pytest -q && \
uv run ruff check . && \
uv run ruff format --check . && \
uv run pyright
```

All four commands must exit 0 before a change is considered review-ready.
