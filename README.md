# Research Operating System

Reproducible quantitative research platform integrating mathematical reasoning,
Bayesian inference, experiment tracking, and report generation.

## Structure

| Directory       | Purpose                                  |
|-----------------|------------------------------------------|
| `src/`          | Reusable implementation                  |
| `tests/`        | Unit and property tests                  |
| `configs/`      | Priors, scenarios, experiment settings   |
| `notes/`        | Mathematical notes and proof sketches    |
| `papers/`       | Draft manuscripts                        |
| `data/raw/`     | Immutable source data (never modified)   |
| `data/processed/` | Derived datasets                       |
| `experiments/`  | Run-scoped outputs with metadata         |
| `reports/`      | Generated charts, tables, summaries      |
| `notebooks/`    | Exploration only                         |

## Quick Start

```bash
uv sync
uv run pytest -q
uv run python -m src.report
```

## Workflow

See `SPEC.md` for system behavior definition and `TASKS.md` for execution order.

Every reportable result must have a discoverable origin: experiment ID → config → code.
