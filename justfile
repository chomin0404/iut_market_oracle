set shell := ["bash", "-cu"]

default:
  @just --list

# Claude workflow
plan task:
  @echo "Use in Claude Code: /formulate-problem {{task}}"
  @echo "Then: /plan-task {{task}}"

verify task:
  @echo "Use in Claude Code: /verify-task {{task}}"

review task:
  @echo "Use in Claude Code: /review-architecture {{task}}"

report exp summary:
  @echo "Use in Claude Code: /experiment-report {{summary}}"

# Local dev
fmt:
  uv run ruff format .
  uv run ruff check . --fix

lint:
  uv run ruff check .

test:
  uv run pytest -q

test-cov:
  uv run pytest --cov=src --cov-report=term-missing

run-example:
  uv run python -m src.gnss --n-mc 40 --seed 42 --out output/gnss_sample.json

clean:
  rm -rf .pytest_cache .ruff_cache .coverage htmlcov output/*.json output/*.csv output/*.png
