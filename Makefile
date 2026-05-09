.PHONY: all fmt lint test test-cov run report ci clean help

all: lint test

help:
	@echo "make fmt      - format and auto-fix"
	@echo "make lint     - run linter"
	@echo "make test     - run tests"
	@echo "make test-cov - run tests with coverage (≥80% gate)"
	@echo "make run      - run sample analysis"
	@echo "make report   - generate report"
	@echo "make ci       - lint + test-cov + report"
	@echo "make clean    - remove generated files"

fmt:
	uv run ruff format .
	uv run ruff check . --fix

lint:
	uv run ruff check .
	uv run ruff format --check .

test:
	uv run pytest -q

test-cov:
	uv run pytest -q --tb=short --cov=src --cov-report=term-missing --cov-fail-under=80

run:
	uv run python -m src.main --input examples/sample.jsonl --output output/sample_result.json

report:
	uv run python -m src.report

ci: lint test-cov report

clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
	rm -f output/*.json output/*.csv output/*.png
