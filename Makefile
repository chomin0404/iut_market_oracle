.PHONY: all fmt lint test test-cov run report ci clean clean-runs help

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
	@echo "make clean-runs [DAYS=7] - delete output/ run dirs older than DAYS days"

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
	uv run python -m src.gnss --n-mc 40 --seed 42 --out output/gnss_sample.json

report:
	uv run python -m src.report

ci: lint test-cov report

clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
	rm -f output/*.json output/*.csv output/*.png

DAYS ?= 7
clean-runs:
	uv run python -c "from gnss.persistence import purge_old_runs; n = purge_old_runs(max_age_days=$(DAYS)); print(f'Deleted {n} run(s) older than $(DAYS) day(s).')"
