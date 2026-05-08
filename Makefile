.PHONY: lint test-cov report ci

lint:
	uv run ruff check .
	uv run ruff format --check .

test-cov:
	uv run pytest -q --tb=short --cov=src --cov-report=term-missing --cov-fail-under=80

report:
	uv run python -m src.report

ci: lint test-cov report
