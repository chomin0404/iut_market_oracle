"""Tests for src/report.py — report generation pipeline."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from report import (
    _results_csv,
    _scenario_comparison_chart,
    _sensitivity_tornado_chart,
    _summary_markdown,
    run_report,
)
from schemas import ScenarioResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SCENARIO_DIR = Path(__file__).parent.parent / "configs" / "scenarios"


def _make_result(name: str, value: float) -> ScenarioResult:
    sensitivity = {
        "initial_revenue": 10.0,
        "revenue_growth": 500.0,
        "ebit_margin": 300.0,
        "tax_rate": -200.0,
        "capex_rate": -100.0,
        "discount_rate": -400.0,
        "terminal_growth_rate": 150.0,
    }
    return ScenarioResult(
        scenario_name=name,
        value=value,
        sensitivity=sensitivity,
        assumption_version="test-v1",
    )


@pytest.fixture()
def three_results() -> list[ScenarioResult]:
    return [
        _make_result("bear", 4000.0),
        _make_result("base", 6000.0),
        _make_result("bull", 9000.0),
    ]


# ---------------------------------------------------------------------------
# _scenario_comparison_chart
# ---------------------------------------------------------------------------


class TestScenarioComparisonChart:
    def test_creates_png(self, tmp_path: Path, three_results):
        out = tmp_path / "comparison.png"
        _scenario_comparison_chart(three_results, out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_single_scenario(self, tmp_path: Path):
        out = tmp_path / "single.png"
        _scenario_comparison_chart([_make_result("base", 5000.0)], out)
        assert out.exists()


# ---------------------------------------------------------------------------
# _sensitivity_tornado_chart
# ---------------------------------------------------------------------------


class TestSensitivityTornadoChart:
    def test_creates_png(self, tmp_path: Path, three_results):
        out = tmp_path / "tornado.png"
        _sensitivity_tornado_chart(three_results[1], out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_empty_sensitivity_does_not_create_file(self, tmp_path: Path):
        result = ScenarioResult(
            scenario_name="base",
            value=5000.0,
            sensitivity={},
            assumption_version="v0",
        )
        out = tmp_path / "tornado.png"
        _sensitivity_tornado_chart(result, out)
        assert not out.exists()

    def test_top_n_respected(self, tmp_path: Path):
        out = tmp_path / "tornado_top2.png"
        _sensitivity_tornado_chart(_make_result("base", 5000.0), out, top_n=2)
        assert out.exists()


# ---------------------------------------------------------------------------
# _results_csv
# ---------------------------------------------------------------------------


class TestResultsCsv:
    def test_header_row(self, three_results):
        csv_text = _results_csv(three_results)
        reader = csv.DictReader(csv_text.splitlines())
        assert "scenario" in reader.fieldnames
        assert "ev_jpy_millions" in reader.fieldnames
        assert "dEV_dinitial_revenue" in reader.fieldnames

    def test_row_count(self, three_results):
        csv_text = _results_csv(three_results)
        rows = list(csv.DictReader(csv_text.splitlines()))
        assert len(rows) == 3

    def test_ev_values_correct(self, three_results):
        csv_text = _results_csv(three_results)
        rows = {
            r["scenario"]: float(r["ev_jpy_millions"])
            for r in csv.DictReader(csv_text.splitlines())
        }
        assert rows["bear"] == pytest.approx(4000.0)
        assert rows["base"] == pytest.approx(6000.0)
        assert rows["bull"] == pytest.approx(9000.0)

    def test_nan_for_missing_sensitivity(self):
        result = ScenarioResult(
            scenario_name="base",
            value=5000.0,
            sensitivity={"discount_rate": -300.0},  # only one param
            assumption_version="v0",
        )
        csv_text = _results_csv([result])
        rows = list(csv.DictReader(csv_text.splitlines()))
        assert rows[0]["dEV_dinitial_revenue"] == "nan"


# ---------------------------------------------------------------------------
# _summary_markdown
# ---------------------------------------------------------------------------


class TestSummaryMarkdown:
    def test_contains_exp_id(self, three_results, tmp_path):
        from datetime import UTC, datetime

        md = _summary_markdown(three_results, "exp-001", tmp_path, datetime.now(UTC))
        assert "exp-001" in md

    def test_contains_all_scenario_names(self, three_results, tmp_path):
        from datetime import UTC, datetime

        md = _summary_markdown(three_results, "exp-001", tmp_path, datetime.now(UTC))
        for name in ("bear", "base", "bull"):
            assert name in md

    def test_contains_output_files_section(self, three_results, tmp_path):
        from datetime import UTC, datetime

        md = _summary_markdown(three_results, "exp-001", tmp_path, datetime.now(UTC))
        assert "scenario_comparison.png" in md
        assert "sensitivity_tornado.png" in md
        assert "results_table.csv" in md
        assert "summary.md" in md

    def test_base_sensitivity_section(self, three_results, tmp_path):
        from datetime import UTC, datetime

        md = _summary_markdown(three_results, "exp-001", tmp_path, datetime.now(UTC))
        assert "Top Sensitivities" in md


# ---------------------------------------------------------------------------
# run_report (integration)
# ---------------------------------------------------------------------------


class TestRunReport:
    def test_returns_four_artifacts(self, tmp_path):
        reports_dir = tmp_path / "reports"
        experiments_dir = tmp_path / "experiments"
        artifacts = run_report(
            scenario_dir=SCENARIO_DIR,
            reports_dir=reports_dir,
            experiments_root=experiments_dir,
        )
        assert set(artifacts.keys()) == {
            "scenario_comparison",
            "sensitivity_tornado",
            "results_table",
            "summary",
        }

    def test_all_artifact_files_exist(self, tmp_path):
        artifacts = run_report(
            scenario_dir=SCENARIO_DIR,
            reports_dir=tmp_path / "reports",
            experiments_root=tmp_path / "experiments",
        )
        for path in artifacts.values():
            assert path.exists(), f"Missing artifact: {path}"

    def test_artifacts_are_nonempty(self, tmp_path):
        artifacts = run_report(
            scenario_dir=SCENARIO_DIR,
            reports_dir=tmp_path / "reports",
            experiments_root=tmp_path / "experiments",
        )
        for path in artifacts.values():
            assert path.stat().st_size > 0, f"Empty artifact: {path}"

    def test_experiment_registered(self, tmp_path):
        experiments_dir = tmp_path / "experiments"
        run_report(
            scenario_dir=SCENARIO_DIR,
            reports_dir=tmp_path / "reports",
            experiments_root=experiments_dir,
        )
        registry = (experiments_dir / "registry.md").read_text(encoding="utf-8")
        assert "exp-001" in registry

    def test_summary_md_contains_scenario_names(self, tmp_path):
        artifacts = run_report(
            scenario_dir=SCENARIO_DIR,
            reports_dir=tmp_path / "reports",
            experiments_root=tmp_path / "experiments",
        )
        content = artifacts["summary"].read_text(encoding="utf-8")
        for name in ("bear", "base", "bull"):
            assert name in content

    def test_csv_has_three_rows(self, tmp_path):
        artifacts = run_report(
            scenario_dir=SCENARIO_DIR,
            reports_dir=tmp_path / "reports",
            experiments_root=tmp_path / "experiments",
        )
        csv_text = artifacts["results_table"].read_text(encoding="utf-8")
        rows = list(csv.DictReader(csv_text.splitlines()))
        assert len(rows) == 3

    def test_missing_scenario_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            run_report(
                scenario_dir=tmp_path / "nonexistent",
                reports_dir=tmp_path / "reports",
                experiments_root=tmp_path / "experiments",
            )

    def test_artifacts_under_exp_subdir(self, tmp_path):
        reports_dir = tmp_path / "reports"
        artifacts = run_report(
            scenario_dir=SCENARIO_DIR,
            reports_dir=reports_dir,
            experiments_root=tmp_path / "experiments",
        )
        for path in artifacts.values():
            assert reports_dir in path.parents, f"{path} not under reports_dir"
