"""Tests for src/gnss/__main__.py CLI entry point."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gnss.__main__ import main
from gnss.resilience_twin import ResilienceTwinConfig, run_resilience_simulation
from schemas import ResilienceTwinReport

# Small n_mc keeps tests fast while still cycling all 4 fault classes.
_FAST_N_MC = 20


class TestRunResilienceSimulation:
    """Unit tests for the core simulation function."""

    def test_returns_resilience_twin_report(self) -> None:
        config = ResilienceTwinConfig(n_mc=_FAST_N_MC, random_seed=0)
        report = run_resilience_simulation(config=config)
        assert isinstance(report, ResilienceTwinReport)

    def test_p_detection_in_unit_interval(self) -> None:
        config = ResilienceTwinConfig(n_mc=_FAST_N_MC, random_seed=1)
        report = run_resilience_simulation(config=config)
        assert 0.0 <= report.p_detection <= 1.0

    def test_p_false_alarm_in_unit_interval(self) -> None:
        config = ResilienceTwinConfig(n_mc=_FAST_N_MC, random_seed=2)
        report = run_resilience_simulation(config=config)
        assert 0.0 <= report.p_false_alarm <= 1.0

    def test_auc_in_unit_interval(self) -> None:
        config = ResilienceTwinConfig(n_mc=_FAST_N_MC, random_seed=3)
        report = run_resilience_simulation(config=config)
        assert 0.0 <= report.auc <= 1.0

    def test_per_class_accuracy_keys(self) -> None:
        config = ResilienceTwinConfig(n_mc=_FAST_N_MC, random_seed=4)
        report = run_resilience_simulation(config=config)
        expected_keys = {"nominal", "multipath", "hardware_fault", "spoofing"}
        assert set(report.per_class_accuracy.keys()) == expected_keys

    def test_confusion_matrix_shape(self) -> None:
        config = ResilienceTwinConfig(n_mc=_FAST_N_MC, random_seed=5)
        report = run_resilience_simulation(config=config)
        assert len(report.confusion_matrix) == 4
        assert all(len(row) == 4 for row in report.confusion_matrix)

    def test_confusion_matrix_row_sums_equal_trial_counts(self) -> None:
        config = ResilienceTwinConfig(n_mc=_FAST_N_MC, random_seed=6)
        report = run_resilience_simulation(config=config)
        for cls, row in zip(report.n_mc_per_class.values(), report.confusion_matrix):
            assert sum(row) == cls

    def test_n_mc_recorded_correctly(self) -> None:
        config = ResilienceTwinConfig(n_mc=_FAST_N_MC, random_seed=7)
        report = run_resilience_simulation(config=config)
        assert report.n_mc == _FAST_N_MC

    def test_n_mc_per_class_sums_to_n_mc(self) -> None:
        config = ResilienceTwinConfig(n_mc=_FAST_N_MC, random_seed=8)
        report = run_resilience_simulation(config=config)
        assert sum(report.n_mc_per_class.values()) == _FAST_N_MC

    def test_mean_confidence_in_unit_interval(self) -> None:
        config = ResilienceTwinConfig(n_mc=_FAST_N_MC, random_seed=9)
        report = run_resilience_simulation(config=config)
        assert 0.0 <= report.mean_confidence <= 1.0

    def test_reproducible_with_same_seed(self) -> None:
        config = ResilienceTwinConfig(n_mc=_FAST_N_MC, random_seed=42)
        r1 = run_resilience_simulation(config=config)
        r2 = run_resilience_simulation(config=config)
        assert r1.p_detection == r2.p_detection
        assert r1.auc == r2.auc

    def test_different_seeds_may_differ(self) -> None:
        r1 = run_resilience_simulation(config=ResilienceTwinConfig(n_mc=40, random_seed=10))
        r2 = run_resilience_simulation(config=ResilienceTwinConfig(n_mc=40, random_seed=99))
        # Not guaranteed to differ, but very likely with distinct seeds.
        # We check at least one metric field is defined (structural test).
        assert isinstance(r1.auc, float)
        assert isinstance(r2.auc, float)


class TestMainCLI:
    """Integration tests for the __main__ CLI entry point."""

    def test_exits_zero(self, tmp_path: Path) -> None:
        out = tmp_path / "report.json"
        rc = main(["--n-mc", "20", "--seed", "42", "--out", str(out)])
        assert rc == 0

    def test_creates_output_file(self, tmp_path: Path) -> None:
        out = tmp_path / "report.json"
        main(["--n-mc", "20", "--seed", "42", "--out", str(out)])
        assert out.exists()

    def test_output_is_valid_json(self, tmp_path: Path) -> None:
        out = tmp_path / "report.json"
        main(["--n-mc", "20", "--seed", "42", "--out", str(out)])
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert "run_id" in payload
        assert "config" in payload
        assert "report" in payload

    def test_output_run_id_format(self, tmp_path: Path) -> None:
        out = tmp_path / "report.json"
        main(["--n-mc", "20", "--seed", "42", "--out", str(out)])
        payload = json.loads(out.read_text(encoding="utf-8"))
        # Expect ISO-8601 UTC string: 20260101T120000Z
        run_id: str = payload["run_id"]
        assert len(run_id) == 16
        assert run_id.endswith("Z")

    def test_output_config_matches_args(self, tmp_path: Path) -> None:
        out = tmp_path / "report.json"
        main(["--n-mc", "20", "--seed", "7", "--out", str(out)])
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["config"]["n_mc"] == 20
        assert payload["config"]["random_seed"] == 7

    def test_output_report_has_required_keys(self, tmp_path: Path) -> None:
        out = tmp_path / "report.json"
        main(["--n-mc", "20", "--seed", "42", "--out", str(out)])
        report = json.loads(out.read_text(encoding="utf-8"))["report"]
        required = ("p_detection", "p_false_alarm", "auc", "per_class_accuracy", "confusion_matrix")
        for key in required:
            assert key in report, f"missing key: {key}"

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        out = tmp_path / "nested" / "dir" / "report.json"
        rc = main(["--n-mc", "20", "--seed", "42", "--out", str(out)])
        assert rc == 0
        assert out.exists()

    def test_invalid_n_mc_not_divisible_by_4(self, tmp_path: Path) -> None:
        out = tmp_path / "report.json"
        with pytest.raises(SystemExit) as exc_info:
            main(["--n-mc", "7", "--out", str(out)])
        assert exc_info.value.code != 0

    def test_invalid_n_mc_too_small(self, tmp_path: Path) -> None:
        out = tmp_path / "report.json"
        with pytest.raises(SystemExit) as exc_info:
            main(["--n-mc", "0", "--out", str(out)])
        assert exc_info.value.code != 0
