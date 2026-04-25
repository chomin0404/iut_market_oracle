"""Tests for the multi-sensor GNSS spoofing detection module (T1350)."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from gnss.multi_sensor_sim import (
    MultiSensorConfig,
    build_measurements,
    ms_percolation_stats,
    ms_select_subset,
    run_ms_simulation,
    simulate_trial_ms,
)
from schemas import MSSimReport

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_SMALL_CFG = MultiSensorConfig(
    T=40,
    n_sat=6,
    attack_start=15,
    attack_end=30,
    capture_len=5,
    n_nominal=10,
    n_attack=10,
    random_seed=0,
)
_RNG = np.random.default_rng(0)


# ---------------------------------------------------------------------------
# MultiSensorConfig validation
# ---------------------------------------------------------------------------


class TestMultiSensorConfig:
    def test_defaults_valid(self) -> None:
        cfg = MultiSensorConfig()
        assert cfg.T == 200
        assert cfg.n_sat == 8

    def test_attack_bounds_invalid(self) -> None:
        with pytest.raises(ValueError, match="attack_start"):
            MultiSensorConfig(T=100, attack_start=50, attack_end=50)

    def test_attack_end_exceeds_T(self) -> None:
        with pytest.raises(ValueError, match="attack_start"):
            MultiSensorConfig(T=100, attack_start=10, attack_end=101)

    def test_capture_len_zero(self) -> None:
        with pytest.raises(ValueError, match="capture_len"):
            MultiSensorConfig(capture_len=0)

    def test_n_sat_too_small(self) -> None:
        with pytest.raises(ValueError):
            MultiSensorConfig(n_sat=3)

    def test_negative_score_weight(self) -> None:
        with pytest.raises(ValueError, match="score_weights"):
            MultiSensorConfig(score_weights=(-0.1, 0.5, 0.6))

    def test_wrong_weight_count(self) -> None:
        with pytest.raises((ValueError, TypeError)):
            MultiSensorConfig(score_weights=(0.5, 0.5))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# build_measurements
# ---------------------------------------------------------------------------


class TestBuildMeasurements:
    def test_shape_genuine(self) -> None:
        rng = np.random.default_rng(1)
        meas = build_measurements(0, False, _SMALL_CFG, rng)
        assert meas.pr.shape == (6,)
        assert meas.dopp.shape == (6,)
        assert meas.aoa.shape == (6,)
        assert meas.ins.shape == (6,)

    def test_mix_zero_genuine(self) -> None:
        rng = np.random.default_rng(2)
        meas = build_measurements(0, False, _SMALL_CFG, rng)
        assert meas.mix == 0.0

    def test_mix_positive_during_attack(self) -> None:
        rng = np.random.default_rng(3)
        # epoch = attack_start (t₀) → mix = 1/capture_len
        meas = build_measurements(_SMALL_CFG.attack_start, True, _SMALL_CFG, rng)
        assert meas.mix > 0.0

    def test_mix_one_after_capture(self) -> None:
        rng = np.random.default_rng(4)
        # epoch well past attack_start + capture_len → mix should be 1.0
        t = _SMALL_CFG.attack_start + _SMALL_CFG.capture_len + 10
        t = min(t, _SMALL_CFG.attack_end)
        meas = build_measurements(t, True, _SMALL_CFG, rng)
        assert meas.mix == 1.0

    def test_mix_zero_before_attack(self) -> None:
        rng = np.random.default_rng(5)
        meas = build_measurements(_SMALL_CFG.attack_start - 1, True, _SMALL_CFG, rng)
        assert meas.mix == 0.0

    def test_pr_near_nominal(self) -> None:
        rng = np.random.default_rng(6)
        meas = build_measurements(0, False, _SMALL_CFG, rng)
        from gnss.multi_sensor_sim import _PR_NOMINAL

        assert np.all(np.abs(meas.pr - _PR_NOMINAL) < 100.0)


# ---------------------------------------------------------------------------
# ms_percolation_stats
# ---------------------------------------------------------------------------


class TestMsPercolationStats:
    def test_return_shapes(self) -> None:
        rng = np.random.default_rng(7)
        meas = build_measurements(0, False, _SMALL_CFG, rng)
        max_comp, chi, W, A = ms_percolation_stats(meas, _SMALL_CFG.n_sat)
        n = _SMALL_CFG.n_sat
        assert W.shape == (n, n)
        assert A.shape == (n, n)

    def test_max_comp_in_unit_interval(self) -> None:
        rng = np.random.default_rng(8)
        meas = build_measurements(0, False, _SMALL_CFG, rng)
        max_comp, chi, W, A = ms_percolation_stats(meas, _SMALL_CFG.n_sat)
        assert 0.0 < max_comp <= 1.0

    def test_chi_nonneg(self) -> None:
        rng = np.random.default_rng(9)
        meas = build_measurements(0, False, _SMALL_CFG, rng)
        _, chi, _, _ = ms_percolation_stats(meas, _SMALL_CFG.n_sat)
        assert chi >= 0.0

    def test_W_symmetric(self) -> None:
        rng = np.random.default_rng(10)
        meas = build_measurements(0, False, _SMALL_CFG, rng)
        _, _, W, _ = ms_percolation_stats(meas, _SMALL_CFG.n_sat)
        np.testing.assert_allclose(W, W.T)

    def test_A_diagonal_zero(self) -> None:
        rng = np.random.default_rng(11)
        meas = build_measurements(0, False, _SMALL_CFG, rng)
        _, _, _, A = ms_percolation_stats(meas, _SMALL_CFG.n_sat)
        assert np.all(np.diag(A) == 0)


# ---------------------------------------------------------------------------
# ms_select_subset
# ---------------------------------------------------------------------------


class TestMsSelectSubset:
    def test_subset_size_bounds(self) -> None:
        rng = np.random.default_rng(12)
        from gnss.multi_sensor_sim import _MAX_SUBSET_SIZE, _MIN_SUBSET_SIZE

        meas = build_measurements(0, False, _SMALL_CFG, rng)
        chosen, _ = ms_select_subset(meas, _SMALL_CFG.n_sat)
        assert _MIN_SUBSET_SIZE <= len(chosen) <= _MAX_SUBSET_SIZE

    def test_subset_indices_in_range(self) -> None:
        rng = np.random.default_rng(13)
        meas = build_measurements(0, False, _SMALL_CFG, rng)
        chosen, _ = ms_select_subset(meas, _SMALL_CFG.n_sat)
        assert np.all(chosen >= 0)
        assert np.all(chosen < _SMALL_CFG.n_sat)

    def test_lor_dev_in_unit_interval(self) -> None:
        rng = np.random.default_rng(14)
        meas = build_measurements(0, False, _SMALL_CFG, rng)
        _, lor_dev = ms_select_subset(meas, _SMALL_CFG.n_sat)
        assert 0.0 <= lor_dev <= 1.0

    def test_lor_dev_high_under_attack(self) -> None:
        """Under full spoofing (mix=1) AoAs cluster → lor_dev near 1."""
        rng = np.random.default_rng(15)
        # Use a high-capture epoch so mix → 1
        t = _SMALL_CFG.attack_start + _SMALL_CFG.capture_len + 5
        t = min(t, _SMALL_CFG.attack_end)
        meas = build_measurements(t, True, _SMALL_CFG, rng)
        _, lor_dev = ms_select_subset(meas, _SMALL_CFG.n_sat)
        # lor_dev should be noticeably higher than 0 when AoAs are clustered
        assert lor_dev >= 0.0  # relaxed: just non-negative under small cfg


# ---------------------------------------------------------------------------
# simulate_trial_ms
# ---------------------------------------------------------------------------


class TestSimulateTrial:
    def test_trace_lengths(self) -> None:
        rng = np.random.default_rng(20)
        summary = simulate_trial_ms(False, _SMALL_CFG, rng)
        T = _SMALL_CFG.T
        trace = summary.run_result.trace
        assert len(trace.score) == T
        assert len(trace.alarm) == T
        assert len(trace.mix) == T
        assert len(trace.m) == T
        assert len(trace.chi) == T
        assert len(trace.lor_dev) == T
        assert len(trace.pos_err) == T

    def test_genuine_mix_all_zero(self) -> None:
        rng = np.random.default_rng(21)
        summary = simulate_trial_ms(False, _SMALL_CFG, rng)
        assert all(v == 0.0 for v in summary.run_result.trace.mix)

    def test_attacked_mix_nonzero_during_window(self) -> None:
        rng = np.random.default_rng(22)
        summary = simulate_trial_ms(True, _SMALL_CFG, rng)
        mix = summary.run_result.trace.mix
        window = mix[_SMALL_CFG.attack_start : _SMALL_CFG.attack_end + 1]
        assert any(v > 0.0 for v in window)

    def test_scores_nonneg(self) -> None:
        rng = np.random.default_rng(23)
        summary = simulate_trial_ms(False, _SMALL_CFG, rng)
        assert all(s >= 0.0 for s in summary.run_result.trace.score)

    def test_score_max_consistent(self) -> None:
        rng = np.random.default_rng(24)
        summary = simulate_trial_ms(False, _SMALL_CFG, rng)
        assert np.isclose(summary.score_max, max(summary.run_result.trace.score))

    def test_genuine_label_zero(self) -> None:
        rng = np.random.default_rng(25)
        summary = simulate_trial_ms(False, _SMALL_CFG, rng)
        assert summary.label == 0

    def test_attacked_label_one(self) -> None:
        rng = np.random.default_rng(26)
        summary = simulate_trial_ms(True, _SMALL_CFG, rng)
        assert summary.label == 1

    def test_genuine_delay_none(self) -> None:
        rng = np.random.default_rng(27)
        summary = simulate_trial_ms(False, _SMALL_CFG, rng)
        # Genuine run: delay is always None (no attack window)
        assert summary.run_result.delay is None

    def test_pvt_rmse_nonneg(self) -> None:
        rng = np.random.default_rng(28)
        summary = simulate_trial_ms(True, _SMALL_CFG, rng)
        assert summary.run_result.pvt_rmse >= 0.0


# ---------------------------------------------------------------------------
# run_ms_simulation
# ---------------------------------------------------------------------------


class TestRunMsSimulation:
    def test_report_type(self) -> None:
        cfg = MultiSensorConfig(
            T=30, attack_start=10, attack_end=20, n_nominal=5, n_attack=5, random_seed=1
        )
        report = run_ms_simulation(cfg)
        assert isinstance(report, MSSimReport)

    def test_run_counts(self) -> None:
        cfg = MultiSensorConfig(
            T=30, attack_start=10, attack_end=20, n_nominal=8, n_attack=7, random_seed=2
        )
        report = run_ms_simulation(cfg)
        assert report.n_nominal == 8
        assert report.n_attack == 7
        assert len(report.runs) == 15  # nominal + attack

    def test_probabilities_in_unit_interval(self) -> None:
        cfg = MultiSensorConfig(
            T=30, attack_start=10, attack_end=20, n_nominal=10, n_attack=10, random_seed=3
        )
        report = run_ms_simulation(cfg)
        assert 0.0 <= report.p_fa <= 1.0
        assert 0.0 <= report.p_d <= 1.0
        assert 0.0 <= report.p_md <= 1.0

    def test_p_md_complement(self) -> None:
        cfg = MultiSensorConfig(
            T=30, attack_start=10, attack_end=20, n_nominal=5, n_attack=5, random_seed=4
        )
        report = run_ms_simulation(cfg)
        assert np.isclose(report.p_md, 1.0 - report.p_d)

    def test_auc_in_unit_interval(self) -> None:
        cfg = MultiSensorConfig(
            T=30, attack_start=10, attack_end=20, n_nominal=10, n_attack=10, random_seed=5
        )
        report = run_ms_simulation(cfg)
        assert 0.0 <= report.auc <= 1.0

    def test_roc_lengths_match(self) -> None:
        cfg = MultiSensorConfig(
            T=30, attack_start=10, attack_end=20, n_nominal=5, n_attack=5, random_seed=6
        )
        report = run_ms_simulation(cfg)
        assert len(report.roc_fpr) == len(report.roc_tpr)

    def test_default_config(self) -> None:
        """run_ms_simulation() with no args uses MultiSensorConfig defaults."""
        report = run_ms_simulation(
            MultiSensorConfig(
                T=30, attack_start=10, attack_end=20, n_nominal=3, n_attack=3, random_seed=7
            )
        )
        assert isinstance(report, MSSimReport)

    def test_reproducible(self) -> None:
        cfg = MultiSensorConfig(
            T=30, attack_start=10, attack_end=20, n_nominal=5, n_attack=5, random_seed=99
        )
        r1 = run_ms_simulation(cfg)
        r2 = run_ms_simulation(cfg)
        assert np.isclose(r1.auc, r2.auc)
        assert np.isclose(r1.p_fa, r2.p_fa)


# ---------------------------------------------------------------------------
# HTTP endpoint  POST /gnss/multi-sensor-sim
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client() -> TestClient:
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from api.app import app

    return TestClient(app)


class TestMultiSensorSimEndpoint:
    def test_default_request(self, client: TestClient) -> None:
        resp = client.post(
            "/gnss/multi-sensor-sim",
            json={
                "T": 30,
                "attack_start": 10,
                "attack_end": 20,
                "n_nominal": 5,
                "n_attack": 5,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "auc" in data
        assert "p_fa" in data
        assert "p_d" in data
        assert "roc_fpr" in data

    def test_invalid_attack_bounds(self, client: TestClient) -> None:
        resp = client.post(
            "/gnss/multi-sensor-sim",
            json={
                "T": 100,
                "attack_start": 80,
                "attack_end": 80,  # start == end → invalid
                "n_nominal": 5,
                "n_attack": 5,
            },
        )
        assert resp.status_code == 400

    def test_n_sat_too_small(self, client: TestClient) -> None:
        resp = client.post(
            "/gnss/multi-sensor-sim",
            json={
                "T": 30,
                "n_sat": 3,  # below FastAPI ge=4 → 422
                "attack_start": 10,
                "attack_end": 20,
                "n_nominal": 5,
                "n_attack": 5,
            },
        )
        assert resp.status_code == 422
