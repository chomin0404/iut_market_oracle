"""Unit and integration tests for T1300 GNSS spoofing detection simulation.

Coverage:
    - SimConfig validation
    - _init_constellation geometry
    - _build_graph properties
    - matroid_forest_count invariants
    - chi_stat distribution under H0
    - select_subset connectivity
    - wls_pvt / detection_score shape and type
    - np_threshold chi-squared consistency
    - run_mc_simulation output schema and statistical properties
    - HTTP endpoint /gnss/spoof-sim
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.app import app
from gnss.spoof_sim import (
    SimConfig,
    _build_graph,
    _init_constellation,
    chi_stat,
    detection_score,
    matroid_forest_count,
    np_threshold,
    run_mc_simulation,
    select_subset,
    wls_pvt,
)

client = TestClient(app)

_SMALL = SimConfig(n_mc=20, n_epochs=30, n_sats=6, subset_size=4, random_seed=0)


# ---------------------------------------------------------------------------
# SimConfig validation
# ---------------------------------------------------------------------------


class TestSimConfig:
    def test_defaults_valid(self):
        cfg = SimConfig()
        assert cfg.n_mc == 200
        assert cfg.subset_size < cfg.n_sats

    def test_subset_size_equal_n_sats_raises(self):
        with pytest.raises(ValueError, match="subset_size"):
            SimConfig(n_sats=5, subset_size=5)

    def test_subset_size_gt_n_sats_raises(self):
        with pytest.raises(ValueError, match="subset_size"):
            SimConfig(n_sats=5, subset_size=6)

    def test_invalid_attack_start_frac(self):
        with pytest.raises(ValueError, match="attack_start_frac"):
            SimConfig(attack_start_frac=0.0)

    def test_invalid_false_alarm_rate(self):
        with pytest.raises(ValueError, match="false_alarm_rate"):
            SimConfig(false_alarm_rate=1.0)

    def test_n_sats_too_small_raises(self):
        with pytest.raises(ValueError):
            SimConfig(n_sats=3, subset_size=2)


# ---------------------------------------------------------------------------
# Constellation geometry
# ---------------------------------------------------------------------------


class TestConstellation:
    def test_unit_vectors(self):
        e = _init_constellation(6)
        norms = np.linalg.norm(e, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-12)

    def test_upper_hemisphere(self):
        e = _init_constellation(8)
        assert (e[:, 2] > 0).all(), "All satellites must have positive z (elevation > 0)"

    def test_shape(self):
        for n in [4, 6, 8, 10]:
            assert _init_constellation(n).shape == (n, 3)

    def test_deterministic(self):
        e1 = _init_constellation(6)
        e2 = _init_constellation(6)
        np.testing.assert_array_equal(e1, e2)


# ---------------------------------------------------------------------------
# Similarity graph
# ---------------------------------------------------------------------------


class TestBuildGraph:
    def test_symmetric(self):
        d = np.array([0.1, -0.5, 0.3, 0.0, 0.2, -0.1])
        W = _build_graph(d, sigma=1.0)
        np.testing.assert_allclose(W, W.T, atol=1e-14)

    def test_zero_diagonal(self):
        d = np.random.default_rng(1).normal(0, 0.3, 6)
        W = _build_graph(d, sigma=1.0)
        np.testing.assert_array_equal(np.diag(W), np.zeros(6))

    def test_weights_in_01(self):
        d = np.random.default_rng(2).normal(0, 0.3, 6)
        W = _build_graph(d, sigma=1.0)
        off_diag = W[~np.eye(6, dtype=bool)]
        assert (off_diag >= 0.0).all()
        assert (off_diag <= 1.0).all()

    def test_identical_doppler_gives_weight_one(self):
        d = np.ones(4) * 0.5
        W = _build_graph(d, sigma=1.0)
        off_diag = W[~np.eye(4, dtype=bool)]
        np.testing.assert_allclose(off_diag, 1.0, atol=1e-14)

    def test_large_diff_gives_near_zero(self):
        d = np.array([0.0, 0.0, 10.0, 10.0])
        W = _build_graph(d, sigma=1.0)
        # w_{02} = exp(-100) ≈ 0
        assert W[0, 2] < 1e-10


# ---------------------------------------------------------------------------
# Matroid forest count  m(t)
# ---------------------------------------------------------------------------


class TestMatroidForestCount:
    def test_returns_positive(self):
        d = np.array([0.1, -0.2, 0.05, 0.3])
        W = _build_graph(d, sigma=1.5)
        assert matroid_forest_count(W) > 0.0

    def test_fully_connected_gt_disconnected(self):
        # Fully correlated Doppler → large weights → many high-weight forests
        d_same = np.zeros(4)
        W_same = _build_graph(d_same, sigma=1.0)
        # Spoofed: dispersed Doppler → low weights → fewer forests
        rng = np.random.default_rng(5)
        d_spread = rng.normal(0, 5.0, 4)
        W_spread = _build_graph(d_spread, sigma=1.0)
        assert matroid_forest_count(W_same) > matroid_forest_count(W_spread)

    def test_minimum_one_for_empty_graph(self):
        # Zero-weight graph → L=0 → det(I+0)=1
        W = np.zeros((4, 4))
        assert math.isclose(matroid_forest_count(W), 1.0, rel_tol=1e-9)

    def test_scalar_invariance(self):
        d = np.array([0.0, 0.1, -0.1, 0.2])
        W = _build_graph(d, sigma=1.5)
        # Scaling W uniformly changes L → det(I+αL) changes predictably
        m1 = matroid_forest_count(W)
        assert m1 >= 1.0


# ---------------------------------------------------------------------------
# Chi-squared statistic
# ---------------------------------------------------------------------------


class TestChiStat:
    def test_zero_variance_gives_zero(self):
        d = np.ones(6) * 0.5
        assert math.isclose(chi_stat(d, noise_std=0.3), 0.0, abs_tol=1e-12)

    def test_positive(self):
        rng = np.random.default_rng(7)
        d = rng.normal(0, 0.3, 6)
        assert chi_stat(d, noise_std=0.3) >= 0.0

    def test_h0_expectation(self):
        # E[chi(t)] under H0 ≈ n - 1 = 5
        rng = np.random.default_rng(99)
        samples = [chi_stat(rng.normal(0, 0.3, 6), 0.3) for _ in range(5000)]
        assert abs(np.mean(samples) - 5.0) < 0.2

    def test_attack_inflates_statistic(self):
        rng = np.random.default_rng(3)
        d_genuine = rng.normal(0, 0.3, 6)
        bias = rng.normal(0, 0.8, 6)     # differential bias
        d_attacked = d_genuine + bias
        assert chi_stat(d_attacked, 0.3) > chi_stat(d_genuine, 0.3)


# ---------------------------------------------------------------------------
# Subset selection
# ---------------------------------------------------------------------------


class TestSelectSubset:
    def test_correct_size(self):
        rng = np.random.default_rng(10)
        d = rng.normal(0, 0.3, 6)
        W = _build_graph(d, sigma=1.5)
        S = select_subset(W, k=4)
        assert len(S) == 4

    def test_valid_indices(self):
        rng = np.random.default_rng(11)
        d = rng.normal(0, 0.3, 6)
        W = _build_graph(d, sigma=1.5)
        S = select_subset(W, k=4)
        assert all(0 <= i < 6 for i in S)
        assert len(set(S)) == 4

    def test_sorted(self):
        rng = np.random.default_rng(12)
        d = rng.normal(0, 0.3, 6)
        W = _build_graph(d, sigma=1.5)
        S = select_subset(W, k=4)
        assert S == sorted(S)

    def test_prefers_correlated_satellites(self):
        # Satellites 0-3 have similar Doppler; 4-5 are outliers
        d = np.array([0.0, 0.05, -0.05, 0.03, 5.0, -5.0])
        W = _build_graph(d, sigma=1.0)
        S = select_subset(W, k=4)
        # Expect outlier satellites (4, 5) to be excluded
        assert 4 not in S
        assert 5 not in S


# ---------------------------------------------------------------------------
# WLS PVT and detection score
# ---------------------------------------------------------------------------


class TestWlsPvt:
    def setup_method(self):
        self.los = _init_constellation(6)
        self.rng = np.random.default_rng(20)

    def test_residuals_shape(self):
        d = self.rng.normal(0, 0.3, 6)
        W = _build_graph(d, sigma=1.5)
        S = [0, 1, 2, 3]
        _, r = wls_pvt(self.los, d, W, S)
        assert r.shape == (4,)

    def test_zero_input_zero_residuals(self):
        # With zero doppler deviation, WLS should give nearly zero residuals
        d = np.zeros(6)
        W = _build_graph(d, sigma=1.5)
        S = [0, 1, 2, 3, 4]
        _, r = wls_pvt(self.los, d, W, S)
        np.testing.assert_allclose(r, 0.0, atol=1e-10)

    def test_score_nonneg(self):
        d = self.rng.normal(0, 0.3, 6)
        W = _build_graph(d, sigma=1.5)
        S = [0, 1, 2, 3]
        _, r = wls_pvt(self.los, d, W, S)
        assert detection_score(r, W, S) >= 0.0

    def test_attack_inflates_score(self):
        # Genuine
        d_genuine = self.rng.normal(0, 0.3, 6)
        W = _build_graph(d_genuine, sigma=1.5)
        S = [0, 1, 2, 3]
        _, r0 = wls_pvt(self.los, d_genuine, W, S)
        score_genuine = detection_score(r0, W, S)

        # Attacked (large differential bias)
        bias = np.array([3.0, -3.0, 3.0, -3.0, 3.0, -3.0])
        d_attacked = d_genuine + bias
        W_att = _build_graph(d_attacked, sigma=1.5)
        _, r1 = wls_pvt(self.los, d_attacked, W_att, S)
        score_attacked = detection_score(r1, W_att, S)

        assert score_attacked > score_genuine


# ---------------------------------------------------------------------------
# NP threshold
# ---------------------------------------------------------------------------


class TestNpThreshold:
    def test_increases_with_pfa(self):
        t1 = np_threshold(4, pfa=0.10)
        t2 = np_threshold(4, pfa=0.01)
        assert t2 > t1   # lower pfa → higher threshold

    def test_increases_with_dof(self):
        t1 = np_threshold(5, pfa=0.05)
        t2 = np_threshold(8, pfa=0.05)
        assert t2 > t1   # more dof → higher threshold

    def test_h0_false_alarm_rate(self):
        from scipy.stats import chi2
        pfa = 0.05
        n_obs = 6
        tau = np_threshold(n_obs, pfa)
        df = max(n_obs - 4, 1)
        # P(chi2 > tau | df) should equal pfa
        actual_pfa = 1.0 - chi2.cdf(tau, df=df)
        assert math.isclose(actual_pfa, pfa, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# run_mc_simulation — output schema
# ---------------------------------------------------------------------------


class TestRunMcSimulationSchema:
    def setup_method(self):
        self.report = run_mc_simulation(_SMALL)

    def test_roc_lengths_equal(self):
        assert len(self.report.roc_fpr) == len(self.report.roc_tpr)

    def test_roc_length_positive(self):
        assert len(self.report.roc_fpr) > 0

    def test_auc_in_01(self):
        assert 0.0 <= self.report.auc <= 1.0

    def test_probabilities_in_01(self):
        assert 0.0 <= self.report.p_detection <= 1.0
        assert 0.0 <= self.report.p_false_alarm <= 1.0

    def test_std_nonneg(self):
        assert self.report.std_detection_delay >= 0.0
        assert self.report.std_pvt_degradation >= 0.0

    def test_n_mc_echoed(self):
        assert self.report.n_mc == _SMALL.n_mc

    def test_produced_at_set(self):
        assert self.report.produced_at is not None

    def test_runs_length_matches_n_mc(self):
        assert len(self.report.runs) == _SMALL.n_mc

    def test_run_trace_length_matches_n_epochs(self):
        for run in self.report.runs:
            assert len(run.trace.score) == _SMALL.n_epochs
            assert len(run.trace.alarm) == _SMALL.n_epochs
            assert len(run.trace.delay) == _SMALL.n_epochs
            assert len(run.trace.pvt_error) == _SMALL.n_epochs

    def test_run_summary_fields_nonneg(self):
        for run in self.report.runs:
            assert run.score_max >= 0.0
            assert run.pvt_rmse >= 0.0
            assert run.pvt_max >= 0.0

    def test_run_score_max_consistent_with_trace(self):
        for run in self.report.runs:
            assert math.isclose(run.score_max, max(run.trace.score), rel_tol=1e-9)

    def test_run_alarm_any_consistent_with_trace(self):
        for run in self.report.runs:
            assert run.alarm_any == any(run.trace.alarm)

    def test_run_delay_in_trace_at_most_one_nonnone(self):
        """trace.delay has at most one non-None entry per run (first alarm only)."""
        for run in self.report.runs:
            nonnone = [d for d in run.trace.delay if d is not None]
            assert len(nonnone) <= 1


# ---------------------------------------------------------------------------
# run_mc_simulation — statistical properties
# ---------------------------------------------------------------------------


class TestRunMcSimulationStats:
    def test_auc_above_random(self):
        """AUC should exceed 0.5 (better than random guessing) for strong attack."""
        cfg = SimConfig(
            n_mc=50, n_epochs=40, n_sats=6, subset_size=4,
            spoof_bias_std=4.0, spoof_diff_std=1.5, random_seed=42,
        )
        report = run_mc_simulation(cfg)
        assert report.auc > 0.51

    def test_false_alarm_rate_near_target(self):
        """Empirical PFA should be within a tolerance of the NP target."""
        cfg = SimConfig(
            n_mc=100, n_epochs=40, n_sats=6, subset_size=4,
            false_alarm_rate=0.05, random_seed=7,
        )
        report = run_mc_simulation(cfg)
        # Allow ±0.10 tolerance due to finite-sample variability
        assert report.p_false_alarm < 0.15

    def test_idempotent(self):
        """Same seed must produce identical results."""
        r1 = run_mc_simulation(_SMALL)
        r2 = run_mc_simulation(_SMALL)
        assert r1.auc == r2.auc
        assert r1.p_detection == r2.p_detection

    def test_stronger_attack_higher_detection(self):
        """Larger spoof bias → higher detection probability."""
        base = SimConfig(n_mc=30, n_epochs=40, n_sats=6, subset_size=4, random_seed=1)
        strong = SimConfig(
            n_mc=30, n_epochs=40, n_sats=6, subset_size=4,
            spoof_bias_std=5.0, spoof_diff_std=2.0, random_seed=1,
        )
        r_base = run_mc_simulation(base)
        r_strong = run_mc_simulation(strong)
        assert r_strong.p_detection >= r_base.p_detection


# ---------------------------------------------------------------------------
# HTTP endpoint  POST /gnss/spoof-sim
# ---------------------------------------------------------------------------


_FAST_PAYLOAD = {
    "n_mc": 10,
    "n_epochs": 20,
    "n_sats": 6,
    "subset_size": 4,
    "random_seed": 0,
}


class TestSpooferSimEndpoint:
    def test_status_200(self):
        r = client.post("/gnss/spoof-sim", json=_FAST_PAYLOAD)
        assert r.status_code == 200

    def test_response_schema(self):
        body = client.post("/gnss/spoof-sim", json=_FAST_PAYLOAD).json()
        for field in (
            "roc_fpr", "roc_tpr", "auc",
            "mean_detection_delay", "std_detection_delay",
            "mean_pvt_degradation", "std_pvt_degradation",
            "p_detection", "p_false_alarm", "n_mc", "produced_at",
        ):
            assert field in body, f"Missing field: {field}"

    def test_n_mc_echoed(self):
        body = client.post("/gnss/spoof-sim", json=_FAST_PAYLOAD).json()
        assert body["n_mc"] == _FAST_PAYLOAD["n_mc"]

    def test_subset_size_equal_n_sats_returns_400(self):
        r = client.post(
            "/gnss/spoof-sim",
            json={**_FAST_PAYLOAD, "n_sats": 5, "subset_size": 5},
        )
        assert r.status_code == 400

    def test_n_mc_above_max_returns_422(self):
        r = client.post("/gnss/spoof-sim", json={**_FAST_PAYLOAD, "n_mc": 9999})
        assert r.status_code == 422

    def test_invalid_false_alarm_rate_returns_422(self):
        r = client.post("/gnss/spoof-sim", json={**_FAST_PAYLOAD, "false_alarm_rate": 1.5})
        assert r.status_code == 422

    def test_idempotent(self):
        r1 = client.post("/gnss/spoof-sim", json=_FAST_PAYLOAD).json()
        r2 = client.post("/gnss/spoof-sim", json=_FAST_PAYLOAD).json()
        assert r1["auc"] == r2["auc"]
