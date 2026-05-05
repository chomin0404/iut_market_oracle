"""Tests for TwinCore components (T800).

Covers: BayesianStateFilter, RegimeTracker, StructDepMonitor,
        helper functions, TwinCore orchestrator, and run_mc_experiment.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from twin.core import (
    BayesianStateFilter,
    MCExperimentConfig,
    MCExperimentResult,
    PosteriorState,
    RegimeState,
    RegimeTracker,
    StructDepMonitor,
    StructuralState,
    TwinCore,
    TwinCoreDiagnosis,
    _build_corr_graph,
    _fiedler_value,
    _mean_clustering,
    run_mc_experiment,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_D = 3  # default state dim
_M = 3  # default obs dim


def _make_filter(**kwargs: object) -> BayesianStateFilter:
    return BayesianStateFilter(**kwargs)  # type: ignore[arg-type]


def _make_obs(d: int = _D, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=d)


# ---------------------------------------------------------------------------
# BayesianStateFilter
# ---------------------------------------------------------------------------


class TestBayesianStateFilter:
    def test_posterior_mean_shape(self) -> None:
        kf = _make_filter()
        ps = kf.step(_make_obs())
        assert ps.mean.shape == (_D,)

    def test_posterior_cov_shape(self) -> None:
        kf = _make_filter()
        ps = kf.step(_make_obs())
        assert ps.cov.shape == (_D, _D)

    def test_innovation_shape(self) -> None:
        kf = _make_filter()
        ps = kf.step(_make_obs())
        assert ps.innovation.shape == (_M,)

    def test_mahal_nonneg(self) -> None:
        kf = _make_filter()
        for seed in range(10):
            ps = kf.step(_make_obs(seed=seed))
            assert ps.mahal >= 0.0

    def test_log_lik_is_finite(self) -> None:
        kf = _make_filter()
        ps = kf.step(_make_obs())
        assert math.isfinite(ps.log_lik)

    def test_posterior_cov_symmetric(self) -> None:
        kf = _make_filter()
        for seed in range(5):
            ps = kf.step(_make_obs(seed=seed))
            assert np.allclose(ps.cov, ps.cov.T, atol=1e-10)

    def test_posterior_cov_positive_definite(self) -> None:
        kf = _make_filter()
        for seed in range(5):
            ps = kf.step(_make_obs(seed=seed))
            eigvals = np.linalg.eigvalsh(ps.cov)
            assert np.all(eigvals > -1e-12)

    def test_reset_restores_initial_state(self) -> None:
        kf = _make_filter()
        kf.step(_make_obs())
        kf.step(_make_obs(seed=1))
        kf.reset()
        # After reset posterior_mean should be zeros
        assert np.allclose(kf.posterior_mean, np.zeros(_D))
        assert np.allclose(kf.posterior_cov, np.eye(_D))

    def test_sequential_steps_change_mean(self) -> None:
        kf = _make_filter()
        ps0 = kf.step(_make_obs(seed=0))
        ps1 = kf.step(_make_obs(seed=1))
        # Posterior mean should update (not identical in general)
        assert not np.allclose(ps0.mean, ps1.mean)

    def test_invalid_obs_shape_raises(self) -> None:
        kf = _make_filter()
        with pytest.raises(ValueError, match="observation must be"):
            kf.step(np.zeros(5))

    def test_custom_state_obs_dim(self) -> None:
        kf = BayesianStateFilter(state_dim=5, obs_dim=2)
        y = np.ones(2)
        ps = kf.step(y)
        assert ps.mean.shape == (5,)
        assert ps.cov.shape == (5, 5)
        assert ps.innovation.shape == (2,)

    def test_invalid_transition_matrix_shape(self) -> None:
        with pytest.raises(ValueError, match="transition_matrix must be"):
            BayesianStateFilter(state_dim=3, transition_matrix=np.eye(2))

    def test_invalid_obs_matrix_shape(self) -> None:
        with pytest.raises(ValueError, match="obs_matrix must be"):
            BayesianStateFilter(state_dim=3, obs_dim=3, obs_matrix=np.eye(2))

    def test_posterior_properties_return_copy(self) -> None:
        kf = _make_filter()
        kf.step(_make_obs())
        m1 = kf.posterior_mean
        m1[:] = 999.0
        assert not np.allclose(kf.posterior_mean, 999.0)

    def test_returns_posterior_state(self) -> None:
        kf = _make_filter()
        ps = kf.step(_make_obs())
        assert isinstance(ps, PosteriorState)

    def test_mahal_large_for_outlier(self) -> None:
        """Observation far from the model should produce large Mahalanobis distance."""
        kf = BayesianStateFilter(obs_noise_std=0.01)
        # Large observation relative to prior
        y = np.array([100.0, 100.0, 100.0])
        ps = kf.step(y)
        assert ps.mahal > 1.0


# ---------------------------------------------------------------------------
# RegimeTracker
# ---------------------------------------------------------------------------


class TestRegimeTracker:
    def test_initial_regime_probs_uniform(self) -> None:
        rt = RegimeTracker()
        assert np.allclose(rt.regime_probs, [0.5, 0.5], atol=1e-12)

    def test_returns_regime_state(self) -> None:
        rt = RegimeTracker()
        rs = rt.update(1.0)
        assert isinstance(rs, RegimeState)

    def test_regime_probs_sum_to_one(self) -> None:
        rt = RegimeTracker()
        for mahal in [0.1, 0.5, 1.0, 2.5, 5.0]:
            rs = rt.update(mahal)
            assert pytest.approx(rs.regime_probs.sum(), abs=1e-9) == 1.0

    def test_regime_in_valid_range(self) -> None:
        rt = RegimeTracker()
        for mahal in [0.1, 1.0, 3.0, 10.0]:
            rs = rt.update(mahal)
            assert rs.regime in {0, 1}

    def test_confidence_is_max_prob(self) -> None:
        rt = RegimeTracker()
        for mahal in [0.2, 0.8, 2.0]:
            rs = rt.update(mahal)
            assert pytest.approx(rs.regime_confidence, abs=1e-9) == float(
                rs.regime_probs.max()
            )

    def test_reset_restores_uniform(self) -> None:
        rt = RegimeTracker()
        for _ in range(10):
            rt.update(5.0)
        rt.reset()
        assert np.allclose(rt.regime_probs, [0.5, 0.5], atol=1e-12)

    def test_transition_flag_on_regime_change(self) -> None:
        """Feed extreme observations to force a regime switch and detect transition."""
        rt = RegimeTracker()
        # Drive to normal regime first
        for _ in range(5):
            rt.update(0.01)
        # Then feed very large mahal to switch to stressed
        rs_list = [rt.update(10.0) for _ in range(10)]
        # At some point a transition must have been detected
        transitions = [rs.transition for rs in rs_list]
        # The stressed regime should eventually dominate
        final_regime = rs_list[-1].regime
        assert final_regime == 1 or any(transitions)

    def test_custom_emission_params(self) -> None:
        rt = RegimeTracker(emission_params=[(0.0, 0.3), (2.0, 0.5)])
        rs = rt.update(1.0)
        assert pytest.approx(rs.regime_probs.sum(), abs=1e-9) == 1.0

    def test_custom_transition_matrix(self) -> None:
        rt = RegimeTracker(transition=[[0.9, 0.1], [0.2, 0.8]])
        rs = rt.update(1.0)
        assert pytest.approx(rs.regime_probs.sum(), abs=1e-9) == 1.0


# ---------------------------------------------------------------------------
# Helper functions: _build_corr_graph, _fiedler_value, _mean_clustering
# ---------------------------------------------------------------------------


class TestGraphHelpers:
    def test_corr_graph_zero_diagonal(self) -> None:
        P = np.array([[1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.0]])
        W = _build_corr_graph(P)
        assert np.allclose(np.diag(W), 0.0)

    def test_corr_graph_symmetric(self) -> None:
        P = np.array([[1.0, 0.4], [0.4, 1.0]])
        W = _build_corr_graph(P)
        assert np.allclose(W, W.T)

    def test_corr_graph_range(self) -> None:
        rng = np.random.default_rng(7)
        A = rng.normal(size=(4, 4))
        P = A.T @ A + np.eye(4)  # PSD
        W = _build_corr_graph(P)
        assert np.all(W >= 0.0)
        assert np.all(W <= 1.0 + 1e-10)

    def test_fiedler_value_identity(self) -> None:
        # For complete graph (all W=1, diag=0), Fiedler value is n
        n = 4
        W = np.ones((n, n)) - np.eye(n)
        lam2 = _fiedler_value(W)
        # Fiedler value for K_n is n
        assert lam2 == pytest.approx(n, abs=1e-9)

    def test_fiedler_value_disconnected_graph(self) -> None:
        # Block-diagonal = disconnected → λ₂ = 0
        W = np.zeros((4, 4))
        W[0, 1] = W[1, 0] = 1.0
        W[2, 3] = W[3, 2] = 1.0
        lam2 = _fiedler_value(W)
        assert lam2 == pytest.approx(0.0, abs=1e-9)

    def test_fiedler_nonneg(self) -> None:
        rng = np.random.default_rng(13)
        for _ in range(5):
            A = rng.normal(size=(5, 5))
            P = A.T @ A + np.eye(5)
            W = _build_corr_graph(P)
            assert _fiedler_value(W) >= -1e-10

    def test_mean_clustering_complete_graph(self) -> None:
        # Complete graph (all edges): CC = 1
        W = np.ones((5, 5)) - np.eye(5)
        cc = _mean_clustering(W, thresh=0.0)
        assert cc == pytest.approx(1.0, abs=1e-9)

    def test_mean_clustering_no_edges(self) -> None:
        # Isolated nodes (after threshold): returns 0 (no counted nodes)
        W = np.zeros((4, 4))
        cc = _mean_clustering(W, thresh=0.5)
        assert cc == 0.0


# ---------------------------------------------------------------------------
# StructDepMonitor
# ---------------------------------------------------------------------------


class TestStructDepMonitor:
    def _identity_cov(self, d: int = 3) -> np.ndarray:
        return np.eye(d)

    def test_returns_structural_state(self) -> None:
        mon = StructDepMonitor()
        ss = mon.update(self._identity_cov())
        assert isinstance(ss, StructuralState)

    def test_first_epoch_change_rate_zero(self) -> None:
        mon = StructDepMonitor()
        ss = mon.update(self._identity_cov())
        assert ss.graph_change_rate == pytest.approx(0.0, abs=1e-12)

    def test_fiedler_nonneg(self) -> None:
        mon = StructDepMonitor()
        rng = np.random.default_rng(5)
        for _ in range(5):
            A = rng.normal(size=(4, 4))
            P = A.T @ A + np.eye(4)
            ss = mon.update(P)
            assert ss.fiedler_value >= -1e-10

    def test_streak_increments_for_low_connectivity(self) -> None:
        # Use identity covariance: W has zero off-diagonal → λ₂ = 0 < threshold
        mon = StructDepMonitor(fiedler_low_thresh=999.0)  # always "low"
        for i in range(1, 5):
            ss = mon.update(self._identity_cov())
            assert ss.fiedler_streak == i

    def test_streak_resets_on_high_connectivity(self) -> None:
        # Use a fully correlated covariance to get large Fiedler value
        # W = all-ones minus diagonal → complete graph → λ₂ = d = 3
        P_full = np.full((3, 3), 0.99) + 0.01 * np.eye(3)
        # Fiedler value ≈ 3, so thresh=0.01 is never triggered
        mon = StructDepMonitor(fiedler_low_thresh=0.01)
        for _ in range(3):
            ss = mon.update(P_full)
        assert ss.fiedler_streak == 0

    def test_alert_fires_when_streak_exceeds_threshold(self) -> None:
        mon = StructDepMonitor(streak_thresh=2, fiedler_low_thresh=999.0)
        for _ in range(2):
            mon.update(self._identity_cov())
        ss = mon.update(self._identity_cov())
        assert ss.alert is True

    def test_alert_fires_on_large_change_rate(self) -> None:
        mon = StructDepMonitor(change_thresh=0.01)
        rng = np.random.default_rng(42)
        A = rng.normal(size=(3, 3))
        P1 = A.T @ A + np.eye(3)
        mon.update(P1)
        # Drastically different covariance
        B = rng.normal(size=(3, 3)) * 100
        P2 = B.T @ B + np.eye(3)
        ss = mon.update(P2)
        assert ss.alert is True

    def test_reset_clears_streak_and_prev_w(self) -> None:
        mon = StructDepMonitor(fiedler_low_thresh=999.0)
        for _ in range(5):
            mon.update(self._identity_cov())
        mon.reset()
        ss = mon.update(self._identity_cov())
        # After reset: streak restarts from 1, change_rate is 0
        assert ss.fiedler_streak == 1
        assert ss.graph_change_rate == pytest.approx(0.0, abs=1e-12)

    def test_clustering_coeff_in_range(self) -> None:
        mon = StructDepMonitor()
        rng = np.random.default_rng(9)
        for _ in range(5):
            A = rng.normal(size=(4, 4))
            P = A.T @ A + np.eye(4)
            ss = mon.update(P)
            assert 0.0 <= ss.clustering_coeff <= 1.0


# ---------------------------------------------------------------------------
# TwinCore
# ---------------------------------------------------------------------------


class TestTwinCore:
    def test_returns_twin_core_diagnosis(self) -> None:
        core = TwinCore()
        diag = core.step(_make_obs())
        assert isinstance(diag, TwinCoreDiagnosis)

    def test_step_counter_increments(self) -> None:
        core = TwinCore()
        for i in range(5):
            diag = core.step(_make_obs(seed=i))
            assert diag.t == i

    def test_anomaly_score_in_range(self) -> None:
        core = TwinCore()
        for seed in range(20):
            diag = core.step(_make_obs(seed=seed))
            assert 0.0 <= diag.anomaly_score <= 1.0

    def test_reset_resets_step_counter(self) -> None:
        core = TwinCore()
        for i in range(3):
            core.step(_make_obs(seed=i))
        core.reset()
        diag = core.step(_make_obs())
        assert diag.t == 0

    def test_posterior_shape(self) -> None:
        core = TwinCore()
        diag = core.step(_make_obs())
        assert diag.posterior.mean.shape == (_D,)
        assert diag.posterior.cov.shape == (_D, _D)

    def test_regime_probs_shape(self) -> None:
        core = TwinCore()
        diag = core.step(_make_obs())
        assert diag.regime.regime_probs.shape == (2,)

    def test_regime_probs_sum_to_one(self) -> None:
        core = TwinCore()
        for seed in range(5):
            diag = core.step(_make_obs(seed=seed))
            assert pytest.approx(diag.regime.regime_probs.sum(), abs=1e-9) == 1.0

    def test_alert_is_bool(self) -> None:
        core = TwinCore()
        diag = core.step(_make_obs())
        assert isinstance(diag.alert, bool)

    def test_alert_fires_on_large_mahal(self) -> None:
        """Observation 100× observation noise → very large mahal → alert."""
        core = TwinCore(obs_noise_std=0.01)
        diag = core.step(np.full(_M, 100.0))
        assert diag.alert is True

    def test_custom_dimensions(self) -> None:
        core = TwinCore(state_dim=5, obs_dim=4)
        y = np.zeros(4)
        diag = core.step(y)
        assert diag.posterior.mean.shape == (5,)

    def test_sequential_diagnosis_differ(self) -> None:
        core = TwinCore()
        d0 = core.step(_make_obs(seed=0))
        d1 = core.step(_make_obs(seed=1))
        assert d0.t != d1.t

    def test_reset_produces_same_result_twice(self) -> None:
        """Resetting TwinCore and replaying the same observations gives the same output."""
        core = TwinCore()
        obs = [_make_obs(seed=i) for i in range(5)]
        diags_a = [core.step(y) for y in obs]

        core.reset()
        diags_b = [core.step(y) for y in obs]

        for a, b in zip(diags_a, diags_b):
            assert a.t == b.t
            assert np.allclose(a.posterior.mean, b.posterior.mean)
            assert pytest.approx(a.anomaly_score, abs=1e-12) == b.anomaly_score


# ---------------------------------------------------------------------------
# run_mc_experiment
# ---------------------------------------------------------------------------


class TestMCExperiment:
    def test_returns_mc_experiment_result(self) -> None:
        result = run_mc_experiment(n_trials=3, horizon=5)
        assert isinstance(result, MCExperimentResult)

    def test_posterior_means_shape(self) -> None:
        cfg = MCExperimentConfig(n_trials=4, horizon=6, state_dim=3)
        result = run_mc_experiment(cfg)
        assert result.posterior_means.shape == (4, 7, 3)

    def test_anomaly_scores_shape(self) -> None:
        cfg = MCExperimentConfig(n_trials=4, horizon=6)
        result = run_mc_experiment(cfg)
        assert result.anomaly_scores.shape == (4, 6)

    def test_regime_probs_shape(self) -> None:
        cfg = MCExperimentConfig(n_trials=4, horizon=6)
        result = run_mc_experiment(cfg)
        assert result.regime_probs.shape == (4, 6, 2)

    def test_final_anomaly_shape(self) -> None:
        cfg = MCExperimentConfig(n_trials=4, horizon=6)
        result = run_mc_experiment(cfg)
        assert result.final_anomaly.shape == (4,)

    def test_mean_traj_shape(self) -> None:
        cfg = MCExperimentConfig(n_trials=3, horizon=5, state_dim=3)
        result = run_mc_experiment(cfg)
        assert result.mean_traj.shape == (6, 3)

    def test_std_traj_shape(self) -> None:
        cfg = MCExperimentConfig(n_trials=3, horizon=5, state_dim=3)
        result = run_mc_experiment(cfg)
        assert result.std_traj.shape == (6, 3)

    def test_anomaly_scores_in_range(self) -> None:
        result = run_mc_experiment(n_trials=5, horizon=10)
        assert np.all(result.anomaly_scores >= 0.0)
        assert np.all(result.anomaly_scores <= 1.0)

    def test_regime_probs_sum_to_one(self) -> None:
        result = run_mc_experiment(n_trials=3, horizon=4)
        row_sums = result.regime_probs.sum(axis=2)
        assert np.allclose(row_sums, 1.0, atol=1e-9)

    def test_reproducibility_same_seed(self) -> None:
        """Two calls with the same config should produce identical results."""
        cfg = MCExperimentConfig(n_trials=5, horizon=8, base_seed=42)
        r1 = run_mc_experiment(cfg)
        r2 = run_mc_experiment(cfg)
        assert np.allclose(r1.posterior_means, r2.posterior_means)
        assert np.allclose(r1.anomaly_scores, r2.anomaly_scores)

    def test_different_seed_different_result(self) -> None:
        r1 = run_mc_experiment(n_trials=5, horizon=8, base_seed=1)
        r2 = run_mc_experiment(n_trials=5, horizon=8, base_seed=9999)
        assert not np.allclose(r1.posterior_means, r2.posterior_means)

    def test_final_anomaly_equals_last_column(self) -> None:
        cfg = MCExperimentConfig(n_trials=4, horizon=6)
        result = run_mc_experiment(cfg)
        assert np.allclose(result.final_anomaly, result.anomaly_scores[:, -1])

    def test_mean_traj_is_mean_over_trials(self) -> None:
        cfg = MCExperimentConfig(n_trials=6, horizon=4, state_dim=3)
        result = run_mc_experiment(cfg)
        expected_mean = result.posterior_means.mean(axis=0)
        assert np.allclose(result.mean_traj, expected_mean)

    def test_config_stored_in_result(self) -> None:
        cfg = MCExperimentConfig(n_trials=3, horizon=4, base_seed=77)
        result = run_mc_experiment(cfg)
        assert result.config is cfg

    def test_kwargs_forwarded_to_config(self) -> None:
        result = run_mc_experiment(n_trials=2, horizon=3, base_seed=11)
        assert result.config.n_trials == 2
        assert result.config.horizon == 3
        assert result.config.base_seed == 11

    def test_custom_x0_mean(self) -> None:
        cfg = MCExperimentConfig(
            n_trials=3, horizon=4, x0_mean=[1.0, 2.0, 3.0]
        )
        result = run_mc_experiment(cfg)
        # Initial posterior_means entries are x0 samples near [1,2,3]
        x0_samples = result.posterior_means[:, 0, :]
        assert x0_samples.mean(axis=0) == pytest.approx([1.0, 2.0, 3.0], abs=1.0)
