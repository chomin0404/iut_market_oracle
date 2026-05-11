"""Tests for Process Yield Twin (T1600)."""

from __future__ import annotations

import numpy as np
import pytest

from schemas import (
    DOERecommendation,
    ExperimentPoint,
    FactorSpec,
    YieldTwinReport,
)
from yield_twin.gp_surrogate import GPHyperparams, GPSurrogate, _rbf_kernel
from yield_twin.twin import (
    ProcessYieldTwin,
    YieldTwinConfig,
    _build_info_matrix_inv,
    _d_leverages,
    _lhs_candidates,
    _quadratic_basis,
    recommend_next_experiment,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def two_factor_specs() -> list[FactorSpec]:
    return [
        FactorSpec(name="temp", low=150.0, high=250.0),
        FactorSpec(name="pressure", low=1.0, high=5.0),
    ]


@pytest.fixture
def three_factor_specs() -> list[FactorSpec]:
    return [
        FactorSpec(name="temp", low=150.0, high=250.0),
        FactorSpec(name="pressure", low=1.0, high=5.0),
        FactorSpec(name="speed", low=100.0, high=400.0),
    ]


@pytest.fixture
def simple_observations() -> list[ExperimentPoint]:
    """6 factorial-grid observations for a 2-factor problem."""
    data = [
        ({"temp": 150.0, "pressure": 1.0}, 0.60),
        ({"temp": 200.0, "pressure": 1.0}, 0.70),
        ({"temp": 250.0, "pressure": 1.0}, 0.65),
        ({"temp": 150.0, "pressure": 3.0}, 0.72),
        ({"temp": 200.0, "pressure": 3.0}, 0.88),
        ({"temp": 250.0, "pressure": 5.0}, 0.75),
    ]
    return [ExperimentPoint(factors=f, yield_obs=y) for f, y in data]


# ---------------------------------------------------------------------------
# FactorSpec validation
# ---------------------------------------------------------------------------


class TestFactorSpec:
    def test_valid(self) -> None:
        fs = FactorSpec(name="x", low=0.0, high=1.0)
        assert fs.high > fs.low

    def test_invalid_low_ge_high(self) -> None:
        with pytest.raises(ValueError):
            FactorSpec(name="x", low=5.0, high=5.0)


# ---------------------------------------------------------------------------
# GP Kernel
# ---------------------------------------------------------------------------


class TestRbfKernel:
    def test_symmetric(self) -> None:
        rng = np.random.default_rng(0)
        X = rng.uniform(size=(5, 3))
        K = _rbf_kernel(X, X, np.ones(3), 1.0)
        assert np.allclose(K, K.T)

    def test_diagonal_equals_signal_var(self) -> None:
        rng = np.random.default_rng(1)
        X = rng.uniform(size=(4, 2))
        sigma_f_sq = 2.5
        K = _rbf_kernel(X, X, np.ones(2), sigma_f_sq)
        assert np.allclose(np.diag(K), sigma_f_sq)

    def test_off_diagonal_bounded(self) -> None:
        rng = np.random.default_rng(2)
        X = rng.uniform(size=(10, 4))
        sigma_f_sq = 1.0
        K = _rbf_kernel(X, X, np.ones(4), sigma_f_sq)
        assert np.all(K >= 0.0)
        assert np.all(K <= sigma_f_sq + 1e-10)


# ---------------------------------------------------------------------------
# GP Surrogate
# ---------------------------------------------------------------------------


class TestGPSurrogate:
    def test_fit_and_predict_shape(self) -> None:
        rng_np = np.random.default_rng(0)
        X = rng_np.uniform(size=(8, 2))
        y = rng_np.uniform(0.6, 0.9, size=8)
        gp = GPSurrogate(n_restarts=1)
        gp.fit(X, y, rng_np)
        X_star = rng_np.uniform(size=(5, 2))
        mu, sigma = gp.predict(X_star)
        assert mu.shape == (5,)
        assert sigma.shape == (5,)

    def test_sigma_nonnegative(self) -> None:
        rng_np = np.random.default_rng(1)
        X = rng_np.uniform(size=(6, 2))
        y = np.sin(X[:, 0]) * 0.5 + 0.7
        gp = GPSurrogate(n_restarts=1)
        gp.fit(X, y, rng_np)
        _, sigma = gp.predict(rng_np.uniform(size=(20, 2)))
        assert np.all(sigma >= 0.0)

    def test_near_training_point_low_sigma(self) -> None:
        rng_np = np.random.default_rng(2)
        X = np.array([[0.2, 0.3], [0.7, 0.8]])
        y = np.array([0.6, 0.9])
        gp = GPSurrogate(n_restarts=1)
        gp.fit(X, y, rng_np)
        mu, sigma = gp.predict(X)
        # Posterior std near training points should be small (not exact 0 due to noise)
        assert np.all(sigma < 0.5)

    def test_ei_nonnegative(self) -> None:
        rng_np = np.random.default_rng(3)
        X = rng_np.uniform(size=(6, 2))
        y = rng_np.uniform(0.5, 0.85, size=6)
        gp = GPSurrogate(n_restarts=1)
        gp.fit(X, y, rng_np)
        candidates = rng_np.uniform(size=(50, 2))
        ei = gp.expected_improvement(candidates, y.max())
        assert np.all(ei >= 0.0)

    def test_loocv_r2_within_bounds(self) -> None:
        rng_np = np.random.default_rng(4)
        X = rng_np.uniform(size=(8, 2))
        y = np.sin(3 * X[:, 0]) * 0.2 + 0.75
        gp = GPSurrogate(n_restarts=1)
        gp.fit(X, y, rng_np)
        r2 = gp.loocv_r2()
        assert r2 is not None
        assert -1.0 <= r2 <= 1.0

    def test_loocv_r2_none_with_few_obs(self) -> None:
        rng_np = np.random.default_rng(5)
        X = rng_np.uniform(size=(2, 2))
        y = np.array([0.7, 0.8])
        gp = GPSurrogate(n_restarts=1)
        gp.fit(X, y, rng_np)
        assert gp.loocv_r2() is None  # < 3 obs

    def test_hyperparams_stored(self) -> None:
        rng_np = np.random.default_rng(6)
        X = rng_np.uniform(size=(5, 3))
        y = rng_np.uniform(size=5)
        gp = GPSurrogate(n_restarts=1)
        gp.fit(X, y, rng_np)
        hp = gp.hyperparams
        assert isinstance(hp, GPHyperparams)
        assert hp.signal_var > 0.0
        assert hp.noise_var > 0.0
        assert all(ls > 0.0 for ls in hp.length_scales)


# ---------------------------------------------------------------------------
# D-Optimal DOE utilities
# ---------------------------------------------------------------------------


class TestQuadraticBasis:
    def test_dimension_d1(self) -> None:
        # p = 1 + 1 + 1*(1+1)//2 = 1 + 1 + 1 = 3
        phi = _quadratic_basis(np.array([0.5]))
        assert len(phi) == 3
        assert phi[0] == 1.0  # intercept
        assert phi[1] == 0.5  # x1
        assert phi[2] == pytest.approx(0.25)  # x1²

    def test_dimension_d2(self) -> None:
        # p = 1 + 2 + 2*(2+1)//2 = 1 + 2 + 3 = 6
        phi = _quadratic_basis(np.array([0.3, 0.7]))
        assert len(phi) == 6
        assert phi[0] == 1.0
        assert phi[1] == pytest.approx(0.3)
        assert phi[2] == pytest.approx(0.7)
        assert phi[3] == pytest.approx(0.09)  # x1²
        assert phi[4] == pytest.approx(0.21)  # x1*x2
        assert phi[5] == pytest.approx(0.49)  # x2²

    def test_dimension_d3(self) -> None:
        # p = 1 + 3 + 3*(3+1)//2 = 1 + 3 + 6 = 10
        phi = _quadratic_basis(np.zeros(3))
        assert len(phi) == 10
        assert phi[0] == 1.0  # intercept


class TestDLeverages:
    def test_leverages_nonnegative(self) -> None:
        rng_np = np.random.default_rng(0)
        X_cand = rng_np.uniform(size=(50, 2))
        X_obs = rng_np.uniform(size=(4, 2))
        M_inv = _build_info_matrix_inv(X_obs, d=2)
        lev = _d_leverages(X_cand, M_inv)
        assert np.all(lev >= 0.0)
        assert lev.shape == (50,)

    def test_empty_design_uses_regularisation(self) -> None:
        # With no observations, M_inv should still be computable
        M_inv = _build_info_matrix_inv(np.empty((0, 2)), d=2)
        phi = _quadratic_basis(np.array([0.5, 0.5]))
        lev = float(phi @ M_inv @ phi)
        assert lev > 0.0


class TestLHSCandidates:
    def test_shape(self) -> None:
        rng_np = np.random.default_rng(0)
        X = _lhs_candidates(100, 3, rng_np)
        assert X.shape == (100, 3)

    def test_in_unit_cube(self) -> None:
        rng_np = np.random.default_rng(1)
        X = _lhs_candidates(200, 4, rng_np)
        assert np.all(X >= 0.0)
        assert np.all(X <= 1.0)


# ---------------------------------------------------------------------------
# ProcessYieldTwin
# ---------------------------------------------------------------------------


class TestProcessYieldTwin:
    def test_recommend_before_observations(self, two_factor_specs) -> None:
        config = YieldTwinConfig(factor_specs=two_factor_specs, n_candidates=100, random_seed=0)
        twin = ProcessYieldTwin(config)
        rec = twin.recommend()
        assert isinstance(rec, DOERecommendation)
        assert rec.acquisition_mode == "doe_explore"
        assert rec.n_observations == 0
        assert 0.0 <= rec.predicted_yield <= 1.0

    def test_recommended_factors_within_bounds(self, two_factor_specs) -> None:
        config = YieldTwinConfig(factor_specs=two_factor_specs, n_candidates=100, random_seed=1)
        twin = ProcessYieldTwin(config)
        rec = twin.recommend()
        assert 150.0 <= rec.factors["temp"] <= 250.0
        assert 1.0 <= rec.factors["pressure"] <= 5.0

    def test_acquisition_mode_transitions(self, two_factor_specs) -> None:
        """Mode should start at doe_explore and eventually reach ei_exploit."""
        config = YieldTwinConfig(factor_specs=two_factor_specs, n_candidates=50, random_seed=2)
        twin = ProcessYieldTwin(config)
        rng = np.random.default_rng(2)

        modes_seen: set[str] = set()
        for _ in range(30):
            factors = {
                "temp": float(rng.uniform(150, 250)),
                "pressure": float(rng.uniform(1, 5)),
            }
            twin.observe(factors, float(rng.uniform(0.6, 0.95)))
            modes_seen.add(twin.recommend().acquisition_mode)

        assert "doe_explore" in modes_seen
        assert "ei_exploit" in modes_seen

    def test_report_structure(self, two_factor_specs, simple_observations) -> None:
        config = YieldTwinConfig(factor_specs=two_factor_specs, n_candidates=100, random_seed=3)
        twin = ProcessYieldTwin(config)
        twin.observe_batch(simple_observations)
        report = twin.report()

        assert isinstance(report, YieldTwinReport)
        assert report.n_observations == 6
        assert report.best_yield_observed == pytest.approx(0.88)
        assert report.best_factors is not None
        assert set(report.best_factors.keys()) == {"temp", "pressure"}

    def test_loocv_r2_available_after_enough_obs(
        self, two_factor_specs, simple_observations
    ) -> None:
        config = YieldTwinConfig(factor_specs=two_factor_specs, n_candidates=50, random_seed=4)
        twin = ProcessYieldTwin(config)
        twin.observe_batch(simple_observations)
        report = twin.report()
        assert report.surrogate_loocv_r2 is not None
        assert -1.0 <= report.surrogate_loocv_r2 <= 1.0

    def test_gp_hyperparams_in_report(self, two_factor_specs, simple_observations) -> None:
        config = YieldTwinConfig(factor_specs=two_factor_specs, n_candidates=50, random_seed=5)
        twin = ProcessYieldTwin(config)
        twin.observe_batch(simple_observations)
        report = twin.report()
        assert "signal_var" in report.gp_hyperparams
        assert "noise_var" in report.gp_hyperparams
        assert "length_scale_temp" in report.gp_hyperparams
        assert "length_scale_pressure" in report.gp_hyperparams

    def test_observe_batch_with_none_yield_skipped(self, two_factor_specs) -> None:
        config = YieldTwinConfig(factor_specs=two_factor_specs, n_candidates=50, random_seed=6)
        twin = ProcessYieldTwin(config)
        pts = [
            ExperimentPoint(factors={"temp": 200.0, "pressure": 3.0}, yield_obs=0.80),
            ExperimentPoint(factors={"temp": 180.0, "pressure": 2.0}, yield_obs=None),
        ]
        twin.observe_batch(pts)
        assert twin.report().n_observations == 1

    def test_three_factor_recommendation(self, three_factor_specs) -> None:
        config = YieldTwinConfig(factor_specs=three_factor_specs, n_candidates=100, random_seed=7)
        twin = ProcessYieldTwin(config)
        rng = np.random.default_rng(7)
        for _ in range(5):
            twin.observe(
                {
                    "temp": float(rng.uniform(150, 250)),
                    "pressure": float(rng.uniform(1, 5)),
                    "speed": float(rng.uniform(100, 400)),
                },
                float(rng.uniform(0.6, 0.9)),
            )
        rec = twin.recommend()
        assert "speed" in rec.factors
        assert 100.0 <= rec.factors["speed"] <= 400.0


# ---------------------------------------------------------------------------
# One-shot convenience function
# ---------------------------------------------------------------------------


class TestRecommendNextExperiment:
    def test_returns_yield_twin_report(self, two_factor_specs, simple_observations) -> None:
        report = recommend_next_experiment(
            two_factor_specs,
            simple_observations,
            random_seed=0,
            n_candidates=100,
        )
        assert isinstance(report, YieldTwinReport)
        assert report.n_observations == 6

    def test_fusion_score_nonneg(self, two_factor_specs, simple_observations) -> None:
        report = recommend_next_experiment(
            two_factor_specs, simple_observations, random_seed=1, n_candidates=100
        )
        assert report.recommendation.fusion_score >= 0.0
