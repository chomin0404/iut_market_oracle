"""Tests for MonteCarloSimulator.

Coverage:
    1. Independent copula → near-zero correlation between variables.
    2. Gaussian copula → empirical correlation approximates target matrix.
    3. Marginal distributions → empirical mean and variance within 5% of theoretical.
    4. SimulationResult shape contract.
    5. n_samples > 100_000 raises ValueError.
    6. Unsupported copula type raises ValueError.
    7. Student-T copula produces valid samples.
    8. Clayton copula produces valid samples (frailty method).
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.stats as st

from core.simulator import MAX_N_SAMPLES, MCSimulationResult, MonteCarloSimulator

SEED = 42
N_SAMPLES = 10_000  # large enough to satisfy 5% tolerance reliably

simulator = MonteCarloSimulator()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _single_var_samples(dist_spec: dict, seed: int = SEED) -> np.ndarray:
    """Simulate N_SAMPLES from a single marginal with independent copula."""
    result = simulator.simulate(
        n_vars=1,
        n_samples=N_SAMPLES,
        distributions=[dist_spec],
        copula={"type": "independent"},
        seed=seed,
    )
    return result.samples[0]  # shape (N_SAMPLES,)


def _check_mean(samples: np.ndarray, true_mean: float, tol: float = 0.05) -> None:
    """Assert |empirical_mean - true_mean| <= tol * (1 + |true_mean|)."""
    err = abs(samples.mean() - true_mean)
    bound = tol * (1.0 + abs(true_mean))
    assert err <= bound, f"mean error {err:.4f} exceeds tolerance {bound:.4f}"


def _check_var(samples: np.ndarray, true_var: float, tol: float = 0.05) -> None:
    """Assert |empirical_var - true_var| <= tol * true_var."""
    err = abs(samples.var() - true_var)
    bound = tol * true_var
    assert err <= bound, f"variance error {err:.4f} exceeds tolerance {bound:.4f}"


# ---------------------------------------------------------------------------
# 1. Independent copula
# ---------------------------------------------------------------------------


class TestIndependentCopula:
    def test_low_pearson_correlation(self) -> None:
        """Variables drawn with independent copula should have near-zero correlation."""
        result = simulator.simulate(
            n_vars=2,
            n_samples=N_SAMPLES,
            distributions=[
                {"name": "normal", "params": {"loc": 0.0, "scale": 1.0}},
                {"name": "normal", "params": {"loc": 0.0, "scale": 1.0}},
            ],
            copula={"type": "independent"},
            seed=SEED,
        )
        corr = np.corrcoef(result.samples)[0, 1]
        assert abs(corr) < 0.05, f"Unexpected correlation {corr:.4f} for independent copula"

    def test_three_vars_low_correlation(self) -> None:
        """Check pairwise correlations across three independent variables."""
        result = simulator.simulate(
            n_vars=3,
            n_samples=N_SAMPLES,
            distributions=[
                {"name": "normal", "params": {"loc": 0.0, "scale": 1.0}},
                {"name": "lognormal", "params": {"s": 0.5, "loc": 0.0, "scale": 1.0}},
                {"name": "uniform", "params": {"loc": 0.0, "scale": 1.0}},
            ],
            copula={"type": "independent"},
            seed=SEED,
        )
        corr_matrix = np.corrcoef(result.samples)
        off_diag = corr_matrix[np.triu_indices(3, k=1)]
        assert np.all(np.abs(off_diag) < 0.05)


# ---------------------------------------------------------------------------
# 2. Gaussian copula
# ---------------------------------------------------------------------------


class TestGaussianCopula:
    def test_correlation_approximated_07(self) -> None:
        """Gaussian copula with rho=0.7 should reproduce empirical rho ~ 0.7."""
        target_rho = 0.7
        result = simulator.simulate(
            n_vars=2,
            n_samples=N_SAMPLES,
            distributions=[
                {"name": "normal", "params": {"loc": 0.0, "scale": 1.0}},
                {"name": "normal", "params": {"loc": 0.0, "scale": 1.0}},
            ],
            copula={"type": "gaussian", "corr_matrix": [[1.0, target_rho], [target_rho, 1.0]]},
            seed=SEED,
        )
        empirical_rho = np.corrcoef(result.samples)[0, 1]
        assert abs(empirical_rho - target_rho) < 0.05

    def test_correlation_approximated_minus05(self) -> None:
        """Gaussian copula with rho=-0.5 should reproduce empirical rho ~ -0.5."""
        target_rho = -0.5
        result = simulator.simulate(
            n_vars=2,
            n_samples=N_SAMPLES,
            distributions=[
                {"name": "normal", "params": {"loc": 0.0, "scale": 1.0}},
                {"name": "normal", "params": {"loc": 0.0, "scale": 1.0}},
            ],
            copula={"type": "gaussian", "corr_matrix": [[1.0, target_rho], [target_rho, 1.0]]},
            seed=SEED,
        )
        empirical_rho = np.corrcoef(result.samples)[0, 1]
        assert abs(empirical_rho - target_rho) < 0.05

    def test_result_shape(self) -> None:
        result = simulator.simulate(
            n_vars=2,
            n_samples=500,
            distributions=[
                {"name": "normal", "params": {"loc": 0.0, "scale": 1.0}},
                {"name": "normal", "params": {"loc": 0.0, "scale": 1.0}},
            ],
            copula={"type": "gaussian", "corr_matrix": [[1.0, 0.5], [0.5, 1.0]]},
            seed=0,
        )
        assert result.samples.shape == (2, 500)
        assert result.n_samples == 500
        assert result.seed_used == 0


# ---------------------------------------------------------------------------
# 3. Marginal distributions — mean and variance within 5% tolerance
# ---------------------------------------------------------------------------


class TestMarginalNormal:
    DIST = {"name": "normal", "params": {"loc": 2.0, "scale": 1.5}}

    def test_mean(self) -> None:
        samples = _single_var_samples(self.DIST)
        frozen = st.norm(loc=2.0, scale=1.5)
        _check_mean(samples, float(frozen.mean()))

    def test_variance(self) -> None:
        samples = _single_var_samples(self.DIST)
        frozen = st.norm(loc=2.0, scale=1.5)
        _check_var(samples, float(frozen.var()))


class TestMarginalLognormal:
    # s=0.5, loc=0, scale=1 → mean = exp(0 + 0.5²/2) ≈ 1.133, var ≈ 0.365
    DIST = {"name": "lognormal", "params": {"s": 0.5, "loc": 0.0, "scale": 1.0}}

    def test_mean(self) -> None:
        samples = _single_var_samples(self.DIST)
        frozen = st.lognorm(s=0.5, loc=0.0, scale=1.0)
        _check_mean(samples, float(frozen.mean()))

    def test_variance(self) -> None:
        samples = _single_var_samples(self.DIST)
        frozen = st.lognorm(s=0.5, loc=0.0, scale=1.0)
        _check_var(samples, float(frozen.var()))


class TestMarginalWeibull:
    # c=2, loc=0, scale=1 → mean = Γ(1.5) ≈ 0.886, var ≈ 0.215
    DIST = {"name": "weibull", "params": {"c": 2.0, "loc": 0.0, "scale": 1.0}}

    def test_mean(self) -> None:
        samples = _single_var_samples(self.DIST)
        frozen = st.weibull_min(c=2.0, loc=0.0, scale=1.0)
        _check_mean(samples, float(frozen.mean()))

    def test_variance(self) -> None:
        samples = _single_var_samples(self.DIST)
        frozen = st.weibull_min(c=2.0, loc=0.0, scale=1.0)
        _check_var(samples, float(frozen.var()))


class TestMarginalGEV:
    # c=0.1: finite mean and variance exist (|c| < 1 required for mean, |c| < 0.5 for var)
    DIST = {"name": "gev", "params": {"c": 0.1, "loc": 0.0, "scale": 1.0}}

    def test_mean(self) -> None:
        samples = _single_var_samples(self.DIST)
        frozen = st.genextreme(c=0.1, loc=0.0, scale=1.0)
        _check_mean(samples, float(frozen.mean()))

    def test_variance(self) -> None:
        samples = _single_var_samples(self.DIST)
        frozen = st.genextreme(c=0.1, loc=0.0, scale=1.0)
        _check_var(samples, float(frozen.var()))


# ---------------------------------------------------------------------------
# 4. Student-T copula
# ---------------------------------------------------------------------------


class TestStudentTCopula:
    def test_produces_valid_samples(self) -> None:
        result = simulator.simulate(
            n_vars=2,
            n_samples=N_SAMPLES,
            distributions=[
                {"name": "normal", "params": {"loc": 0.0, "scale": 1.0}},
                {"name": "normal", "params": {"loc": 0.0, "scale": 1.0}},
            ],
            copula={"type": "student_t", "corr_matrix": [[1.0, 0.6], [0.6, 1.0]], "df": 5.0},
            seed=SEED,
        )
        assert result.samples.shape == (2, N_SAMPLES)
        assert np.all(np.isfinite(result.samples))

    def test_correlation_positive(self) -> None:
        """Student-T copula with rho=0.7 should produce positively correlated samples."""
        result = simulator.simulate(
            n_vars=2,
            n_samples=N_SAMPLES,
            distributions=[
                {"name": "normal", "params": {"loc": 0.0, "scale": 1.0}},
                {"name": "normal", "params": {"loc": 0.0, "scale": 1.0}},
            ],
            copula={"type": "student_t", "corr_matrix": [[1.0, 0.7], [0.7, 1.0]], "df": 5.0},
            seed=SEED,
        )
        empirical_rho = np.corrcoef(result.samples)[0, 1]
        assert empirical_rho > 0.5  # positive and substantial


# ---------------------------------------------------------------------------
# 5. Clayton copula
# ---------------------------------------------------------------------------


class TestClaytonCopula:
    def test_produces_valid_samples(self) -> None:
        result = simulator.simulate(
            n_vars=2,
            n_samples=N_SAMPLES,
            distributions=[
                {"name": "uniform", "params": {"loc": 0.0, "scale": 1.0}},
                {"name": "uniform", "params": {"loc": 0.0, "scale": 1.0}},
            ],
            copula={"type": "clayton", "theta": 2.0},
            seed=SEED,
        )
        assert result.samples.shape == (2, N_SAMPLES)
        # U_i should remain in [0, 1] for uniform marginals
        assert np.all(result.samples >= 0.0)
        assert np.all(result.samples <= 1.0)

    def test_positive_dependence(self) -> None:
        """Clayton copula induces positive lower-tail dependence; overall rho > 0."""
        result = simulator.simulate(
            n_vars=2,
            n_samples=N_SAMPLES,
            distributions=[
                {"name": "normal", "params": {"loc": 0.0, "scale": 1.0}},
                {"name": "normal", "params": {"loc": 0.0, "scale": 1.0}},
            ],
            copula={"type": "clayton", "theta": 3.0},
            seed=SEED,
        )
        empirical_rho = np.corrcoef(result.samples)[0, 1]
        assert empirical_rho > 0.3

    def test_multivariate_three_vars(self) -> None:
        """Frailty method supports d > 2."""
        result = simulator.simulate(
            n_vars=3,
            n_samples=N_SAMPLES,
            distributions=[
                {"name": "normal", "params": {"loc": 0.0, "scale": 1.0}},
                {"name": "normal", "params": {"loc": 0.0, "scale": 1.0}},
                {"name": "normal", "params": {"loc": 0.0, "scale": 1.0}},
            ],
            copula={"type": "clayton", "theta": 2.0},
            seed=SEED,
        )
        assert result.samples.shape == (3, N_SAMPLES)
        assert np.all(np.isfinite(result.samples))

    def test_invalid_theta_raises(self) -> None:
        with pytest.raises(ValueError, match="theta > 0"):
            simulator.simulate(
                n_vars=1,
                n_samples=100,
                distributions=[{"name": "normal", "params": {"loc": 0.0, "scale": 1.0}}],
                copula={"type": "clayton", "theta": -1.0},
                seed=0,
            )


# ---------------------------------------------------------------------------
# 6. SimulationResult shape contract
# ---------------------------------------------------------------------------


class TestSimulationResultContract:
    def test_shape_n_vars_by_n_samples(self) -> None:
        result = simulator.simulate(
            n_vars=3,
            n_samples=200,
            distributions=[
                {"name": "normal", "params": {"loc": 0.0, "scale": 1.0}},
                {"name": "lognormal", "params": {"s": 0.5, "loc": 0.0, "scale": 1.0}},
                {"name": "weibull", "params": {"c": 2.0, "loc": 0.0, "scale": 1.0}},
            ],
            copula={"type": "independent"},
            seed=0,
        )
        assert isinstance(result, MCSimulationResult)
        assert result.samples.shape == (3, 200)
        assert result.n_samples == 200
        assert result.seed_used == 0

    def test_seed_none_stored(self) -> None:
        result = simulator.simulate(
            n_vars=1,
            n_samples=100,
            distributions=[{"name": "normal", "params": {"loc": 0.0, "scale": 1.0}}],
            copula={"type": "independent"},
            seed=None,
        )
        assert result.seed_used is None


# ---------------------------------------------------------------------------
# 7. Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_n_samples_over_limit_raises(self) -> None:
        with pytest.raises(ValueError, match="exceeds the limit"):
            simulator.simulate(
                n_vars=1,
                n_samples=MAX_N_SAMPLES + 1,
                distributions=[{"name": "normal", "params": {"loc": 0.0, "scale": 1.0}}],
                copula={"type": "independent"},
                seed=0,
            )

    def test_unsupported_copula_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported copula type"):
            simulator.simulate(
                n_vars=1,
                n_samples=100,
                distributions=[{"name": "normal", "params": {"loc": 0.0, "scale": 1.0}}],
                copula={"type": "frank"},
                seed=0,
            )

    def test_unsupported_distribution_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported distribution"):
            simulator.simulate(
                n_vars=1,
                n_samples=100,
                distributions=[{"name": "pareto", "params": {"b": 2.0}}],
                copula={"type": "independent"},
                seed=0,
            )

    def test_non_pd_corr_matrix_raises(self) -> None:
        with pytest.raises(ValueError, match="positive definite"):
            simulator.simulate(
                n_vars=2,
                n_samples=100,
                distributions=[
                    {"name": "normal", "params": {"loc": 0.0, "scale": 1.0}},
                    {"name": "normal", "params": {"loc": 0.0, "scale": 1.0}},
                ],
                copula={"type": "gaussian", "corr_matrix": [[1.0, 2.0], [2.0, 1.0]]},
                seed=0,
            )
