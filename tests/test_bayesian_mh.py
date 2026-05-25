"""Tests for src/bayesian/mh.py (Metropolis–Hastings sampler)."""

from __future__ import annotations

import numpy as np
import pytest

from bayesian.kernels import GaussianRWKernel
from bayesian.mh import MCMCResult, run_mh
from bayesian.sampler import TargetDistribution

# ---------------------------------------------------------------------------
# Minimal target distributions for testing
# ---------------------------------------------------------------------------


class GaussianTarget(TargetDistribution):
    """Isotropic Gaussian N(mu, sigma^2 I)."""

    def __init__(self, mu: np.ndarray, sigma: float) -> None:
        self._mu = np.asarray(mu, dtype=float)
        self._sigma = sigma

    @property
    def dim(self) -> int:
        return self._mu.size

    def log_prob(self, x: np.ndarray) -> float:
        delta = x - self._mu
        return float(-0.5 * np.dot(delta, delta) / self._sigma**2)


class BoundedUniformTarget(TargetDistribution):
    """Uniform on [0, 1]^d — returns 0 inside, -inf outside."""

    def __init__(self, d: int) -> None:
        self._d = d

    @property
    def dim(self) -> int:
        return self._d

    def log_prob(self, x: np.ndarray) -> float:
        if np.all((x >= 0.0) & (x <= 1.0)):
            return 0.0
        return float("-inf")


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_run_mh_rejects_n_samples_zero() -> None:
    target = GaussianTarget(np.zeros(1), 1.0)
    kernel = GaussianRWKernel(step_size=0.5)
    with pytest.raises(ValueError, match="n_samples"):
        run_mh(target, kernel, np.zeros(1), n_samples=0, rng=np.random.default_rng(0))


def test_run_mh_rejects_negative_burn_in() -> None:
    target = GaussianTarget(np.zeros(1), 1.0)
    kernel = GaussianRWKernel(step_size=0.5)
    with pytest.raises(ValueError, match="burn_in"):
        run_mh(target, kernel, np.zeros(1), n_samples=10, rng=np.random.default_rng(0), burn_in=-1)


def test_run_mh_rejects_thin_zero() -> None:
    target = GaussianTarget(np.zeros(1), 1.0)
    kernel = GaussianRWKernel(step_size=0.5)
    with pytest.raises(ValueError, match="thin"):
        run_mh(target, kernel, np.zeros(1), n_samples=10, rng=np.random.default_rng(0), thin=0)


def test_run_mh_rejects_initial_wrong_shape() -> None:
    target = GaussianTarget(np.zeros(3), 1.0)
    kernel = GaussianRWKernel(step_size=0.5)
    with pytest.raises(ValueError, match="shape"):
        run_mh(target, kernel, np.zeros(2), n_samples=10, rng=np.random.default_rng(0))


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------


def test_result_is_mcmc_result() -> None:
    target = GaussianTarget(np.zeros(2), 1.0)
    kernel = GaussianRWKernel(step_size=0.5)
    result = run_mh(target, kernel, np.zeros(2), n_samples=20, rng=np.random.default_rng(0))
    assert isinstance(result, MCMCResult)


def test_samples_shape() -> None:
    d = 3
    n = 50
    target = GaussianTarget(np.zeros(d), 1.0)
    kernel = GaussianRWKernel(step_size=0.5)
    result = run_mh(target, kernel, np.zeros(d), n_samples=n, rng=np.random.default_rng(0))
    assert result.samples.shape == (n, d)


def test_acceptance_rate_in_unit_interval() -> None:
    target = GaussianTarget(np.zeros(2), 1.0)
    kernel = GaussianRWKernel(step_size=0.5)
    result = run_mh(target, kernel, np.zeros(2), n_samples=200, rng=np.random.default_rng(1))
    assert 0.0 <= result.acceptance_rate <= 1.0


def test_n_accepted_consistent() -> None:
    target = GaussianTarget(np.zeros(2), 1.0)
    kernel = GaussianRWKernel(step_size=0.5)
    result = run_mh(target, kernel, np.zeros(2), n_samples=100, rng=np.random.default_rng(2))
    assert result.n_accepted == round(result.acceptance_rate * result.n_total)


def test_n_total_with_burn_in_and_thin() -> None:
    target = GaussianTarget(np.zeros(1), 1.0)
    kernel = GaussianRWKernel(step_size=0.5)
    result = run_mh(
        target, kernel, np.zeros(1), n_samples=10, rng=np.random.default_rng(0),
        burn_in=50, thin=3,
    )
    assert result.n_total == 50 + 10 * 3
    assert result.samples.shape == (10, 1)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_reproducible_with_same_seed() -> None:
    target = GaussianTarget(np.zeros(2), 1.0)
    kernel = GaussianRWKernel(step_size=0.3)
    r1 = run_mh(target, kernel, np.zeros(2), n_samples=30, rng=np.random.default_rng(99))
    r2 = run_mh(target, kernel, np.zeros(2), n_samples=30, rng=np.random.default_rng(99))
    np.testing.assert_array_equal(r1.samples, r2.samples)
    assert r1.acceptance_rate == r2.acceptance_rate


# ---------------------------------------------------------------------------
# Statistical correctness — 1D Gaussian
# ---------------------------------------------------------------------------


def test_1d_gaussian_empirical_mean() -> None:
    """Empirical mean of chain should converge to true mean.

    Initial point is set at the true mean to avoid cold-start mixing issues.
    """
    mu_true = np.array([3.0])
    sigma_true = 1.0
    target = GaussianTarget(mu_true, sigma_true)
    kernel = GaussianRWKernel(step_size=1.0)
    result = run_mh(
        target, kernel, mu_true.copy(), n_samples=8000, rng=np.random.default_rng(0),
        burn_in=200,
    )
    # MCMC 自己相関により有効サンプルサイズは n_samples より小さい。
    # 許容誤差は 3σ_eff ≈ 3 / sqrt(ESS) ≈ 0.15 を目安とする。
    assert abs(result.samples[:, 0].mean() - mu_true[0]) < 0.15


def test_1d_gaussian_empirical_std() -> None:
    """Empirical std of chain should converge to true std."""
    sigma_true = 2.0
    target = GaussianTarget(np.zeros(1), sigma_true)
    kernel = GaussianRWKernel(step_size=1.0)
    result = run_mh(
        target, kernel, np.zeros(1), n_samples=8000, rng=np.random.default_rng(1),
        burn_in=500,
    )
    assert abs(result.samples[:, 0].std() - sigma_true) < 0.2


# ---------------------------------------------------------------------------
# Bounded support — uniform target
# ---------------------------------------------------------------------------


def test_bounded_uniform_samples_in_support() -> None:
    """All samples must lie in [0, 1]^2 for a uniform target."""
    target = BoundedUniformTarget(d=2)
    kernel = GaussianRWKernel(step_size=0.2)
    result = run_mh(
        target, kernel, np.array([0.5, 0.5]), n_samples=500, rng=np.random.default_rng(5),
        burn_in=100,
    )
    assert np.all(result.samples >= 0.0)
    assert np.all(result.samples <= 1.0)


# ---------------------------------------------------------------------------
# Very large step size → low acceptance
# ---------------------------------------------------------------------------


def test_large_step_size_low_acceptance() -> None:
    target = GaussianTarget(np.zeros(1), 1.0)
    kernel = GaussianRWKernel(step_size=50.0)
    result = run_mh(
        target, kernel, np.zeros(1), n_samples=500, rng=np.random.default_rng(3),
    )
    assert result.acceptance_rate < 0.2


# ---------------------------------------------------------------------------
# Very small step size → high acceptance
# ---------------------------------------------------------------------------


def test_small_step_size_high_acceptance() -> None:
    target = GaussianTarget(np.zeros(1), 1.0)
    kernel = GaussianRWKernel(step_size=0.001)
    result = run_mh(
        target, kernel, np.zeros(1), n_samples=500, rng=np.random.default_rng(4),
    )
    assert result.acceptance_rate > 0.9
