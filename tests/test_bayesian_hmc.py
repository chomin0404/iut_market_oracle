"""Tests for src/bayesian/hmc.py (Hamiltonian Monte Carlo sampler)."""

from __future__ import annotations

import numpy as np
import pytest

from bayesian.hmc import HMCResult, run_hmc
from bayesian.sampler import TargetDistribution

# ---------------------------------------------------------------------------
# Target distributions
# ---------------------------------------------------------------------------


class GaussianTarget(TargetDistribution):
    """Isotropic Gaussian N(mu, sigma^2 I) with analytic gradient."""

    def __init__(self, mu: np.ndarray, sigma: float) -> None:
        self._mu = np.asarray(mu, dtype=float)
        self._sigma = sigma

    @property
    def dim(self) -> int:
        return self._mu.size

    def log_prob(self, x: np.ndarray) -> float:
        delta = x - self._mu
        return float(-0.5 * np.dot(delta, delta) / self._sigma**2)

    def grad_log_prob(self, x: np.ndarray) -> np.ndarray:
        return -(x - self._mu) / self._sigma**2


class NoGradTarget(TargetDistribution):
    """Target that does not implement grad_log_prob."""

    @property
    def dim(self) -> int:
        return 1

    def log_prob(self, x: np.ndarray) -> float:
        return float(-0.5 * x[0] ** 2)


class BananaTwist(TargetDistribution):
    """Non-Gaussian banana-shaped distribution in 2D."""

    def __init__(self, b: float = 0.1) -> None:
        self._b = b

    @property
    def dim(self) -> int:
        return 2

    def log_prob(self, x: np.ndarray) -> float:
        # log p̃(x₁, x₂) = -x₁²/2 - (x₂ + b·x₁²)²/2
        return float(-0.5 * x[0] ** 2 - 0.5 * (x[1] + self._b * x[0] ** 2) ** 2)

    def grad_log_prob(self, x: np.ndarray) -> np.ndarray:
        g1 = -x[0] - 2.0 * self._b * x[0] * (x[1] + self._b * x[0] ** 2)
        g2 = -(x[1] + self._b * x[0] ** 2)
        return np.array([g1, g2])


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_rejects_nonpositive_step_size() -> None:
    t = GaussianTarget(np.zeros(1), 1.0)
    with pytest.raises(ValueError, match="step_size"):
        run_hmc(t, step_size=0.0, n_leapfrog=5, initial=np.zeros(1),
                n_samples=10, rng=np.random.default_rng(0))


def test_rejects_zero_n_leapfrog() -> None:
    t = GaussianTarget(np.zeros(1), 1.0)
    with pytest.raises(ValueError, match="n_leapfrog"):
        run_hmc(t, step_size=0.1, n_leapfrog=0, initial=np.zeros(1),
                n_samples=10, rng=np.random.default_rng(0))


def test_rejects_zero_n_samples() -> None:
    t = GaussianTarget(np.zeros(1), 1.0)
    with pytest.raises(ValueError, match="n_samples"):
        run_hmc(t, step_size=0.1, n_leapfrog=5, initial=np.zeros(1),
                n_samples=0, rng=np.random.default_rng(0))


def test_rejects_negative_burn_in() -> None:
    t = GaussianTarget(np.zeros(1), 1.0)
    with pytest.raises(ValueError, match="burn_in"):
        run_hmc(t, step_size=0.1, n_leapfrog=5, initial=np.zeros(1),
                n_samples=10, rng=np.random.default_rng(0), burn_in=-1)


def test_rejects_thin_zero() -> None:
    t = GaussianTarget(np.zeros(1), 1.0)
    with pytest.raises(ValueError, match="thin"):
        run_hmc(t, step_size=0.1, n_leapfrog=5, initial=np.zeros(1),
                n_samples=10, rng=np.random.default_rng(0), thin=0)


def test_rejects_initial_wrong_shape() -> None:
    t = GaussianTarget(np.zeros(3), 1.0)
    with pytest.raises(ValueError, match="shape"):
        run_hmc(t, step_size=0.1, n_leapfrog=5, initial=np.zeros(2),
                n_samples=10, rng=np.random.default_rng(0))


def test_rejects_mass_wrong_shape() -> None:
    t = GaussianTarget(np.zeros(2), 1.0)
    with pytest.raises(ValueError, match="mass shape"):
        run_hmc(t, step_size=0.1, n_leapfrog=5, initial=np.zeros(2),
                n_samples=10, rng=np.random.default_rng(0),
                mass=np.ones(3))


def test_rejects_nonpositive_mass() -> None:
    t = GaussianTarget(np.zeros(2), 1.0)
    with pytest.raises(ValueError, match="mass entries"):
        run_hmc(t, step_size=0.1, n_leapfrog=5, initial=np.zeros(2),
                n_samples=10, rng=np.random.default_rng(0),
                mass=np.array([1.0, 0.0]))


def test_raises_if_grad_not_implemented() -> None:
    t = NoGradTarget()
    with pytest.raises(NotImplementedError):
        run_hmc(t, step_size=0.1, n_leapfrog=5, initial=np.zeros(1),
                n_samples=10, rng=np.random.default_rng(0))


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------


def test_result_type() -> None:
    t = GaussianTarget(np.zeros(2), 1.0)
    r = run_hmc(t, step_size=0.2, n_leapfrog=5, initial=np.zeros(2),
                n_samples=20, rng=np.random.default_rng(0))
    assert isinstance(r, HMCResult)


def test_samples_shape() -> None:
    d, n = 4, 50
    t = GaussianTarget(np.zeros(d), 1.0)
    r = run_hmc(t, step_size=0.2, n_leapfrog=5, initial=np.zeros(d),
                n_samples=n, rng=np.random.default_rng(0))
    assert r.samples.shape == (n, d)


def test_acceptance_rate_in_unit_interval() -> None:
    t = GaussianTarget(np.zeros(2), 1.0)
    r = run_hmc(t, step_size=0.2, n_leapfrog=5, initial=np.zeros(2),
                n_samples=200, rng=np.random.default_rng(1))
    assert 0.0 <= r.acceptance_rate <= 1.0


def test_n_total_with_burn_in_and_thin() -> None:
    t = GaussianTarget(np.zeros(1), 1.0)
    r = run_hmc(t, step_size=0.2, n_leapfrog=3, initial=np.zeros(1),
                n_samples=10, rng=np.random.default_rng(0),
                burn_in=50, thin=3)
    assert r.n_total == 50 + 10 * 3
    assert r.samples.shape == (10, 1)


def test_n_accepted_consistent_with_rate() -> None:
    t = GaussianTarget(np.zeros(2), 1.0)
    r = run_hmc(t, step_size=0.2, n_leapfrog=5, initial=np.zeros(2),
                n_samples=100, rng=np.random.default_rng(2))
    assert r.n_accepted == round(r.acceptance_rate * r.n_total)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_reproducible_with_same_seed() -> None:
    t = GaussianTarget(np.zeros(2), 1.0)
    r1 = run_hmc(t, step_size=0.2, n_leapfrog=5, initial=np.zeros(2),
                 n_samples=30, rng=np.random.default_rng(7))
    r2 = run_hmc(t, step_size=0.2, n_leapfrog=5, initial=np.zeros(2),
                 n_samples=30, rng=np.random.default_rng(7))
    np.testing.assert_array_equal(r1.samples, r2.samples)


# ---------------------------------------------------------------------------
# Statistical correctness — 2D Gaussian
# ---------------------------------------------------------------------------


def test_2d_gaussian_empirical_mean() -> None:
    """Empirical mean should be close to true mean (0, 0)."""
    t = GaussianTarget(np.zeros(2), 1.0)
    r = run_hmc(t, step_size=0.3, n_leapfrog=10, initial=np.zeros(2),
                n_samples=3000, rng=np.random.default_rng(0), burn_in=200)
    assert np.all(np.abs(r.samples.mean(axis=0)) < 0.1)


def test_2d_gaussian_empirical_std() -> None:
    """Empirical std should be close to true std (1.5, 1.5)."""
    sigma = 1.5
    t = GaussianTarget(np.zeros(2), sigma)
    r = run_hmc(t, step_size=0.4, n_leapfrog=10, initial=np.zeros(2),
                n_samples=3000, rng=np.random.default_rng(1), burn_in=200)
    assert np.all(np.abs(r.samples.std(axis=0) - sigma) < 0.15)


# ---------------------------------------------------------------------------
# Non-Gaussian target — banana distribution
# ---------------------------------------------------------------------------


def test_banana_runs_without_error() -> None:
    t = BananaTwist(b=0.1)
    r = run_hmc(t, step_size=0.15, n_leapfrog=20, initial=np.zeros(2),
                n_samples=500, rng=np.random.default_rng(3), burn_in=100)
    assert r.samples.shape == (500, 2)
    assert 0.0 < r.acceptance_rate <= 1.0


# ---------------------------------------------------------------------------
# Mass matrix
# ---------------------------------------------------------------------------


def test_custom_mass_matrix_runs() -> None:
    t = GaussianTarget(np.zeros(2), 1.0)
    r = run_hmc(t, step_size=0.2, n_leapfrog=5, initial=np.zeros(2),
                n_samples=100, rng=np.random.default_rng(5),
                mass=np.array([1.0, 2.0]))
    assert r.samples.shape == (100, 2)
    assert 0.0 <= r.acceptance_rate <= 1.0


# ---------------------------------------------------------------------------
# Acceptance rate behaviour
# ---------------------------------------------------------------------------


def test_small_step_size_high_acceptance() -> None:
    """Very small ε → near-perfect energy conservation → high acceptance."""
    t = GaussianTarget(np.zeros(2), 1.0)
    r = run_hmc(t, step_size=0.01, n_leapfrog=5, initial=np.zeros(2),
                n_samples=200, rng=np.random.default_rng(9))
    assert r.acceptance_rate > 0.9


def test_large_step_size_low_acceptance() -> None:
    """Very large ε → large energy error → low acceptance."""
    t = GaussianTarget(np.zeros(2), 1.0)
    r = run_hmc(t, step_size=5.0, n_leapfrog=5, initial=np.zeros(2),
                n_samples=200, rng=np.random.default_rng(10))
    assert r.acceptance_rate < 0.5
