"""Tests for src/bayesian/sampler.py and src/bayesian/kernels.py."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.stats as st

from bayesian.kernels import GaussianRWKernel
from bayesian.sampler import ProposalKernel, TargetDistribution

RNG = np.random.default_rng(0)


# ---------------------------------------------------------------------------
# Abstract interface smoke tests
# ---------------------------------------------------------------------------


def test_target_distribution_is_abstract() -> None:
    with pytest.raises(TypeError):
        TargetDistribution()  # type: ignore[abstract]


def test_proposal_kernel_is_abstract() -> None:
    with pytest.raises(TypeError):
        ProposalKernel()  # type: ignore[abstract]


def test_grad_log_prob_raises_by_default() -> None:
    class MinimalTarget(TargetDistribution):
        @property
        def dim(self) -> int:
            return 1

        def log_prob(self, x: np.ndarray) -> float:
            return float(-0.5 * np.dot(x, x))

    t = MinimalTarget()
    with pytest.raises(NotImplementedError):
        t.grad_log_prob(np.zeros(1))


# ---------------------------------------------------------------------------
# GaussianRWKernel — construction validation
# ---------------------------------------------------------------------------


def test_isotropic_requires_positive_step_size() -> None:
    with pytest.raises(ValueError, match="step_size must be > 0"):
        GaussianRWKernel(step_size=0.0)

    with pytest.raises(ValueError, match="step_size must be > 0"):
        GaussianRWKernel(step_size=-1.0)


def test_must_specify_exactly_one_of_step_size_or_cov() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        GaussianRWKernel()  # neither

    with pytest.raises(ValueError, match="exactly one"):
        GaussianRWKernel(step_size=0.1, cov=np.eye(2))  # both


def test_cov_must_be_square() -> None:
    with pytest.raises(ValueError, match="square"):
        GaussianRWKernel(cov=np.ones((2, 3)))


def test_cov_must_be_positive_definite() -> None:
    with pytest.raises(Exception):  # scipy LinAlgError
        GaussianRWKernel(cov=np.array([[1.0, 2.0], [2.0, 1.0]]))  # not PD


# ---------------------------------------------------------------------------
# GaussianRWKernel — is_symmetric
# ---------------------------------------------------------------------------


def test_is_symmetric_isotropic() -> None:
    k = GaussianRWKernel(step_size=0.5)
    assert k.is_symmetric is True


def test_is_symmetric_full_cov() -> None:
    k = GaussianRWKernel(cov=np.eye(3))
    assert k.is_symmetric is True


# ---------------------------------------------------------------------------
# GaussianRWKernel — propose
# ---------------------------------------------------------------------------


def test_propose_isotropic_shape() -> None:
    k = GaussianRWKernel(step_size=0.1)
    x = np.zeros(4)
    x_new = k.propose(x, RNG)
    assert x_new.shape == (4,)


def test_propose_full_cov_shape() -> None:
    cov = np.diag([1.0, 2.0, 0.5])
    k = GaussianRWKernel(cov=cov)
    x = np.ones(3)
    x_new = k.propose(x, RNG)
    assert x_new.shape == (3,)


def test_propose_reproducible_with_seed() -> None:
    k = GaussianRWKernel(step_size=0.5)
    x = np.zeros(3)
    x1 = k.propose(x, np.random.default_rng(42))
    x2 = k.propose(x, np.random.default_rng(42))
    np.testing.assert_array_equal(x1, x2)


def test_propose_isotropic_empirical_std() -> None:
    """Empirical std of increments should be close to step_size."""
    sigma = 0.3
    k = GaussianRWKernel(step_size=sigma)
    x = np.zeros(1)
    rng = np.random.default_rng(7)
    samples = np.array([k.propose(x, rng)[0] for _ in range(5000)])
    assert abs(samples.std() - sigma) < 0.02


# ---------------------------------------------------------------------------
# GaussianRWKernel — log_transition_prob
# ---------------------------------------------------------------------------


def test_log_transition_prob_isotropic_matches_scipy() -> None:
    sigma = 0.4
    k = GaussianRWKernel(step_size=sigma)
    x = np.array([1.0, 2.0, 3.0])
    x_new = np.array([1.1, 1.9, 3.2])
    expected = float(st.multivariate_normal.logpdf(x_new, mean=x, cov=sigma**2 * np.eye(3)))
    assert abs(k.log_transition_prob(x_new, x) - expected) < 1e-10


def test_log_transition_prob_full_cov_matches_scipy() -> None:
    cov = np.array([[2.0, 0.5], [0.5, 1.0]])
    k = GaussianRWKernel(cov=cov)
    x = np.array([0.0, 0.0])
    x_new = np.array([0.3, -0.1])
    expected = float(st.multivariate_normal.logpdf(x_new, mean=x, cov=cov))
    assert abs(k.log_transition_prob(x_new, x) - expected) < 1e-10


def test_log_transition_prob_symmetry_isotropic() -> None:
    """log q(x'|x) == log q(x|x') for isotropic kernel."""
    k = GaussianRWKernel(step_size=0.2)
    x = np.array([1.0, -0.5])
    x_new = np.array([0.8, 0.1])
    assert abs(k.log_transition_prob(x_new, x) - k.log_transition_prob(x, x_new)) < 1e-12


def test_log_transition_prob_symmetry_full_cov() -> None:
    """log q(x'|x) == log q(x|x') for full-covariance kernel."""
    cov = np.array([[3.0, 1.0], [1.0, 2.0]])
    k = GaussianRWKernel(cov=cov)
    x = np.array([1.0, 2.0])
    x_new = np.array([0.5, 2.5])
    assert abs(k.log_transition_prob(x_new, x) - k.log_transition_prob(x, x_new)) < 1e-12
