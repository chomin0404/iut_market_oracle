"""Tests for src/entropy/wasserstein.py — Villani W₂ distance (2003, 2008)."""

from __future__ import annotations

import math

import pytest

from entropy.wasserstein import (
    W2Result,
    w2_beta,
    w2_beta_squared,
    w2_normal,
    w2_normal_squared,
    w2_posterior,
    w2_rolling_mean,
    w2_series,
)
from schemas import PosteriorSummary, PriorSpec

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normal_prior(mean: float = 0.0, std: float = 1.0) -> PriorSpec:
    return PriorSpec(distribution="normal", params={"mean": mean, "std": std})


def _beta_prior(alpha: float = 2.0, beta: float = 2.0) -> PriorSpec:
    return PriorSpec(distribution="beta", params={"alpha": alpha, "beta": beta})


def _normal_posterior(mean: float, variance: float, n: int = 1) -> PosteriorSummary:
    sigma = math.sqrt(variance)
    return PosteriorSummary(
        mean=mean,
        variance=variance,
        credible_interval_95=(mean - 1.96 * sigma, mean + 1.96 * sigma),
        n_evidence=n,
    )


def _beta_posterior(mu: float, var: float, n: int = 10) -> PosteriorSummary:
    """Build a PosteriorSummary consistent with Beta MoM (mu in (0,1), var < mu*(1-mu))."""
    return PosteriorSummary(
        mean=mu,
        variance=var,
        credible_interval_95=(max(0.0, mu - 0.1), min(1.0, mu + 0.1)),
        n_evidence=n,
    )


# ---------------------------------------------------------------------------
# w2_normal — closed form
# ---------------------------------------------------------------------------


class TestW2Normal:
    def test_identity_is_zero(self) -> None:
        """W₂(N(μ,σ²), N(μ,σ²)) = 0."""
        assert math.isclose(w2_normal(1.0, 4.0, 1.0, 4.0), 0.0, abs_tol=1e-15)

    def test_mean_shift_only(self) -> None:
        """σ₁ = σ₂: W₂ = |μ₁ − μ₂|."""
        d = w2_normal(3.0, 1.0, 0.0, 1.0)
        assert math.isclose(d, 3.0, rel_tol=1e-12)

    def test_std_shift_only(self) -> None:
        """μ₁ = μ₂: W₂ = |σ₁ − σ₂|."""
        d = w2_normal(0.0, 9.0, 0.0, 1.0)  # σ₁=3, σ₂=1 → W₂=2
        assert math.isclose(d, 2.0, rel_tol=1e-12)

    def test_both_shifts(self) -> None:
        """W₂² = (μ₁−μ₂)² + (σ₁−σ₂)²."""
        # μ shift = 3, σ₁=2, σ₂=1 → σ shift = 1
        # W₂² = 9 + 1 = 10
        d = w2_normal(3.0, 4.0, 0.0, 1.0)
        assert math.isclose(d, math.sqrt(10.0), rel_tol=1e-12)

    def test_symmetry(self) -> None:
        """W₂(p, q) = W₂(q, p)  — unlike KL divergence."""
        d_pq = w2_normal(2.0, 1.0, 0.0, 4.0)
        d_qp = w2_normal(0.0, 4.0, 2.0, 1.0)
        assert math.isclose(d_pq, d_qp, rel_tol=1e-12)

    def test_non_negative(self) -> None:
        assert w2_normal(-5.0, 0.5, 5.0, 2.0) >= 0.0

    def test_invalid_var_q(self) -> None:
        with pytest.raises(ValueError, match="var_q"):
            w2_normal(0.0, 0.0, 0.0, 1.0)

    def test_invalid_var_p(self) -> None:
        with pytest.raises(ValueError, match="var_p"):
            w2_normal(0.0, 1.0, 0.0, -1.0)

    def test_gaussian_variance_prior(self) -> None:
        """Works with 'variance' key in prior params (not 'std')."""
        # Direct test of w2_normal: var_p=4 → σ_p=2
        d = w2_normal(0.0, 9.0, 0.0, 4.0)  # σ_q=3, σ_p=2 → W₂=1
        assert math.isclose(d, 1.0, rel_tol=1e-12)


class TestW2NormalSquared:
    def test_squared_equals_distance_squared(self) -> None:
        d = w2_normal(2.0, 1.0, 0.0, 4.0)
        sq = w2_normal_squared(2.0, 1.0, 0.0, 4.0)
        assert math.isclose(sq, d**2, rel_tol=1e-12)

    def test_formula_explicit(self) -> None:
        """W₂² = (μ₁−μ₂)² + (σ₁−σ₂)² verified directly."""
        mu_q, var_q, mu_p, var_p = 1.0, 4.0, -1.0, 9.0
        sigma_q, sigma_p = math.sqrt(var_q), math.sqrt(var_p)
        expected = (mu_q - mu_p) ** 2 + (sigma_q - sigma_p) ** 2
        assert math.isclose(w2_normal_squared(mu_q, var_q, mu_p, var_p), expected, rel_tol=1e-12)


# ---------------------------------------------------------------------------
# w2_beta — numerical quantile coupling
# ---------------------------------------------------------------------------


class TestW2Beta:
    def test_identity_is_zero(self) -> None:
        """W₂(Beta(α,β), Beta(α,β)) = 0."""
        d = w2_beta(2.0, 5.0, 2.0, 5.0)
        assert math.isclose(d, 0.0, abs_tol=1e-8)

    def test_non_negative(self) -> None:
        assert w2_beta(1.0, 1.0, 2.0, 2.0) >= 0.0

    def test_symmetry(self) -> None:
        """W₂(p,q) = W₂(q,p)."""
        d_pq = w2_beta(2.0, 5.0, 4.0, 2.0)
        d_qp = w2_beta(4.0, 2.0, 2.0, 5.0)
        assert math.isclose(d_pq, d_qp, rel_tol=1e-6)

    def test_uniform_vs_concentrated(self) -> None:
        """Beta(1,1) is uniform; Beta(50,50) is concentrated near 0.5.

        W₂ > 0 since distributions differ.
        """
        d = w2_beta(1.0, 1.0, 50.0, 50.0)
        assert d > 0.0

    def test_skewed_vs_symmetric(self) -> None:
        """Beta(0.5,5) (left-skewed) vs Beta(5,0.5) (right-skewed): large W₂."""
        d = w2_beta(0.5, 5.0, 5.0, 0.5)
        assert d > 0.3  # Empirical lower bound; means are ~0.09 and ~0.91

    def test_invalid_alpha_q(self) -> None:
        with pytest.raises(ValueError, match="alpha_q"):
            w2_beta(0.0, 1.0, 2.0, 2.0)

    def test_invalid_beta_p(self) -> None:
        with pytest.raises(ValueError, match="beta_p"):
            w2_beta(1.0, 1.0, 2.0, -1.0)

    def test_close_distributions_small_distance(self) -> None:
        """Nearby Beta distributions → small but positive W₂."""
        d1 = w2_beta(5.0, 5.0, 5.0, 5.0)  # same → 0
        d2 = w2_beta(5.0, 5.0, 5.1, 5.1)  # very close
        assert d1 < d2 + 1e-8


class TestW2BetaSquared:
    def test_squared_equals_distance_squared(self) -> None:
        d = w2_beta(2.0, 3.0, 4.0, 5.0)
        sq = w2_beta_squared(2.0, 3.0, 4.0, 5.0)
        assert math.isclose(sq, d**2, rel_tol=1e-8)

    def test_identity_is_zero(self) -> None:
        sq = w2_beta_squared(3.0, 3.0, 3.0, 3.0)
        assert math.isclose(sq, 0.0, abs_tol=1e-10)


# ---------------------------------------------------------------------------
# w2_posterior — dispatcher
# ---------------------------------------------------------------------------


class TestW2Posterior:
    def test_normal_returns_w2result(self) -> None:
        prior = _normal_prior(mean=0.0, std=1.0)
        posterior = _normal_posterior(mean=0.0, variance=1.0)
        result = w2_posterior(posterior, prior)
        assert isinstance(result, W2Result)
        assert result.family == "normal"

    def test_normal_identity(self) -> None:
        """Posterior == prior → W₂ = 0."""
        prior = _normal_prior(mean=0.0, std=1.0)
        posterior = _normal_posterior(mean=0.0, variance=1.0)
        result = w2_posterior(posterior, prior)
        assert math.isclose(result.distance, 0.0, abs_tol=1e-12)
        assert math.isclose(result.squared, 0.0, abs_tol=1e-12)

    def test_normal_mean_shift(self) -> None:
        prior = _normal_prior(mean=0.0, std=1.0)
        posterior = _normal_posterior(mean=3.0, variance=1.0)
        result = w2_posterior(posterior, prior)
        assert math.isclose(result.distance, 3.0, rel_tol=1e-12)

    def test_normal_variance_key(self) -> None:
        """'variance' key in prior.params is accepted."""
        prior = PriorSpec(distribution="normal", params={"mean": 0.0, "variance": 4.0})
        posterior = _normal_posterior(mean=0.0, variance=4.0)
        result = w2_posterior(posterior, prior)
        assert math.isclose(result.distance, 0.0, abs_tol=1e-12)

    def test_normal_missing_std_raises(self) -> None:
        prior = PriorSpec(distribution="normal", params={"mean": 0.0})
        posterior = _normal_posterior(mean=0.0, variance=1.0)
        with pytest.raises(KeyError):
            w2_posterior(posterior, prior)

    def test_beta_returns_w2result(self) -> None:
        prior = _beta_prior(2.0, 2.0)
        posterior = _beta_posterior(mu=0.5, var=0.04)
        result = w2_posterior(posterior, prior)
        assert isinstance(result, W2Result)
        assert result.family == "beta"

    def test_beta_distance_non_negative(self) -> None:
        prior = _beta_prior(3.0, 5.0)
        posterior = _beta_posterior(mu=0.4, var=0.02)
        result = w2_posterior(posterior, prior)
        assert result.distance >= 0.0
        assert result.squared >= 0.0

    def test_squared_consistent_with_distance(self) -> None:
        prior = _normal_prior(mean=1.0, std=2.0)
        posterior = _normal_posterior(mean=3.0, variance=9.0)
        result = w2_posterior(posterior, prior)
        assert math.isclose(result.squared, result.distance**2, rel_tol=1e-12)

    def test_unsupported_family_raises(self) -> None:
        prior = PriorSpec(distribution="uniform", params={"low": 0.0, "high": 1.0})
        posterior = _normal_posterior(mean=0.5, variance=0.1)
        with pytest.raises(ValueError, match="Unsupported"):
            w2_posterior(posterior, prior)

    def test_gaussian_alias(self) -> None:
        """'gaussian' is treated the same as 'normal'."""
        prior = PriorSpec(distribution="gaussian", params={"mean": 0.0, "std": 1.0})
        posterior = _normal_posterior(mean=0.0, variance=1.0)
        result = w2_posterior(posterior, prior)
        assert result.family == "normal"
        assert math.isclose(result.distance, 0.0, abs_tol=1e-12)


# ---------------------------------------------------------------------------
# w2_series
# ---------------------------------------------------------------------------


class TestW2Series:
    def test_empty_posteriors(self) -> None:
        prior = _normal_prior()
        assert w2_series([], prior) == []

    def test_length_matches_posteriors(self) -> None:
        prior = _normal_prior(mean=0.0, std=1.0)
        posteriors = [_normal_posterior(mean=float(i), variance=1.0) for i in range(5)]
        result = w2_series(posteriors, prior)
        assert len(result) == 5

    def test_consistent_with_w2_posterior(self) -> None:
        """Each element of w2_series matches a direct w2_posterior call."""
        prior = _normal_prior(mean=0.0, std=2.0)
        posteriors = [_normal_posterior(mean=float(i), variance=4.0) for i in range(4)]
        series = w2_series(posteriors, prior)
        for i, p in enumerate(posteriors):
            expected = w2_posterior(p, prior).distance
            assert math.isclose(series[i], expected, rel_tol=1e-12)

    def test_monotone_increasing_shift(self) -> None:
        """As posterior drifts from prior, W₂ should increase."""
        prior = _normal_prior(mean=0.0, std=1.0)
        posteriors = [_normal_posterior(mean=float(i), variance=1.0) for i in range(5)]
        series = w2_series(posteriors, prior)
        for a, b in zip(series, series[1:]):
            assert b > a

    def test_returns_floats(self) -> None:
        prior = _normal_prior()
        posteriors = [_normal_posterior(mean=1.0, variance=2.0)]
        result = w2_series(posteriors, prior)
        assert all(isinstance(v, float) for v in result)


# ---------------------------------------------------------------------------
# w2_rolling_mean
# ---------------------------------------------------------------------------


class TestW2RollingMean:
    def test_window_1_identity(self) -> None:
        """Window=1: rolling mean = original series."""
        vals = [1.0, 2.0, 3.0, 4.0]
        result = w2_rolling_mean(vals, window=1)
        assert result == vals

    def test_window_equals_length(self) -> None:
        """Window = len: single output = mean of all values."""
        vals = [1.0, 3.0, 5.0]
        result = w2_rolling_mean(vals, window=3)
        assert len(result) == 1
        assert math.isclose(result[0], 3.0, rel_tol=1e-12)

    def test_window_2_values(self) -> None:
        vals = [0.0, 2.0, 4.0, 6.0]
        result = w2_rolling_mean(vals, window=2)
        expected = [1.0, 3.0, 5.0]
        assert len(result) == 3
        for a, b in zip(result, expected):
            assert math.isclose(a, b, rel_tol=1e-12)

    def test_output_length(self) -> None:
        """Output length = len(vals) − window + 1."""
        for n, w in [(10, 3), (5, 5), (7, 1), (4, 4)]:
            vals = list(range(n))
            result = w2_rolling_mean(vals, window=w)
            assert len(result) == n - w + 1

    def test_too_short_returns_empty(self) -> None:
        """len(vals) < window → empty list."""
        assert w2_rolling_mean([1.0, 2.0], window=5) == []

    def test_empty_input_returns_empty(self) -> None:
        assert w2_rolling_mean([], window=3) == []

    def test_invalid_window_raises(self) -> None:
        with pytest.raises(ValueError, match="window"):
            w2_rolling_mean([1.0, 2.0], window=0)

    def test_non_negative_for_w2_input(self) -> None:
        """Rolling mean of non-negative W₂ values is non-negative."""
        prior = _normal_prior(mean=0.0, std=1.0)
        posteriors = [_normal_posterior(mean=float(i) * 0.5, variance=1.0) for i in range(8)]
        series = w2_series(posteriors, prior)
        means = w2_rolling_mean(series, window=3)
        assert all(v >= 0.0 for v in means)


# ---------------------------------------------------------------------------
# W₂ vs KL comparison: symmetry property
# ---------------------------------------------------------------------------


class TestW2SymmetryVsKL:
    """W₂ is symmetric; KL is not.  Confirm numerically."""

    def test_normal_symmetry_holds(self) -> None:
        """W₂(N(a), N(b)) = W₂(N(b), N(a)) exactly."""
        d_ab = w2_normal(2.0, 1.0, 0.0, 4.0)
        d_ba = w2_normal(0.0, 4.0, 2.0, 1.0)
        assert math.isclose(d_ab, d_ba, rel_tol=1e-14)

    def test_beta_symmetry_holds(self) -> None:
        """W₂(Beta(a), Beta(b)) ≈ W₂(Beta(b), Beta(a)) within quadrature error."""
        d_ab = w2_beta(2.0, 8.0, 8.0, 2.0)
        d_ba = w2_beta(8.0, 2.0, 2.0, 8.0)
        assert math.isclose(d_ab, d_ba, rel_tol=1e-6)

    def test_kl_is_asymmetric(self) -> None:
        """Control: KL(N(a)||N(b)) ≠ KL(N(b)||N(a)) in general."""
        from entropy.monitor import kl_normal

        kl_ab = kl_normal(2.0, 1.0, 0.0, 4.0)
        kl_ba = kl_normal(0.0, 4.0, 2.0, 1.0)
        assert not math.isclose(kl_ab, kl_ba, rel_tol=1e-3)
