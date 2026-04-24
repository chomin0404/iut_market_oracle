"""Tests for src/matroid/log_concavity.py (T1200)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from matroid.log_concavity import compute_log_concave_weights
from schemas import MatroidLogConcavityResult

# ---------------------------------------------------------------------------
# Structural invariants
# ---------------------------------------------------------------------------


class TestStructuralInvariants:
    def test_subset_sizes_range(self):
        result = compute_log_concave_weights(10)
        assert result.subset_sizes == list(range(11))

    def test_all_series_length_n_plus_one(self):
        n = 8
        result = compute_log_concave_weights(n)
        assert len(result.probability_mass) == n + 1
        assert len(result.log_probability) == n + 1
        assert len(result.subset_sizes) == n + 1

    def test_log_concavity_checks_length_n_minus_one(self):
        n = 7
        result = compute_log_concave_weights(n)
        assert len(result.log_concavity_checks) == n - 1

    def test_params_stored(self):
        result = compute_log_concave_weights(5, rank_weight=0.6, corank_weight=1.4)
        assert result.n_assets == 5
        assert result.rank_weight == pytest.approx(0.6)
        assert result.corank_weight == pytest.approx(1.4)

    def test_n_assets_one(self):
        """Edge case: n=1 has subsets of size 0 and 1; checks list is empty."""
        result = compute_log_concave_weights(1)
        assert result.n_assets == 1
        assert len(result.subset_sizes) == 2
        assert result.log_concavity_checks == []
        assert result.is_log_concave is True


# ---------------------------------------------------------------------------
# Probability mass properties
# ---------------------------------------------------------------------------


class TestProbabilityMass:
    def test_sums_to_one(self):
        result = compute_log_concave_weights(10)
        assert math.isclose(sum(result.probability_mass), 1.0, rel_tol=1e-9)

    def test_all_positive(self):
        result = compute_log_concave_weights(10)
        assert all(p > 0 for p in result.probability_mass)

    def test_mode_shifts_with_rank_weight(self):
        """Higher rank_weight shifts probability mass toward larger subsets."""
        low = compute_log_concave_weights(10, rank_weight=0.2, corank_weight=1.8)
        high = compute_log_concave_weights(10, rank_weight=1.8, corank_weight=0.2)
        mode_low = max(range(11), key=lambda k: low.probability_mass[k])
        mode_high = max(range(11), key=lambda k: high.probability_mass[k])
        assert mode_high > mode_low

    def test_symmetric_weights_gives_symmetric_pmf(self):
        """alpha == beta implies PMF is symmetric around k = n/2."""
        n = 10
        result = compute_log_concave_weights(n, rank_weight=1.0, corank_weight=1.0)
        p = result.probability_mass
        # p[k] == p[n - k] up to floating-point rounding
        for k in range(n + 1):
            assert p[k] == pytest.approx(p[n - k], rel=1e-9)

    def test_log_probability_consistent_with_probability_mass(self):
        result = compute_log_concave_weights(8)
        for p, lp in zip(result.probability_mass, result.log_probability):
            assert lp == pytest.approx(math.log(p), rel=1e-6)


# ---------------------------------------------------------------------------
# Log-concavity correctness
# ---------------------------------------------------------------------------


class TestLogConcavity:
    def test_standard_params_are_log_concave(self):
        result = compute_log_concave_weights(10, rank_weight=0.8, corank_weight=1.2)
        assert result.is_log_concave is True
        assert all(result.log_concavity_checks)

    def test_equal_weights_are_log_concave(self):
        result = compute_log_concave_weights(15, rank_weight=1.0, corank_weight=1.0)
        assert result.is_log_concave is True

    def test_extreme_weights_are_log_concave(self):
        """Binomial PMF is always log-concave regardless of p."""
        result = compute_log_concave_weights(20, rank_weight=0.01, corank_weight=99.0)
        assert result.is_log_concave is True

    def test_log_concavity_check_definition(self):
        """Each check must equal b_k^2 >= b_{k-1} * b_{k+1}."""
        result = compute_log_concave_weights(6)
        p = result.probability_mass
        for i, ok in enumerate(result.log_concavity_checks):
            k = i + 1  # checks start at k=1
            expected = p[k] ** 2 >= p[k - 1] * p[k + 1] - 1e-12
            assert ok == expected

    def test_large_n_no_overflow(self):
        """n=100 should complete without overflow (log-space implementation)."""
        result = compute_log_concave_weights(100)
        assert result.is_log_concave is True
        assert math.isclose(sum(result.probability_mass), 1.0, rel_tol=1e-9)

    def test_is_log_concave_consistent_with_checks(self):
        result = compute_log_concave_weights(12)
        assert result.is_log_concave == all(result.log_concavity_checks)


# ---------------------------------------------------------------------------
# Numerical properties
# ---------------------------------------------------------------------------


class TestNumericalProperties:
    def test_log_probability_is_concave_curve(self):
        """ln(b_k) should form a concave sequence (second differences <= 0)."""
        result = compute_log_concave_weights(12)
        lp = result.log_probability
        second_diffs = [lp[k + 1] - 2 * lp[k] + lp[k - 1] for k in range(1, len(lp) - 1)]
        assert all(d <= 1e-10 for d in second_diffs)

    def test_binomial_pmf_equivalence(self):
        """b_k / sum(b_j) == Binomial(n, p=alpha/(alpha+beta)) PMF."""
        from scipy.stats import binom

        n, alpha, beta = 12, 0.8, 1.2
        p_binom = alpha / (alpha + beta)
        result = compute_log_concave_weights(n, rank_weight=alpha, corank_weight=beta)
        expected = [binom.pmf(k, n, p_binom) for k in range(n + 1)]
        np.testing.assert_allclose(result.probability_mass, expected, rtol=1e-9)


# ---------------------------------------------------------------------------
# Invalid inputs
# ---------------------------------------------------------------------------


class TestInvalidInputs:
    def test_n_assets_zero_raises(self):
        with pytest.raises(ValueError, match="n_assets must be >= 1"):
            compute_log_concave_weights(0)

    def test_rank_weight_zero_raises(self):
        with pytest.raises(ValueError, match="rank_weight must be > 0"):
            compute_log_concave_weights(5, rank_weight=0.0)

    def test_corank_weight_negative_raises(self):
        with pytest.raises(ValueError, match="corank_weight must be > 0"):
            compute_log_concave_weights(5, corank_weight=-1.0)


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


class TestMatroidLogConcavityResultSchema:
    def test_probability_mass_length_mismatch_raises(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="probability_mass length"):
            MatroidLogConcavityResult(
                n_assets=3,
                rank_weight=0.8,
                corank_weight=1.2,
                subset_sizes=[0, 1, 2, 3],
                probability_mass=[0.25, 0.5],  # wrong length
                log_probability=[0.0, 0.0, 0.0, 0.0],
                log_concavity_checks=[True, True],
                is_log_concave=True,
            )

    def test_log_concavity_checks_length_mismatch_raises(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="log_concavity_checks length"):
            MatroidLogConcavityResult(
                n_assets=3,
                rank_weight=0.8,
                corank_weight=1.2,
                subset_sizes=[0, 1, 2, 3],
                probability_mass=[0.1, 0.4, 0.4, 0.1],
                log_probability=[-2.3, -0.9, -0.9, -2.3],
                log_concavity_checks=[True],  # wrong: should be length 2
                is_log_concave=True,
            )
