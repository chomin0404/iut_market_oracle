"""Tests for src/twin/regime_simulator.py (T1100)."""

from __future__ import annotations

import numpy as np
import pytest

from schemas import MarketEvolutionResult, RegimeSwitchResult
from twin.regime_simulator import simulate_market_evolution, simulate_regime_switching

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_P_NORMAL = 0.95
_P_VOLATILE = 0.90
_N_STEPS = 200
_INIT_PRICE = 100.0


def _rng(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# simulate_regime_switching — determinism
# ---------------------------------------------------------------------------


class TestRegimeSwitchingDeterminism:
    def test_same_seed_identical_output(self):
        a = simulate_regime_switching(_N_STEPS, _INIT_PRICE, _P_NORMAL, _P_VOLATILE, _rng(0))
        b = simulate_regime_switching(_N_STEPS, _INIT_PRICE, _P_NORMAL, _P_VOLATILE, _rng(0))
        assert a.prices == b.prices
        assert a.regimes == b.regimes

    def test_different_seeds_differ(self):
        a = simulate_regime_switching(_N_STEPS, _INIT_PRICE, _P_NORMAL, _P_VOLATILE, _rng(1))
        b = simulate_regime_switching(_N_STEPS, _INIT_PRICE, _P_NORMAL, _P_VOLATILE, _rng(2))
        assert a.prices != b.prices


# ---------------------------------------------------------------------------
# simulate_regime_switching — structural invariants
# ---------------------------------------------------------------------------


class TestRegimeSwitchingInvariants:
    def test_prices_length_equals_n_steps(self):
        result = simulate_regime_switching(
            _N_STEPS, _INIT_PRICE, _P_NORMAL, _P_VOLATILE, _rng()
        )
        assert len(result.prices) == _N_STEPS

    def test_regimes_length_equals_n_steps(self):
        result = simulate_regime_switching(
            _N_STEPS, _INIT_PRICE, _P_NORMAL, _P_VOLATILE, _rng()
        )
        assert len(result.regimes) == _N_STEPS

    def test_initial_price_preserved(self):
        result = simulate_regime_switching(
            _N_STEPS, _INIT_PRICE, _P_NORMAL, _P_VOLATILE, _rng()
        )
        assert result.prices[0] == pytest.approx(_INIT_PRICE)

    def test_initial_regime_is_zero(self):
        result = simulate_regime_switching(
            _N_STEPS, _INIT_PRICE, _P_NORMAL, _P_VOLATILE, _rng()
        )
        assert result.regimes[0] == 0

    def test_regimes_are_binary(self):
        result = simulate_regime_switching(
            _N_STEPS, _INIT_PRICE, _P_NORMAL, _P_VOLATILE, _rng()
        )
        assert all(r in (0, 1) for r in result.regimes)

    def test_prices_all_positive(self):
        result = simulate_regime_switching(
            _N_STEPS, _INIT_PRICE, _P_NORMAL, _P_VOLATILE, _rng()
        )
        assert all(p > 0 for p in result.prices)

    def test_n_steps_stored(self):
        result = simulate_regime_switching(
            _N_STEPS, _INIT_PRICE, _P_NORMAL, _P_VOLATILE, _rng()
        )
        assert result.n_steps == _N_STEPS

    def test_transition_probs_stored(self):
        result = simulate_regime_switching(
            _N_STEPS, _INIT_PRICE, _P_NORMAL, _P_VOLATILE, _rng()
        )
        assert result.p_stay_normal == pytest.approx(_P_NORMAL)
        assert result.p_stay_volatile == pytest.approx(_P_VOLATILE)


# ---------------------------------------------------------------------------
# simulate_regime_switching — statistical properties
# ---------------------------------------------------------------------------


class TestRegimeSwitchingStats:
    def test_volatile_regime_appears_with_high_p_normal(self):
        """With p_stay_normal < 1, regime 1 must appear at least once in 1000 steps."""
        result = simulate_regime_switching(1000, _INIT_PRICE, 0.80, _P_VOLATILE, _rng(7))
        assert 1 in result.regimes

    def test_high_p_volatile_stays_in_volatile(self):
        """Markov chain started in volatile regime (forced) should persist with p=0.99."""
        # Force regime 1 at step 0 by using p_stay_normal very close to 0
        result = simulate_regime_switching(500, _INIT_PRICE, 0.01, 0.99, _rng(5))
        # Regime 1 should dominate
        frac_volatile = sum(r == 1 for r in result.regimes) / len(result.regimes)
        assert frac_volatile > 0.5


# ---------------------------------------------------------------------------
# simulate_regime_switching — invalid inputs
# ---------------------------------------------------------------------------


class TestRegimeSwitchingInvalidInputs:
    def test_n_steps_zero_raises(self):
        with pytest.raises(ValueError, match="n_steps must be >= 1"):
            simulate_regime_switching(0, _INIT_PRICE, _P_NORMAL, _P_VOLATILE, _rng())

    def test_initial_price_zero_raises(self):
        with pytest.raises(ValueError, match="initial_price must be > 0"):
            simulate_regime_switching(_N_STEPS, 0.0, _P_NORMAL, _P_VOLATILE, _rng())

    def test_p_stay_normal_zero_raises(self):
        with pytest.raises(ValueError, match="p_stay_normal must be in"):
            simulate_regime_switching(_N_STEPS, _INIT_PRICE, 0.0, _P_VOLATILE, _rng())

    def test_p_stay_normal_one_raises(self):
        with pytest.raises(ValueError, match="p_stay_normal must be in"):
            simulate_regime_switching(_N_STEPS, _INIT_PRICE, 1.0, _P_VOLATILE, _rng())

    def test_p_stay_volatile_zero_raises(self):
        with pytest.raises(ValueError, match="p_stay_volatile must be in"):
            simulate_regime_switching(_N_STEPS, _INIT_PRICE, _P_NORMAL, 0.0, _rng())


# ---------------------------------------------------------------------------
# simulate_market_evolution — determinism
# ---------------------------------------------------------------------------


class TestMarketEvolutionDeterminism:
    def test_same_seed_identical_output(self):
        a = simulate_market_evolution(200, 2.0, 1.0, _rng(0))
        b = simulate_market_evolution(200, 2.0, 1.0, _rng(0))
        assert a.new_customers == b.new_customers
        assert a.market_capture == b.market_capture

    def test_different_seeds_differ(self):
        a = simulate_market_evolution(200, 2.0, 1.0, _rng(1))
        b = simulate_market_evolution(200, 2.0, 1.0, _rng(2))
        assert a.new_customers != b.new_customers


# ---------------------------------------------------------------------------
# simulate_market_evolution — structural invariants
# ---------------------------------------------------------------------------


class TestMarketEvolutionInvariants:
    def test_all_series_length_equals_n_steps(self):
        n = 150
        result = simulate_market_evolution(n, 2.0, 1.0, _rng())
        assert result.n_steps == n
        assert len(result.new_customers) == n
        assert len(result.cumulative_base) == n
        assert len(result.sigmoid_factor) == n
        assert len(result.market_capture) == n

    def test_new_customers_non_negative(self):
        result = simulate_market_evolution(200, 2.0, 1.0, _rng())
        assert all(k >= 0 for k in result.new_customers)

    def test_cumulative_base_monotone_non_decreasing(self):
        result = simulate_market_evolution(200, 2.0, 1.0, _rng())
        cb = result.cumulative_base
        assert all(cb[t] <= cb[t + 1] for t in range(len(cb) - 1))

    def test_sigmoid_factor_in_unit_interval(self):
        result = simulate_market_evolution(200, 2.0, 1.0, _rng())
        assert all(0.0 < s < 1.0 for s in result.sigmoid_factor)

    def test_sigmoid_factor_monotone_increasing(self):
        result = simulate_market_evolution(200, 2.0, 1.0, _rng())
        sf = result.sigmoid_factor
        assert all(sf[t] < sf[t + 1] for t in range(len(sf) - 1))

    def test_gamma_params_stored(self):
        result = simulate_market_evolution(100, 3.0, 2.0, _rng())
        assert result.gamma_alpha == pytest.approx(3.0)
        assert result.gamma_beta == pytest.approx(2.0)

    def test_market_capture_non_negative(self):
        result = simulate_market_evolution(200, 2.0, 1.0, _rng())
        assert all(mc >= 0.0 for mc in result.market_capture)


# ---------------------------------------------------------------------------
# simulate_market_evolution — invalid inputs
# ---------------------------------------------------------------------------


class TestMarketEvolutionInvalidInputs:
    def test_n_steps_zero_raises(self):
        with pytest.raises(ValueError, match="n_steps must be >= 1"):
            simulate_market_evolution(0, 2.0, 1.0, _rng())

    def test_gamma_alpha_zero_raises(self):
        with pytest.raises(ValueError, match="gamma_alpha must be > 0"):
            simulate_market_evolution(100, 0.0, 1.0, _rng())

    def test_gamma_beta_negative_raises(self):
        with pytest.raises(ValueError, match="gamma_beta must be > 0"):
            simulate_market_evolution(100, 2.0, -1.0, _rng())


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


class TestRegimeSwitchResultSchema:
    def test_prices_length_mismatch_raises(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="prices length"):
            RegimeSwitchResult(
                n_steps=5,
                prices=[1.0, 2.0],  # wrong length
                regimes=[0, 0, 0, 0, 0],
                p_stay_normal=0.95,
                p_stay_volatile=0.90,
            )

    def test_regimes_length_mismatch_raises(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="regimes length"):
            RegimeSwitchResult(
                n_steps=3,
                prices=[1.0, 1.01, 1.02],
                regimes=[0, 1],  # wrong length
                p_stay_normal=0.95,
                p_stay_volatile=0.90,
            )


class TestMarketEvolutionResultSchema:
    def test_new_customers_length_mismatch_raises(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="new_customers length"):
            MarketEvolutionResult(
                n_steps=3,
                new_customers=[1, 2],  # wrong length
                cumulative_base=[1.0, 3.0, 5.0],
                sigmoid_factor=[0.1, 0.5, 0.9],
                market_capture=[0.1, 1.5, 4.5],
                gamma_alpha=2.0,
                gamma_beta=1.0,
            )
