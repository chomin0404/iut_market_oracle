"""Tests for src/valuation/dcf.py.

Analytical derivations used in deterministic checks
----------------------------------------------------
project_fcfs
    FCF_t = initial_fcf × (1 + g)^t   (t = 1 … n)

discount_cash_flows
    PV_t = CF_t / (1 + r)^t

gordon_terminal_value
    TV = FCF_n × (1 + g_t) / (r − g_t)

dcf_valuation, 1-year closed form
    With n = 1:
        EV = FCF_1/(1+r)  +  TV/(1+r)
           = FCF_1/(1+r) × [1 + (1+g_t)/(r−g_t)]
           = FCF_1/(1+r) × (1+r)/(r−g_t)
           = FCF_1 / (r − g_t)

reverse_dcf_implied_growth
    Bisection over growth_rate; round-trip: dcf_valuation(implied_g) ≈ target_ev
"""

from __future__ import annotations

import math

import pytest

from valuation.dcf import (
    DCFInputs,
    DCFResult,
    dcf_valuation,
    discount_cash_flows,
    gordon_terminal_value,
    project_fcfs,
    reverse_dcf_implied_growth,
)

# ---------------------------------------------------------------------------
# DCFInputs validation
# ---------------------------------------------------------------------------


class TestDCFInputsValidation:
    def test_initial_fcf_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="initial_fcf"):
            DCFInputs(initial_fcf=0.0, growth_rate=0.05, discount_rate=0.10)

    def test_initial_fcf_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="initial_fcf"):
            DCFInputs(initial_fcf=-1.0, growth_rate=0.05, discount_rate=0.10)

    def test_discount_rate_at_minus_one_raises(self) -> None:
        with pytest.raises(ValueError, match="discount_rate"):
            DCFInputs(initial_fcf=100.0, growth_rate=0.0, discount_rate=-1.0)

    def test_discount_rate_below_minus_one_raises(self) -> None:
        with pytest.raises(ValueError, match="discount_rate"):
            DCFInputs(initial_fcf=100.0, growth_rate=0.0, discount_rate=-2.0)

    def test_forecast_years_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="forecast_years"):
            DCFInputs(initial_fcf=100.0, growth_rate=0.05, discount_rate=0.10, forecast_years=0)

    def test_forecast_years_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="forecast_years"):
            DCFInputs(initial_fcf=100.0, growth_rate=0.05, discount_rate=0.10, forecast_years=-1)

    def test_terminal_growth_rate_equal_to_discount_rate_raises(self) -> None:
        with pytest.raises(ValueError, match="terminal_growth_rate"):
            DCFInputs(
                initial_fcf=100.0,
                growth_rate=0.05,
                discount_rate=0.10,
                terminal_growth_rate=0.10,
            )

    def test_terminal_growth_rate_exceeds_discount_rate_raises(self) -> None:
        with pytest.raises(ValueError, match="terminal_growth_rate"):
            DCFInputs(
                initial_fcf=100.0,
                growth_rate=0.05,
                discount_rate=0.10,
                terminal_growth_rate=0.12,
            )

    def test_valid_inputs_do_not_raise(self) -> None:
        inputs = DCFInputs(initial_fcf=100.0, growth_rate=0.05, discount_rate=0.10)
        assert inputs.initial_fcf == 100.0

    def test_negative_growth_rate_is_valid(self) -> None:
        # Negative growth is economically valid (shrinking FCF)
        inputs = DCFInputs(initial_fcf=100.0, growth_rate=-0.05, discount_rate=0.10)
        assert inputs.growth_rate == -0.05


# ---------------------------------------------------------------------------
# project_fcfs
# ---------------------------------------------------------------------------


class TestProjectFCFs:
    def test_output_length_equals_years(self) -> None:
        assert len(project_fcfs(100.0, 0.10, 5)) == 5

    def test_geometric_growth(self) -> None:
        # FCF_t = initial_fcf × (1 + g)^t
        fcfs = project_fcfs(100.0, 0.10, 3)
        assert fcfs[0] == pytest.approx(110.0, rel=1e-12)
        assert fcfs[1] == pytest.approx(121.0, rel=1e-12)
        assert fcfs[2] == pytest.approx(133.1, rel=1e-12)

    def test_zero_growth_returns_constant(self) -> None:
        fcfs = project_fcfs(100.0, 0.0, 4)
        assert all(f == pytest.approx(100.0, rel=1e-12) for f in fcfs)

    def test_negative_growth_shrinks_fcf(self) -> None:
        fcfs = project_fcfs(100.0, -0.10, 3)
        assert fcfs[0] < 100.0
        assert fcfs[1] < fcfs[0]
        assert fcfs[2] < fcfs[1]

    def test_each_year_strictly_increasing_for_positive_growth(self) -> None:
        fcfs = project_fcfs(100.0, 0.05, 10)
        assert all(fcfs[i] < fcfs[i + 1] for i in range(len(fcfs) - 1))

    def test_single_year(self) -> None:
        assert project_fcfs(200.0, 0.05, 1) == pytest.approx([210.0], rel=1e-12)

    def test_initial_fcf_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="initial_fcf"):
            project_fcfs(0.0, 0.05, 5)

    def test_initial_fcf_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="initial_fcf"):
            project_fcfs(-100.0, 0.05, 5)

    def test_years_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="years"):
            project_fcfs(100.0, 0.05, 0)

    def test_years_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="years"):
            project_fcfs(100.0, 0.05, -1)


# ---------------------------------------------------------------------------
# discount_cash_flows
# ---------------------------------------------------------------------------


class TestDiscountCashFlows:
    def test_output_length_matches_input(self) -> None:
        cfs = [100.0, 110.0, 121.0]
        assert len(discount_cash_flows(cfs, 0.10)) == 3

    def test_empty_input_returns_empty(self) -> None:
        assert discount_cash_flows([], 0.10) == []

    def test_pv_formula(self) -> None:
        # PV_t = CF_t / (1 + r)^t
        cfs = [110.0, 121.0, 133.1]
        r = 0.10
        pvs = discount_cash_flows(cfs, r)
        assert pvs[0] == pytest.approx(110.0 / 1.10, rel=1e-12)
        assert pvs[1] == pytest.approx(121.0 / 1.10**2, rel=1e-12)
        assert pvs[2] == pytest.approx(133.1 / 1.10**3, rel=1e-12)

    def test_pv_less_than_fv_for_positive_rate(self) -> None:
        pvs = discount_cash_flows([100.0, 100.0, 100.0], 0.10)
        assert all(pv < 100.0 for pv in pvs)

    def test_zero_discount_rate_returns_cash_flows_unchanged(self) -> None:
        cfs = [100.0, 200.0, 300.0]
        pvs = discount_cash_flows(cfs, 0.0)
        assert pvs == pytest.approx(cfs, rel=1e-12)

    def test_later_cash_flows_discounted_more(self) -> None:
        # Same amount at different horizons; later PV must be smaller
        pvs = discount_cash_flows([100.0, 100.0, 100.0], 0.10)
        assert pvs[0] > pvs[1] > pvs[2]

    def test_discount_rate_at_minus_one_raises(self) -> None:
        with pytest.raises(ValueError, match="discount_rate"):
            discount_cash_flows([100.0], -1.0)

    def test_discount_rate_below_minus_one_raises(self) -> None:
        with pytest.raises(ValueError, match="discount_rate"):
            discount_cash_flows([100.0], -2.0)


# ---------------------------------------------------------------------------
# gordon_terminal_value
# ---------------------------------------------------------------------------


class TestGordonTerminalValue:
    def test_formula(self) -> None:
        # TV = fcf × (1 + g_t) / (r − g_t)
        tv = gordon_terminal_value(
            final_year_fcf=100.0, discount_rate=0.10, terminal_growth_rate=0.03
        )
        assert tv == pytest.approx(100.0 * 1.03 / 0.07, rel=1e-12)

    def test_higher_terminal_growth_raises_tv(self) -> None:
        tv_low = gordon_terminal_value(100.0, 0.10, 0.02)
        tv_high = gordon_terminal_value(100.0, 0.10, 0.05)
        assert tv_high > tv_low

    def test_higher_discount_rate_lowers_tv(self) -> None:
        tv_low = gordon_terminal_value(100.0, 0.08, 0.03)
        tv_high = gordon_terminal_value(100.0, 0.15, 0.03)
        assert tv_low > tv_high

    def test_final_year_fcf_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="final_year_fcf"):
            gordon_terminal_value(0.0, 0.10, 0.03)

    def test_final_year_fcf_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="final_year_fcf"):
            gordon_terminal_value(-50.0, 0.10, 0.03)

    def test_terminal_growth_equal_to_discount_raises(self) -> None:
        with pytest.raises(ValueError, match="terminal_growth_rate"):
            gordon_terminal_value(100.0, 0.10, 0.10)

    def test_terminal_growth_exceeds_discount_raises(self) -> None:
        with pytest.raises(ValueError, match="terminal_growth_rate"):
            gordon_terminal_value(100.0, 0.10, 0.15)


# ---------------------------------------------------------------------------
# dcf_valuation
# ---------------------------------------------------------------------------


class TestDCFValuation:
    def _inputs(self, **overrides) -> DCFInputs:
        defaults = dict(
            initial_fcf=100.0,
            growth_rate=0.10,
            discount_rate=0.12,
            forecast_years=5,
            terminal_growth_rate=0.03,
        )
        defaults.update(overrides)
        return DCFInputs(**defaults)

    def test_returns_dcf_result(self) -> None:
        result = dcf_valuation(self._inputs())
        assert isinstance(result, DCFResult)

    def test_projected_fcfs_length(self) -> None:
        result = dcf_valuation(self._inputs(forecast_years=5))
        assert len(result.projected_fcfs) == 5

    def test_discounted_fcfs_length(self) -> None:
        result = dcf_valuation(self._inputs(forecast_years=5))
        assert len(result.discounted_fcfs) == 5

    def test_enterprise_value_equals_sum_of_parts(self) -> None:
        # EV = Σ discounted_fcfs + discounted_terminal_value
        result = dcf_valuation(self._inputs())
        expected = sum(result.discounted_fcfs) + result.discounted_terminal_value
        assert result.enterprise_value == pytest.approx(expected, rel=1e-12)

    def test_one_year_closed_form(self) -> None:
        # With n=1: EV = FCF_1 / (r − g_t)
        # FCF_1 = 100 × 1.10 = 110
        # EV = 110 / (0.12 − 0.03) = 110 / 0.09
        result = dcf_valuation(self._inputs(forecast_years=1))
        expected = 100.0 * 1.10 / (0.12 - 0.03)
        assert result.enterprise_value == pytest.approx(expected, rel=1e-9)

    def test_higher_discount_rate_lowers_ev(self) -> None:
        ev_low = dcf_valuation(self._inputs(discount_rate=0.08)).enterprise_value
        ev_high = dcf_valuation(self._inputs(discount_rate=0.15)).enterprise_value
        assert ev_low > ev_high

    def test_higher_growth_rate_raises_ev(self) -> None:
        ev_slow = dcf_valuation(self._inputs(growth_rate=0.02)).enterprise_value
        ev_fast = dcf_valuation(self._inputs(growth_rate=0.20)).enterprise_value
        assert ev_fast > ev_slow

    def test_longer_forecast_changes_ev(self) -> None:
        # Not a monotone relationship in general, but the two values should differ
        ev_5 = dcf_valuation(self._inputs(forecast_years=5)).enterprise_value
        ev_10 = dcf_valuation(self._inputs(forecast_years=10)).enterprise_value
        assert ev_5 != pytest.approx(ev_10, rel=1e-4)

    def test_deterministic(self) -> None:
        inputs = self._inputs()
        assert dcf_valuation(inputs).enterprise_value == dcf_valuation(inputs).enterprise_value

    def test_enterprise_value_positive(self) -> None:
        assert dcf_valuation(self._inputs()).enterprise_value > 0.0

    def test_all_fields_finite(self) -> None:
        result = dcf_valuation(self._inputs())
        assert math.isfinite(result.terminal_value)
        assert math.isfinite(result.discounted_terminal_value)
        assert math.isfinite(result.enterprise_value)
        assert all(math.isfinite(v) for v in result.projected_fcfs)
        assert all(math.isfinite(v) for v in result.discounted_fcfs)

    def test_discounted_terminal_value_less_than_terminal_value(self) -> None:
        result = dcf_valuation(self._inputs())
        assert result.discounted_terminal_value < result.terminal_value


# ---------------------------------------------------------------------------
# reverse_dcf_implied_growth
# ---------------------------------------------------------------------------


class TestReverseDCFImpliedGrowth:
    _BASE = dict(
        initial_fcf=100.0,
        discount_rate=0.12,
        forecast_years=5,
        terminal_growth_rate=0.03,
    )

    def _roundtrip(self, growth_rate: float) -> float:
        """Compute EV at growth_rate, then recover implied growth."""
        ev = dcf_valuation(
            DCFInputs(growth_rate=growth_rate, **self._BASE)
        ).enterprise_value
        return reverse_dcf_implied_growth(target_enterprise_value=ev, **self._BASE)

    def test_round_trip_low_growth(self) -> None:
        implied = self._roundtrip(0.02)
        assert implied == pytest.approx(0.02, abs=1e-6)

    def test_round_trip_moderate_growth(self) -> None:
        implied = self._roundtrip(0.10)
        assert implied == pytest.approx(0.10, abs=1e-6)

    def test_round_trip_high_growth(self) -> None:
        implied = self._roundtrip(0.30)
        assert implied == pytest.approx(0.30, abs=1e-6)

    def test_round_trip_negative_growth(self) -> None:
        implied = self._roundtrip(-0.10)
        assert implied == pytest.approx(-0.10, abs=1e-6)

    def test_result_in_search_bounds(self) -> None:
        implied = self._roundtrip(0.10)
        assert -0.50 <= implied <= 0.80

    def test_target_ev_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="target_enterprise_value"):
            reverse_dcf_implied_growth(target_enterprise_value=0.0, **self._BASE)

    def test_target_ev_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="target_enterprise_value"):
            reverse_dcf_implied_growth(target_enterprise_value=-1.0, **self._BASE)

    def test_initial_fcf_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="initial_fcf"):
            reverse_dcf_implied_growth(
                target_enterprise_value=1000.0,
                initial_fcf=0.0,
                discount_rate=0.12,
            )

    def test_terminal_growth_exceeds_discount_raises(self) -> None:
        with pytest.raises(ValueError, match="terminal_growth_rate"):
            reverse_dcf_implied_growth(
                target_enterprise_value=1000.0,
                initial_fcf=100.0,
                discount_rate=0.05,
                terminal_growth_rate=0.08,
            )

    def test_unbracketable_target_raises(self) -> None:
        # EV is monotone in growth_rate; a target below EV(low) is unbracketable
        ev_at_low = dcf_valuation(
            DCFInputs(growth_rate=-0.50, **self._BASE)
        ).enterprise_value
        with pytest.raises(ValueError, match="bracket"):
            reverse_dcf_implied_growth(
                target_enterprise_value=ev_at_low * 0.1,
                **self._BASE,
            )

    def test_higher_target_ev_implies_higher_growth(self) -> None:
        g_low = self._roundtrip(0.05)
        g_high = self._roundtrip(0.25)
        assert g_high > g_low
