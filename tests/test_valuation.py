"""Tests for src/valuation/scenario.py.

All expected EV values are derived analytically or via known closed-form
properties of the DCF model; no "magic numbers" are hardcoded without derivation.

DCF identity checks
-------------------
Zero margin  → FCF = 0 → EV = 0
capex_rate=1 → Capex = NOPAT → FCF = 0 → EV = 0
Higher WACC  → smaller PV factors → lower EV
Higher growth → larger FCFs → higher EV
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from schemas import AssumptionSet
from valuation.scenario import (
    _dcf,
    load_assumption_yaml,
    run_all_scenarios,
    run_scenario,
)

SCENARIO_DIR = Path(__file__).parent.parent / "configs" / "scenarios"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def base_params(**overrides) -> dict:
    p = {
        "initial_revenue": 10_000.0,
        "revenue_growth": 0.05,
        "ebit_margin": 0.12,
        "tax_rate": 0.30,
        "capex_rate": 0.30,
        "discount_rate": 0.10,
        "terminal_growth_rate": 0.02,
        "forecast_years": 5,
    }
    p.update(overrides)
    return p


def make_assumption(**overrides) -> AssumptionSet:
    return AssumptionSet(name="test", version="1.0", params=base_params(**overrides))


# ---------------------------------------------------------------------------
# _dcf: core engine
# ---------------------------------------------------------------------------


class TestDcfEngine:
    def test_zero_ebit_margin_gives_zero_ev(self):
        ev = _dcf(**{**base_params(), "ebit_margin": 0.0})
        assert ev == pytest.approx(0.0, abs=1e-6)

    def test_break_even_capex_rate_gives_zero_ev(self):
        # FCF = NOPAT − Capex = ebit*(1−tax) − ebit*capex = ebit*(1−tax−capex)
        # FCF = 0  ⟺  capex_rate = 1 − tax_rate
        # With tax_rate=0.30: capex_rate = 0.70 → FCF = 0 → EV = 0
        ev = _dcf(**{**base_params(), "capex_rate": 0.70})
        assert ev == pytest.approx(0.0, abs=1e-6)

    def test_higher_wacc_lowers_ev(self):
        ev_low = _dcf(**base_params(discount_rate=0.08))
        ev_high = _dcf(**base_params(discount_rate=0.15))
        assert ev_high < ev_low

    def test_higher_growth_raises_ev(self):
        ev_slow = _dcf(**base_params(revenue_growth=0.01))
        ev_fast = _dcf(**base_params(revenue_growth=0.10))
        assert ev_fast > ev_slow

    def test_higher_margin_raises_ev(self):
        ev_thin = _dcf(**base_params(ebit_margin=0.05))
        ev_fat = _dcf(**base_params(ebit_margin=0.20))
        assert ev_fat > ev_thin

    def test_discount_rate_must_exceed_terminal_growth(self):
        with pytest.raises(ValueError, match="discount_rate.*must be >"):
            _dcf(**base_params(discount_rate=0.02, terminal_growth_rate=0.03))

    def test_equal_rates_raises(self):
        with pytest.raises(ValueError):
            _dcf(**base_params(discount_rate=0.05, terminal_growth_rate=0.05))

    def test_forecast_years_zero_raises(self):
        with pytest.raises(ValueError, match="forecast_years"):
            _dcf(**base_params(forecast_years=0))

    def test_single_year_deterministic(self):
        """Manually compute 1-year DCF and verify."""
        p = base_params(forecast_years=1)
        # Year 1: revenue = 10000 * 1.05 = 10500
        # ebit = 10500 * 0.12 = 1260
        # nopat = 1260 * 0.70 = 882
        # capex = 1260 * 0.30 = 378
        # fcf = 882 - 378 = 504
        # pv_fcf = 504 / 1.10 = 458.1818...
        # terminal_fcf = 504 * 1.02 = 514.08
        # tv = 514.08 / (0.10 - 0.02) = 6426.0
        # pv_tv = 6426.0 / 1.10 = 5841.8181...
        # ev = 458.1818 + 5841.8181 = 6300.0
        ev = _dcf(**p)
        assert ev == pytest.approx(6300.0, rel=1e-9)

    def test_ev_positive_for_standard_params(self):
        assert _dcf(**base_params()) > 0.0

    def test_deterministic(self):
        p = base_params()
        assert _dcf(**p) == _dcf(**p)


# ---------------------------------------------------------------------------
# run_scenario: output schema
# ---------------------------------------------------------------------------


class TestRunScenario:
    def test_returns_scenario_result(self):
        result = run_scenario(make_assumption())
        assert result.scenario_name == "test"
        assert result.assumption_version == "1.0"

    def test_value_positive(self):
        assert run_scenario(make_assumption()).value > 0.0

    def test_unit_is_jpy_millions(self):
        assert run_scenario(make_assumption()).unit == "JPY millions"

    def test_output_path_recorded(self):
        result = run_scenario(make_assumption(), output_path="/tmp/test.yaml")
        assert result.output_path == "/tmp/test.yaml"

    def test_deterministic(self):
        a = make_assumption()
        assert run_scenario(a).value == run_scenario(a).value


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------


class TestSensitivity:
    def test_sensitivity_keys_present(self):
        result = run_scenario(make_assumption())
        expected_keys = {
            "initial_revenue",
            "revenue_growth",
            "ebit_margin",
            "tax_rate",
            "capex_rate",
            "discount_rate",
            "terminal_growth_rate",
        }
        assert expected_keys <= set(result.sensitivity.keys())

    def test_forecast_years_excluded(self):
        # forecast_years is int and non-differentiable
        result = run_scenario(make_assumption())
        assert "forecast_years" not in result.sensitivity

    def test_revenue_growth_sensitivity_positive(self):
        # More growth → higher EV → ∂EV/∂g > 0
        s = run_scenario(make_assumption()).sensitivity
        assert s["revenue_growth"] > 0.0

    def test_discount_rate_sensitivity_negative(self):
        # Higher WACC → lower EV → ∂EV/∂WACC < 0
        s = run_scenario(make_assumption()).sensitivity
        assert s["discount_rate"] < 0.0

    def test_ebit_margin_sensitivity_positive(self):
        s = run_scenario(make_assumption()).sensitivity
        assert s["ebit_margin"] > 0.0

    def test_capex_rate_sensitivity_negative(self):
        # Higher capex → lower FCF → ∂EV/∂capex < 0
        s = run_scenario(make_assumption()).sensitivity
        assert s["capex_rate"] < 0.0

    def test_tax_rate_sensitivity_negative(self):
        s = run_scenario(make_assumption()).sensitivity
        assert s["tax_rate"] < 0.0

    def test_initial_revenue_sensitivity_positive(self):
        s = run_scenario(make_assumption()).sensitivity
        assert s["initial_revenue"] > 0.0

    def test_sensitivity_finite(self):
        for key, val in run_scenario(make_assumption()).sensitivity.items():
            assert math.isfinite(val), f"sensitivity[{key!r}] is not finite: {val}"

    def test_sensitivity_approximation_quality(self):
        """∂EV/∂g · Δg should approximate ΔEV to within O(Δg²)."""
        base = make_assumption()
        delta_g = 0.001  # 0.1% absolute change in growth rate
        ev_base = run_scenario(base).value
        ev_up = run_scenario(make_assumption(revenue_growth=0.05 + delta_g)).value

        dEV_dg = run_scenario(base).sensitivity["revenue_growth"]
        approx_delta = dEV_dg * delta_g
        actual_delta = ev_up - ev_base

        # First-order approximation should be accurate to < 1% of actual change
        assert abs(approx_delta - actual_delta) < 0.01 * abs(actual_delta)


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------


class TestLoadAssumptionYaml:
    @pytest.mark.skipif(not SCENARIO_DIR.exists(), reason="configs/scenarios/ not found")
    def test_load_base(self):
        a = load_assumption_yaml(SCENARIO_DIR / "base.yaml")
        assert a.name == "base"
        assert a.params["initial_revenue"] > 0

    @pytest.mark.skipif(not SCENARIO_DIR.exists(), reason="configs/scenarios/ not found")
    def test_load_bull(self):
        a = load_assumption_yaml(SCENARIO_DIR / "bull.yaml")
        assert a.params["revenue_growth"] > 0.05  # bull > base growth

    @pytest.mark.skipif(not SCENARIO_DIR.exists(), reason="configs/scenarios/ not found")
    def test_load_bear(self):
        a = load_assumption_yaml(SCENARIO_DIR / "bear.yaml")
        assert a.params["revenue_growth"] < 0.05  # bear < base growth

    @pytest.mark.skipif(not SCENARIO_DIR.exists(), reason="configs/scenarios/ not found")
    def test_bull_ev_greater_than_base(self):
        bull = load_assumption_yaml(SCENARIO_DIR / "bull.yaml")
        base = load_assumption_yaml(SCENARIO_DIR / "base.yaml")
        assert run_scenario(bull).value > run_scenario(base).value

    @pytest.mark.skipif(not SCENARIO_DIR.exists(), reason="configs/scenarios/ not found")
    def test_bear_ev_less_than_base(self):
        bear = load_assumption_yaml(SCENARIO_DIR / "bear.yaml")
        base = load_assumption_yaml(SCENARIO_DIR / "base.yaml")
        assert run_scenario(bear).value < run_scenario(base).value


# ---------------------------------------------------------------------------
# run_all_scenarios
# ---------------------------------------------------------------------------


class TestRunAllScenarios:
    @pytest.mark.skipif(not SCENARIO_DIR.exists(), reason="configs/scenarios/ not found")
    def test_returns_three_results(self):
        results = run_all_scenarios(SCENARIO_DIR)
        assert len(results) == 3

    @pytest.mark.skipif(not SCENARIO_DIR.exists(), reason="configs/scenarios/ not found")
    def test_sorted_by_name(self):
        results = run_all_scenarios(SCENARIO_DIR)
        names = [r.scenario_name for r in results]
        assert names == sorted(names)

    @pytest.mark.skipif(not SCENARIO_DIR.exists(), reason="configs/scenarios/ not found")
    def test_all_values_positive(self):
        for r in run_all_scenarios(SCENARIO_DIR):
            assert r.value > 0.0, f"scenario {r.scenario_name} has non-positive EV"
