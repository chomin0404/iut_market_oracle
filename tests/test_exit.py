"""Tests for src/exit/option_pricer.py and src/exit/timing_map.py (T900)."""

from __future__ import annotations

import numpy as np
import pytest

from exit.option_pricer import price_all_options, price_option
from exit.timing_map import (
    build_timing_map,
    compare_exit_options,
    price_with_timing_map,
    timing_sensitivity,
)
from schemas import ExitOption, ExitType, TimingDistribution

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SCENARIOS_3 = {"bear": 80.0, "base": 150.0, "bull": 280.0}
SCENARIOS_2 = {"low": 50.0, "high": 200.0}


def _opt(
    name: str = "ipo",
    exit_type: ExitType = ExitType.IPO,
    earliest: float = 1.0,
    expected: float = 3.0,
    latest: float = 5.0,
    values: dict[str, float] | None = None,
    floor: float = 0.0,
    rate: float = 0.10,
) -> ExitOption:
    return ExitOption(
        name=name,
        exit_type=exit_type,
        timing_earliest=earliest,
        timing_expected=expected,
        timing_latest=latest,
        value_by_scenario=values or SCENARIOS_3,
        floor_value=floor,
        discount_rate=rate,
    )


# ---------------------------------------------------------------------------
# ExitOption schema validation
# ---------------------------------------------------------------------------


class TestExitOptionSchema:
    def test_valid_option_created(self):
        opt = _opt()
        assert opt.name == "ipo"
        assert opt.exit_type == ExitType.IPO

    def test_timing_order_earliest_gt_expected_raises(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="timing must satisfy"):
            ExitOption(
                name="x",
                exit_type=ExitType.MA,
                timing_earliest=4.0,
                timing_expected=2.0,  # < earliest
                timing_latest=6.0,
                value_by_scenario={"base": 100.0},
                discount_rate=0.10,
            )

    def test_timing_order_expected_gt_latest_raises(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="timing must satisfy"):
            ExitOption(
                name="x",
                exit_type=ExitType.MA,
                timing_earliest=1.0,
                timing_expected=7.0,  # > latest
                timing_latest=5.0,
                value_by_scenario={"base": 100.0},
                discount_rate=0.10,
            )

    def test_empty_scenarios_raises(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="at least one scenario"):
            ExitOption(
                name="x",
                exit_type=ExitType.IPO,
                timing_earliest=1.0,
                timing_expected=3.0,
                timing_latest=5.0,
                value_by_scenario={},
                discount_rate=0.10,
            )

    def test_negative_floor_raises(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            _opt(floor=-1.0)

    def test_zero_discount_rate_raises(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            _opt(rate=0.0)

    def test_equal_timing_valid(self):
        """earliest == expected == latest is degenerate but valid."""
        opt = _opt(earliest=3.0, expected=3.0, latest=3.0)
        assert opt.timing_earliest == opt.timing_latest


# ---------------------------------------------------------------------------
# option_pricer — floor behaviour
# ---------------------------------------------------------------------------


class TestFloorBehaviour:
    def test_floor_zero_all_positive_payoffs(self):
        """With floor=0 and all V > 0, every payoff equals V."""
        opt = _opt(values={"base": 100.0}, floor=0.0)
        result = price_option(opt)
        assert result.scenario_payoffs["base"] == pytest.approx(100.0)

    def test_floor_clips_negative_ev_to_zero(self):
        """Scenario value below floor → payoff = 0."""
        opt = _opt(values={"bear": 50.0}, floor=100.0)
        result = price_option(opt)
        assert result.scenario_payoffs["bear"] == pytest.approx(0.0)
        assert result.expected_value == pytest.approx(0.0)

    def test_floor_partially_clips(self):
        """Some scenarios above floor, some below."""
        opt = _opt(values={"low": 80.0, "high": 200.0}, floor=100.0)
        result = price_option(opt)
        assert result.scenario_payoffs["low"] == pytest.approx(0.0)
        assert result.scenario_payoffs["high"] == pytest.approx(100.0)

    def test_floor_exactly_at_value_gives_zero(self):
        opt = _opt(values={"base": 100.0}, floor=100.0)
        result = price_option(opt)
        assert result.scenario_payoffs["base"] == pytest.approx(0.0)

    def test_higher_floor_reduces_expected_value(self):
        opt_low = _opt(values=SCENARIOS_3, floor=50.0)
        opt_high = _opt(values=SCENARIOS_3, floor=120.0)
        assert price_option(opt_high).expected_value < price_option(opt_low).expected_value


# ---------------------------------------------------------------------------
# option_pricer — discounting
# ---------------------------------------------------------------------------


class TestDiscounting:
    def test_zero_timing_no_discounting(self):
        """t_expected = 0: PV equals payoff."""
        opt = _opt(values={"base": 100.0}, floor=0.0, expected=0.0, earliest=0.0)
        result = price_option(opt)
        assert result.scenario_pvs["base"] == pytest.approx(100.0)

    def test_positive_timing_reduces_pv(self):
        opt_now = _opt(values={"base": 100.0}, floor=0.0, expected=0.0, earliest=0.0)
        opt_later = _opt(values={"base": 100.0}, floor=0.0, expected=3.0)
        pv_later = price_option(opt_later).scenario_pvs["base"]
        pv_now = price_option(opt_now).scenario_pvs["base"]
        assert pv_later < pv_now

    def test_pv_formula_exact(self):
        """PV = payoff / (1 + r)^t should match the formula exactly."""
        V, floor, r, t = 200.0, 50.0, 0.10, 3.0
        opt = _opt(values={"s": V}, floor=floor, rate=r, expected=t, earliest=1.0)
        result = price_option(opt)
        expected_pv = (V - floor) / (1.0 + r) ** t
        assert result.scenario_pvs["s"] == pytest.approx(expected_pv, rel=1e-9)

    def test_higher_discount_rate_reduces_ev(self):
        opt_low = _opt(values=SCENARIOS_3, rate=0.05)
        opt_high = _opt(values=SCENARIOS_3, rate=0.20)
        assert price_option(opt_high).expected_value < price_option(opt_low).expected_value


# ---------------------------------------------------------------------------
# option_pricer — scenario probabilities
# ---------------------------------------------------------------------------


class TestScenarioProbabilities:
    def test_uniform_probs_by_default(self):
        """Three equally-weighted scenarios: EV = mean of PVs."""
        opt = _opt(
            values={"a": 90.0, "b": 150.0, "c": 210.0}, floor=0.0, expected=0.0, earliest=0.0
        )
        result = price_option(opt)
        expected = (90.0 + 150.0 + 210.0) / 3.0
        assert result.expected_value == pytest.approx(expected, rel=1e-9)

    def test_custom_probs_applied(self):
        opt = _opt(values={"low": 50.0, "high": 200.0}, floor=0.0, expected=0.0, earliest=0.0)
        probs = {"low": 0.8, "high": 0.2}
        result = price_option(opt, scenario_probs=probs)
        expected = 0.8 * 50.0 + 0.2 * 200.0
        assert result.expected_value == pytest.approx(expected, rel=1e-9)

    def test_mismatched_prob_keys_raises(self):
        opt = _opt(values=SCENARIOS_3)
        with pytest.raises(ValueError, match="scenario_probs keys"):
            price_option(opt, scenario_probs={"wrong": 1.0})

    def test_probs_not_summing_to_one_raises(self):
        opt = _opt(values=SCENARIOS_2)
        with pytest.raises(ValueError, match="sum to ~1.0"):
            price_option(opt, scenario_probs={"low": 0.3, "high": 0.3})


# ---------------------------------------------------------------------------
# option_pricer — sensitivity
# ---------------------------------------------------------------------------


class TestOptionPricerSensitivity:
    def test_discount_rate_sensitivity_negative(self):
        """Higher discount rate → lower PV → ∂EV/∂r < 0."""
        opt = _opt(values=SCENARIOS_3, floor=0.0)
        result = price_option(opt)
        assert result.sensitivity["discount_rate"] < 0.0

    def test_timing_sensitivity_negative(self):
        """Later exit → lower PV → ∂EV/∂t < 0."""
        opt = _opt(values=SCENARIOS_3, floor=0.0, expected=3.0)
        result = price_option(opt)
        assert result.sensitivity["timing_expected"] < 0.0

    def test_floor_sensitivity_negative(self):
        """Higher floor → lower payoffs → ∂EV/∂floor < 0."""
        opt = _opt(values=SCENARIOS_3, floor=50.0)
        result = price_option(opt)
        assert result.sensitivity["floor_value"] < 0.0

    def test_sensitivity_absent_when_floor_zero(self):
        """floor_value=0: sensitivity key absent (degenerate perturbation)."""
        opt = _opt(values=SCENARIOS_3, floor=0.0)
        result = price_option(opt)
        assert "floor_value" not in result.sensitivity

    def test_timing_sensitivity_absent_at_zero_timing(self):
        """timing_expected=0: timing key absent."""
        opt = _opt(values=SCENARIOS_3, expected=0.0, earliest=0.0)
        result = price_option(opt)
        assert "timing_expected" not in result.sensitivity


# ---------------------------------------------------------------------------
# price_all_options
# ---------------------------------------------------------------------------


class TestPriceAllOptions:
    def test_sorted_descending_by_ev(self):
        opt_a = _opt(name="a", values={"s": 50.0}, floor=0.0)
        opt_b = _opt(name="b", values={"s": 200.0}, floor=0.0)
        results = price_all_options([opt_a, opt_b])
        assert results[0].expected_value >= results[1].expected_value

    def test_all_options_included(self):
        opts = [_opt(name=f"opt_{i}", values={"s": float(i * 100)}) for i in range(1, 5)]
        results = price_all_options(opts)
        assert len(results) == 4


# ---------------------------------------------------------------------------
# build_timing_map — normalisation
# ---------------------------------------------------------------------------


class TestBuildTimingMap:
    def test_probabilities_sum_to_one(self):
        opt = _opt()
        tmap = build_timing_map(opt, n_steps=40)
        assert sum(tmap.probabilities) == pytest.approx(1.0, abs=1e-6)

    def test_n_steps_correct(self):
        opt = _opt()
        tmap = build_timing_map(opt, n_steps=20)
        assert len(tmap.time_steps) == 20
        assert len(tmap.probabilities) == 20

    def test_time_steps_within_bounds(self):
        opt = _opt(earliest=1.0, expected=3.0, latest=5.0)
        tmap = build_timing_map(opt)
        assert min(tmap.time_steps) >= opt.timing_earliest - 1e-9
        assert max(tmap.time_steps) <= opt.timing_latest + 1e-9

    def test_expected_timing_between_earliest_and_latest(self):
        opt = _opt(earliest=1.0, expected=2.0, latest=6.0)
        tmap = build_timing_map(opt)
        assert opt.timing_earliest <= tmap.expected_timing <= opt.timing_latest

    def test_degenerate_timing_all_mass_at_single_point(self):
        """earliest == expected == latest: all probability at one point."""
        opt = _opt(earliest=3.0, expected=3.0, latest=3.0)
        tmap = build_timing_map(opt, n_steps=10)
        assert sum(tmap.probabilities) == pytest.approx(1.0, abs=1e-6)
        assert tmap.expected_timing == pytest.approx(3.0, abs=1e-6)

    def test_option_name_propagated(self):
        opt = _opt(name="my_exit")
        tmap = build_timing_map(opt)
        assert tmap.option_name == "my_exit"

    def test_mode_at_expected_has_highest_density(self):
        """For symmetric triangle (a=1, c=3, b=5), mode bin should be near t=3."""
        opt = _opt(earliest=1.0, expected=3.0, latest=5.0)
        tmap = build_timing_map(opt, n_steps=40)
        peak_idx = int(np.argmax(tmap.probabilities))
        peak_time = tmap.time_steps[peak_idx]
        assert abs(peak_time - 3.0) <= (5.0 - 1.0) / 40 * 2  # within 2 bins

    def test_invalid_n_steps_raises(self):
        with pytest.raises(ValueError, match="n_steps must be >= 2"):
            build_timing_map(_opt(), n_steps=1)


# ---------------------------------------------------------------------------
# TimingDistribution schema validation
# ---------------------------------------------------------------------------


class TestTimingDistributionSchema:
    def test_mismatched_lengths_raises(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="must equal"):
            TimingDistribution(
                option_name="x",
                time_steps=[1.0, 2.0, 3.0],
                probabilities=[0.5, 0.5],  # length mismatch
                expected_timing=1.5,
            )

    def test_probs_not_summing_to_one_raises(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="sum to ~1.0"):
            TimingDistribution(
                option_name="x",
                time_steps=[1.0, 2.0],
                probabilities=[0.3, 0.3],  # sums to 0.6
                expected_timing=1.5,
            )


# ---------------------------------------------------------------------------
# price_with_timing_map
# ---------------------------------------------------------------------------


class TestPriceWithTimingMap:
    def test_timing_weighted_ev_positive(self):
        opt = _opt(values=SCENARIOS_3, floor=0.0)
        tmap = build_timing_map(opt)
        ev = price_with_timing_map(opt, tmap)
        assert ev > 0.0

    def test_all_zero_payoffs_gives_zero_ev(self):
        opt = _opt(values={"s": 50.0}, floor=100.0)
        tmap = build_timing_map(opt)
        ev = price_with_timing_map(opt, tmap)
        assert ev == pytest.approx(0.0)

    def test_higher_discount_reduces_ev(self):
        opt_low = _opt(values=SCENARIOS_3, rate=0.05)
        opt_high = _opt(values=SCENARIOS_3, rate=0.20)
        tmap_low = build_timing_map(opt_low)
        tmap_high = build_timing_map(opt_high)
        assert price_with_timing_map(opt_high, tmap_high) < price_with_timing_map(opt_low, tmap_low)

    def test_point_mass_timing_matches_option_pricer(self):
        """Degenerate timing (a=b=c): timing-adjusted EV ≈ option_pricer EV."""
        t = 3.0
        opt = _opt(values={"s": 200.0}, floor=50.0, expected=t, earliest=t, latest=t)
        tmap = build_timing_map(opt, n_steps=10)
        ev_timing = price_with_timing_map(opt, tmap)
        ev_point = price_option(opt).expected_value
        assert ev_timing == pytest.approx(ev_point, rel=1e-3)


# ---------------------------------------------------------------------------
# timing_sensitivity
# ---------------------------------------------------------------------------


class TestTimingSensitivity:
    def test_discount_rate_sensitivity_negative(self):
        """∂EV_timing/∂r < 0: higher discount → lower timing-adjusted EV."""
        opt = _opt(values=SCENARIOS_3, floor=0.0)
        sens = timing_sensitivity(opt)
        assert sens["discount_rate"] < 0.0

    def test_timing_mode_sensitivity_negative(self):
        """∂EV_timing/∂t_mode < 0: later mode → more discounting → lower EV."""
        opt = _opt(values=SCENARIOS_3, floor=0.0)
        sens = timing_sensitivity(opt)
        assert sens["timing_expected"] < 0.0

    def test_degenerate_timing_has_no_mode_sensitivity(self):
        """When earliest == latest, mode perturbation is undefined → key absent."""
        opt = _opt(earliest=3.0, expected=3.0, latest=3.0, values=SCENARIOS_3)
        sens = timing_sensitivity(opt)
        assert "timing_expected" not in sens


# ---------------------------------------------------------------------------
# compare_exit_options
# ---------------------------------------------------------------------------


class TestCompareExitOptions:
    def test_sorted_by_expected_timing(self):
        opt_late = _opt(name="late", earliest=4.0, expected=5.0, latest=6.0)
        opt_early = _opt(name="early", earliest=1.0, expected=2.0, latest=3.0)
        maps = compare_exit_options([opt_late, opt_early])
        assert maps[0].expected_timing <= maps[1].expected_timing

    def test_all_options_included(self):
        opts = [
            _opt(name=f"o{i}", earliest=float(i), expected=float(i) + 1, latest=float(i) + 2)
            for i in range(3)
        ]
        maps = compare_exit_options(opts)
        assert len(maps) == 3
