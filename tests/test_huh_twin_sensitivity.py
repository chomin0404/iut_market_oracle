"""Tests for src/huh_twin/sensitivity.py.

Invariants verified
-------------------
_replace_inputs
    Produces a new DCFInputs with exactly the specified field(s) changed;
    the original is not mutated.

one_way_sensitivity
    - Output length equals len(values).
    - Each row carries the correct variable name and swept value.
    - enterprise_value matches dcf_valuation called with that exact input.
    - pct_change_vs_base = (ev / base_ev) - 1.
    - pct_change == 0 when the swept value equals the base value.
    - Monotone: growth_rate ↑ → EV ↑  /  discount_rate ↑ → EV ↓.

two_way_sensitivity
    - Output length equals len(x_values) × len(y_values).
    - Every (x, y) pair from the Cartesian product appears exactly once.
    - pct_change == 0 when both axes are at their base values.
    - Grid point EV matches dcf_valuation for that (x, y) combination.

to_row_dicts_one_way / to_row_dicts_two_way
    - Return list of plain dicts with the correct keys.
    - Values round-trip from the original dataclasses without loss.
"""

from __future__ import annotations

import math

import pytest

from huh_twin.sensitivity import (
    SensitivityGridPoint,
    SensitivityPoint,
    _replace_inputs,
    one_way_sensitivity,
    to_row_dicts_one_way,
    to_row_dicts_two_way,
    two_way_sensitivity,
)
from huh_twin.valuation import DCFInputs, dcf_valuation

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

BASE = DCFInputs(
    initial_fcf=100.0,
    growth_rate=0.10,
    discount_rate=0.10,
    forecast_years=5,
    terminal_growth_rate=0.03,
)


# ---------------------------------------------------------------------------
# _replace_inputs
# ---------------------------------------------------------------------------


class TestReplaceInputs:
    def test_single_field_is_updated(self) -> None:
        updated = _replace_inputs(BASE, growth_rate=0.20)
        assert updated.growth_rate == pytest.approx(0.20)

    def test_unmodified_fields_are_preserved(self) -> None:
        updated = _replace_inputs(BASE, growth_rate=0.20)
        assert updated.initial_fcf == BASE.initial_fcf
        assert updated.discount_rate == BASE.discount_rate
        assert updated.forecast_years == BASE.forecast_years
        assert updated.terminal_growth_rate == BASE.terminal_growth_rate

    def test_multiple_fields_updated_simultaneously(self) -> None:
        updated = _replace_inputs(BASE, growth_rate=0.15, discount_rate=0.12)
        assert updated.growth_rate == pytest.approx(0.15)
        assert updated.discount_rate == pytest.approx(0.12)

    def test_original_is_not_mutated(self) -> None:
        _replace_inputs(BASE, growth_rate=0.99)
        assert BASE.growth_rate == pytest.approx(0.10)

    def test_returns_dcf_inputs_instance(self) -> None:
        assert isinstance(_replace_inputs(BASE, growth_rate=0.05), DCFInputs)


# ---------------------------------------------------------------------------
# one_way_sensitivity
# ---------------------------------------------------------------------------


class TestOneWaySensitivity:
    def test_output_length_equals_values_length(self) -> None:
        values = [0.05, 0.10, 0.15, 0.20]
        rows = one_way_sensitivity(BASE, "growth_rate", values)
        assert len(rows) == len(values)

    def test_empty_values_returns_empty_list(self) -> None:
        assert one_way_sensitivity(BASE, "growth_rate", []) == []

    def test_variable_name_propagated(self) -> None:
        rows = one_way_sensitivity(BASE, "discount_rate", [0.08, 0.12])
        assert all(r.variable == "discount_rate" for r in rows)

    def test_swept_value_matches_input(self) -> None:
        values = [0.05, 0.10, 0.20]
        rows = one_way_sensitivity(BASE, "growth_rate", values)
        assert [r.value for r in rows] == pytest.approx(values)

    def test_enterprise_value_matches_dcf_valuation(self) -> None:
        value = 0.15
        rows = one_way_sensitivity(BASE, "growth_rate", [value])
        expected = dcf_valuation(_replace_inputs(BASE, growth_rate=value)).enterprise_value
        assert rows[0].enterprise_value == pytest.approx(expected, rel=1e-12)

    def test_pct_change_formula(self) -> None:
        base_ev = dcf_valuation(BASE).enterprise_value
        rows = one_way_sensitivity(BASE, "growth_rate", [0.05, 0.15, 0.20])
        for row in rows:
            expected_pct = (row.enterprise_value / base_ev) - 1.0
            assert row.pct_change_vs_base == pytest.approx(expected_pct, rel=1e-12)

    def test_pct_change_is_zero_at_base_value(self) -> None:
        rows = one_way_sensitivity(BASE, "growth_rate", [BASE.growth_rate])
        assert rows[0].pct_change_vs_base == pytest.approx(0.0, abs=1e-12)

    def test_higher_growth_increases_ev(self) -> None:
        rows = one_way_sensitivity(BASE, "growth_rate", [0.05, 0.10, 0.20])
        evs = [r.enterprise_value for r in rows]
        assert evs == sorted(evs)

    def test_higher_discount_rate_decreases_ev(self) -> None:
        rows = one_way_sensitivity(BASE, "discount_rate", [0.08, 0.10, 0.15])
        evs = [r.enterprise_value for r in rows]
        assert evs == sorted(evs, reverse=True)

    def test_pct_change_positive_for_above_base_growth(self) -> None:
        rows = one_way_sensitivity(BASE, "growth_rate", [BASE.growth_rate + 0.05])
        assert rows[0].pct_change_vs_base > 0.0

    def test_pct_change_negative_for_below_base_growth(self) -> None:
        rows = one_way_sensitivity(BASE, "growth_rate", [BASE.growth_rate - 0.05])
        assert rows[0].pct_change_vs_base < 0.0

    def test_all_enterprise_values_finite(self) -> None:
        rows = one_way_sensitivity(BASE, "growth_rate", [0.01, 0.05, 0.10, 0.15])
        assert all(math.isfinite(r.enterprise_value) for r in rows)

    def test_returns_sensitivity_point_instances(self) -> None:
        rows = one_way_sensitivity(BASE, "growth_rate", [0.10])
        assert isinstance(rows[0], SensitivityPoint)


# ---------------------------------------------------------------------------
# two_way_sensitivity
# ---------------------------------------------------------------------------


class TestTwoWaySensitivity:
    X_VALUES = [0.05, 0.10, 0.15]
    Y_VALUES = [0.08, 0.10, 0.12]

    def test_output_length_is_cartesian_product(self) -> None:
        rows = two_way_sensitivity(
            BASE, "growth_rate", self.X_VALUES, "discount_rate", self.Y_VALUES
        )
        assert len(rows) == len(self.X_VALUES) * len(self.Y_VALUES)

    def test_all_combinations_present(self) -> None:
        rows = two_way_sensitivity(
            BASE, "growth_rate", self.X_VALUES, "discount_rate", self.Y_VALUES
        )
        pairs = {(r.x_value, r.y_value) for r in rows}
        from itertools import product
        expected = {(x, y) for x, y in product(self.X_VALUES, self.Y_VALUES)}
        assert pairs == expected

    def test_axis_names_propagated(self) -> None:
        rows = two_way_sensitivity(
            BASE, "growth_rate", self.X_VALUES, "discount_rate", self.Y_VALUES
        )
        assert all(r.x_name == "growth_rate" for r in rows)
        assert all(r.y_name == "discount_rate" for r in rows)

    def test_pct_change_zero_at_base_values(self) -> None:
        rows = two_way_sensitivity(
            BASE,
            "growth_rate", [BASE.growth_rate],
            "discount_rate", [BASE.discount_rate],
        )
        assert len(rows) == 1
        assert rows[0].pct_change_vs_base == pytest.approx(0.0, abs=1e-12)

    def test_enterprise_value_matches_dcf_valuation(self) -> None:
        x, y = 0.15, 0.12
        rows = two_way_sensitivity(BASE, "growth_rate", [x], "discount_rate", [y])
        expected = dcf_valuation(
            _replace_inputs(BASE, growth_rate=x, discount_rate=y)
        ).enterprise_value
        assert rows[0].enterprise_value == pytest.approx(expected, rel=1e-12)

    def test_pct_change_formula(self) -> None:
        base_ev = dcf_valuation(BASE).enterprise_value
        rows = two_way_sensitivity(
            BASE, "growth_rate", self.X_VALUES, "discount_rate", self.Y_VALUES
        )
        for row in rows:
            expected = (row.enterprise_value / base_ev) - 1.0
            assert row.pct_change_vs_base == pytest.approx(expected, rel=1e-12)

    def test_empty_x_values_returns_empty(self) -> None:
        rows = two_way_sensitivity(BASE, "growth_rate", [], "discount_rate", self.Y_VALUES)
        assert rows == []

    def test_empty_y_values_returns_empty(self) -> None:
        rows = two_way_sensitivity(BASE, "growth_rate", self.X_VALUES, "discount_rate", [])
        assert rows == []

    def test_returns_sensitivity_grid_point_instances(self) -> None:
        rows = two_way_sensitivity(BASE, "growth_rate", [0.10], "discount_rate", [0.10])
        assert isinstance(rows[0], SensitivityGridPoint)

    def test_all_enterprise_values_finite(self) -> None:
        rows = two_way_sensitivity(
            BASE, "growth_rate", self.X_VALUES, "discount_rate", self.Y_VALUES
        )
        assert all(math.isfinite(r.enterprise_value) for r in rows)


# ---------------------------------------------------------------------------
# to_row_dicts_one_way
# ---------------------------------------------------------------------------


class TestToRowDictsOneWay:
    _EXPECTED_KEYS = {"variable", "value", "enterprise_value", "pct_change_vs_base"}

    def test_returns_list_of_dicts(self) -> None:
        rows = one_way_sensitivity(BASE, "growth_rate", [0.05, 0.10])
        result = to_row_dicts_one_way(rows)
        assert isinstance(result, list)
        assert all(isinstance(d, dict) for d in result)

    def test_length_preserved(self) -> None:
        rows = one_way_sensitivity(BASE, "growth_rate", [0.05, 0.10, 0.15])
        assert len(to_row_dicts_one_way(rows)) == 3

    def test_correct_keys(self) -> None:
        rows = one_way_sensitivity(BASE, "growth_rate", [0.10])
        d = to_row_dicts_one_way(rows)[0]
        assert set(d.keys()) == self._EXPECTED_KEYS

    def test_values_round_trip(self) -> None:
        rows = one_way_sensitivity(BASE, "growth_rate", [0.05, 0.20])
        dicts = to_row_dicts_one_way(rows)
        for row, d in zip(rows, dicts):
            assert d["variable"] == row.variable
            assert d["value"] == pytest.approx(row.value)
            assert d["enterprise_value"] == pytest.approx(row.enterprise_value)
            assert d["pct_change_vs_base"] == pytest.approx(row.pct_change_vs_base)

    def test_empty_input_returns_empty(self) -> None:
        assert to_row_dicts_one_way([]) == []


# ---------------------------------------------------------------------------
# to_row_dicts_two_way
# ---------------------------------------------------------------------------


class TestToRowDictsTwoWay:
    _EXPECTED_KEYS = {
        "x_name", "x_value", "y_name", "y_value", "enterprise_value", "pct_change_vs_base"
    }

    def test_returns_list_of_dicts(self) -> None:
        rows = two_way_sensitivity(BASE, "growth_rate", [0.05, 0.10], "discount_rate", [0.08, 0.12])
        result = to_row_dicts_two_way(rows)
        assert isinstance(result, list)
        assert all(isinstance(d, dict) for d in result)

    def test_length_preserved(self) -> None:
        rows = two_way_sensitivity(BASE, "growth_rate", [0.05, 0.10], "discount_rate", [0.08, 0.12])
        assert len(to_row_dicts_two_way(rows)) == 4

    def test_correct_keys(self) -> None:
        rows = two_way_sensitivity(BASE, "growth_rate", [0.10], "discount_rate", [0.10])
        d = to_row_dicts_two_way(rows)[0]
        assert set(d.keys()) == self._EXPECTED_KEYS

    def test_values_round_trip(self) -> None:
        rows = two_way_sensitivity(BASE, "growth_rate", [0.05, 0.15], "discount_rate", [0.08, 0.12])
        dicts = to_row_dicts_two_way(rows)
        for row, d in zip(rows, dicts):
            assert d["x_name"] == row.x_name
            assert d["x_value"] == pytest.approx(row.x_value)
            assert d["y_name"] == row.y_name
            assert d["y_value"] == pytest.approx(row.y_value)
            assert d["enterprise_value"] == pytest.approx(row.enterprise_value)
            assert d["pct_change_vs_base"] == pytest.approx(row.pct_change_vs_base)

    def test_empty_input_returns_empty(self) -> None:
        assert to_row_dicts_two_way([]) == []
