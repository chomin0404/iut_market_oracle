"""Tests for src/huh_twin/config_runner.py.

Two test scopes
---------------
Fixture-driven (tmp_path)
    Inline YAML written per test — verifies parsing logic in isolation,
    independent of the committed configs/valuation.yaml.

Integration (VALUATION_CONFIG)
    Reads the committed configs/valuation.yaml directly — verifies that the
    real file satisfies all structural and numeric contracts.
    Skipped automatically when the file is absent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from huh_twin.config_runner import ValuationRunResult, load_valuation_config
from huh_twin.valuation import DCFInputs, dcf_valuation

VALUATION_CONFIG = Path(__file__).parent.parent / "configs" / "valuation.yaml"

# ---------------------------------------------------------------------------
# Minimal YAML helpers
# ---------------------------------------------------------------------------

_MINIMAL_YAML = """\
base_case:
  initial_fcf: 100.0
  growth_rate: 0.10
  discount_rate: 0.10
  forecast_years: 5
  terminal_growth_rate: 0.03

sensitivity:
  growth_rate:
    values: [0.05, 0.10, 0.15]
  discount_rate:
    values: [0.08, 0.10, 0.12]
  two_way:
    x_name: growth_rate
    x_values: [0.08, 0.10, 0.12]
    y_name: discount_rate
    y_values: [0.09, 0.10, 0.11]
"""

_NO_TWO_WAY_YAML = """\
base_case:
  initial_fcf: 200.0
  growth_rate: 0.05
  discount_rate: 0.12
  forecast_years: 3
  terminal_growth_rate: 0.02

sensitivity:
  growth_rate:
    values: [0.03, 0.05, 0.08]
"""


def _write(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "valuation.yaml"
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# load_valuation_config — base_inputs
# ---------------------------------------------------------------------------


class TestBaseInputsParsing:
    def test_returns_valuation_run_result(self, tmp_path: Path) -> None:
        result = load_valuation_config(_write(tmp_path, _MINIMAL_YAML))
        assert isinstance(result, ValuationRunResult)

    def test_base_inputs_type(self, tmp_path: Path) -> None:
        result = load_valuation_config(_write(tmp_path, _MINIMAL_YAML))
        assert isinstance(result.base_inputs, DCFInputs)

    def test_base_inputs_values(self, tmp_path: Path) -> None:
        result = load_valuation_config(_write(tmp_path, _MINIMAL_YAML))
        b = result.base_inputs
        assert b.initial_fcf == pytest.approx(100.0)
        assert b.growth_rate == pytest.approx(0.10)
        assert b.discount_rate == pytest.approx(0.10)
        assert b.forecast_years == 5
        assert b.terminal_growth_rate == pytest.approx(0.03)

    def test_alternate_base_case_parsed_correctly(self, tmp_path: Path) -> None:
        result = load_valuation_config(_write(tmp_path, _NO_TWO_WAY_YAML))
        b = result.base_inputs
        assert b.initial_fcf == pytest.approx(200.0)
        assert b.forecast_years == 3


# ---------------------------------------------------------------------------
# load_valuation_config — one_way results
# ---------------------------------------------------------------------------


class TestOneWayResults:
    def test_one_way_keys_match_yaml_variables(self, tmp_path: Path) -> None:
        result = load_valuation_config(_write(tmp_path, _MINIMAL_YAML))
        assert set(result.one_way.keys()) == {"growth_rate", "discount_rate"}

    def test_two_way_key_absent_from_one_way(self, tmp_path: Path) -> None:
        result = load_valuation_config(_write(tmp_path, _MINIMAL_YAML))
        assert "two_way" not in result.one_way

    def test_row_count_matches_values_list(self, tmp_path: Path) -> None:
        result = load_valuation_config(_write(tmp_path, _MINIMAL_YAML))
        assert len(result.one_way["growth_rate"]) == 3
        assert len(result.one_way["discount_rate"]) == 3

    def test_variable_name_propagated(self, tmp_path: Path) -> None:
        result = load_valuation_config(_write(tmp_path, _MINIMAL_YAML))
        for row in result.one_way["growth_rate"]:
            assert row.variable == "growth_rate"

    def test_swept_values_match_yaml(self, tmp_path: Path) -> None:
        result = load_valuation_config(_write(tmp_path, _MINIMAL_YAML))
        actual = [r.value for r in result.one_way["growth_rate"]]
        assert actual == pytest.approx([0.05, 0.10, 0.15])

    def test_enterprise_values_match_dcf_valuation(self, tmp_path: Path) -> None:
        result = load_valuation_config(_write(tmp_path, _MINIMAL_YAML))
        base = result.base_inputs
        for row in result.one_way["growth_rate"]:
            from huh_twin.sensitivity import _replace_inputs
            expected = dcf_valuation(_replace_inputs(base, growth_rate=row.value)).enterprise_value
            assert row.enterprise_value == pytest.approx(expected, rel=1e-12)

    def test_pct_change_zero_at_base_growth_rate(self, tmp_path: Path) -> None:
        result = load_valuation_config(_write(tmp_path, _MINIMAL_YAML))
        base_row = next(r for r in result.one_way["growth_rate"] if r.value == pytest.approx(0.10))
        assert base_row.pct_change_vs_base == pytest.approx(0.0, abs=1e-12)

    def test_growth_rate_sweep_is_monotone_increasing(self, tmp_path: Path) -> None:
        result = load_valuation_config(_write(tmp_path, _MINIMAL_YAML))
        evs = [r.enterprise_value for r in result.one_way["growth_rate"]]
        assert evs == sorted(evs)

    def test_discount_rate_sweep_is_monotone_decreasing(self, tmp_path: Path) -> None:
        result = load_valuation_config(_write(tmp_path, _MINIMAL_YAML))
        evs = [r.enterprise_value for r in result.one_way["discount_rate"]]
        assert evs == sorted(evs, reverse=True)

    def test_no_two_way_key_gives_none(self, tmp_path: Path) -> None:
        result = load_valuation_config(_write(tmp_path, _NO_TWO_WAY_YAML))
        assert result.two_way is None

    def test_no_two_way_one_way_still_populated(self, tmp_path: Path) -> None:
        result = load_valuation_config(_write(tmp_path, _NO_TWO_WAY_YAML))
        assert "growth_rate" in result.one_way
        assert len(result.one_way["growth_rate"]) == 3


# ---------------------------------------------------------------------------
# load_valuation_config — two_way results
# ---------------------------------------------------------------------------


class TestTwoWayResults:
    def test_two_way_grid_size(self, tmp_path: Path) -> None:
        # x_values=[0.08, 0.10, 0.12] × y_values=[0.09, 0.10, 0.11] → 9 cells
        result = load_valuation_config(_write(tmp_path, _MINIMAL_YAML))
        assert result.two_way is not None
        assert len(result.two_way) == 9

    def test_two_way_axis_names(self, tmp_path: Path) -> None:
        result = load_valuation_config(_write(tmp_path, _MINIMAL_YAML))
        assert result.two_way is not None
        for point in result.two_way:
            assert point.x_name == "growth_rate"
            assert point.y_name == "discount_rate"

    def test_two_way_pct_change_zero_at_base(self, tmp_path: Path) -> None:
        result = load_valuation_config(_write(tmp_path, _MINIMAL_YAML))
        assert result.two_way is not None
        base_point = next(
            p for p in result.two_way
            if p.x_value == pytest.approx(0.10) and p.y_value == pytest.approx(0.10)
        )
        assert base_point.pct_change_vs_base == pytest.approx(0.0, abs=1e-12)

    def test_two_way_all_x_values_present(self, tmp_path: Path) -> None:
        result = load_valuation_config(_write(tmp_path, _MINIMAL_YAML))
        assert result.two_way is not None
        x_vals = sorted({p.x_value for p in result.two_way})
        assert x_vals == pytest.approx([0.08, 0.10, 0.12])

    def test_two_way_all_y_values_present(self, tmp_path: Path) -> None:
        result = load_valuation_config(_write(tmp_path, _MINIMAL_YAML))
        assert result.two_way is not None
        y_vals = sorted({p.y_value for p in result.two_way})
        assert y_vals == pytest.approx([0.09, 0.10, 0.11])


# ---------------------------------------------------------------------------
# Integration — configs/valuation.yaml
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not VALUATION_CONFIG.exists(), reason="configs/valuation.yaml not found")
class TestIntegrationValuationYaml:
    def test_loads_without_error(self) -> None:
        result = load_valuation_config(VALUATION_CONFIG)
        assert isinstance(result, ValuationRunResult)

    def test_base_case_matches_yaml(self) -> None:
        result = load_valuation_config(VALUATION_CONFIG)
        b = result.base_inputs
        assert b.initial_fcf == pytest.approx(100.0)
        assert b.growth_rate == pytest.approx(0.10)
        assert b.discount_rate == pytest.approx(0.10)
        assert b.forecast_years == 5
        assert b.terminal_growth_rate == pytest.approx(0.03)

    def test_one_way_variables_present(self) -> None:
        result = load_valuation_config(VALUATION_CONFIG)
        assert {"growth_rate", "discount_rate", "terminal_growth_rate"} <= set(result.one_way)

    def test_growth_rate_sweep_length(self) -> None:
        result = load_valuation_config(VALUATION_CONFIG)
        assert len(result.one_way["growth_rate"]) == 5

    def test_discount_rate_sweep_length(self) -> None:
        result = load_valuation_config(VALUATION_CONFIG)
        assert len(result.one_way["discount_rate"]) == 5

    def test_terminal_growth_rate_sweep_length(self) -> None:
        result = load_valuation_config(VALUATION_CONFIG)
        assert len(result.one_way["terminal_growth_rate"]) == 4

    def test_two_way_grid_size(self) -> None:
        # x_values=[0.08, 0.10, 0.12] × y_values=[0.09, 0.10, 0.11] → 9
        result = load_valuation_config(VALUATION_CONFIG)
        assert result.two_way is not None
        assert len(result.two_way) == 9

    def test_growth_rate_sensitivity_monotone(self) -> None:
        result = load_valuation_config(VALUATION_CONFIG)
        evs = [r.enterprise_value for r in result.one_way["growth_rate"]]
        assert evs == sorted(evs)

    def test_discount_rate_sensitivity_monotone(self) -> None:
        result = load_valuation_config(VALUATION_CONFIG)
        evs = [r.enterprise_value for r in result.one_way["discount_rate"]]
        assert evs == sorted(evs, reverse=True)

    def test_terminal_growth_rate_sensitivity_monotone(self) -> None:
        # Higher terminal_growth_rate → higher terminal value → higher EV
        result = load_valuation_config(VALUATION_CONFIG)
        evs = [r.enterprise_value for r in result.one_way["terminal_growth_rate"]]
        assert evs == sorted(evs)

    def test_all_enterprise_values_positive(self) -> None:
        result = load_valuation_config(VALUATION_CONFIG)
        for rows in result.one_way.values():
            for row in rows:
                assert row.enterprise_value > 0.0
        if result.two_way:
            for point in result.two_way:
                assert point.enterprise_value > 0.0
