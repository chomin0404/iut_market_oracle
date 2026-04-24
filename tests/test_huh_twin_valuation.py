import math

import pytest

from huh_twin.valuation import (
    DCFInputs,
    dcf_valuation,
    gordon_terminal_value,
    reverse_dcf_implied_growth,
)


def test_gordon_terminal_value_formula() -> None:
    tv = gordon_terminal_value(
        final_year_fcf=100.0,
        discount_rate=0.10,
        terminal_growth_rate=0.03,
    )
    expected = 100.0 * 1.03 / (0.10 - 0.03)
    assert math.isclose(tv, expected, rel_tol=1e-12)


def test_terminal_growth_must_be_less_than_discount_rate() -> None:
    with pytest.raises(ValueError):
        gordon_terminal_value(
            final_year_fcf=100.0,
            discount_rate=0.08,
            terminal_growth_rate=0.08,
        )


def test_dcf_valuation_returns_positive_enterprise_value() -> None:
    result = dcf_valuation(
        DCFInputs(
            initial_fcf=100.0,
            growth_rate=0.10,
            discount_rate=0.10,
            forecast_years=5,
            terminal_growth_rate=0.03,
        )
    )

    assert len(result.projected_fcfs) == 5
    assert len(result.discounted_fcfs) == 5
    assert result.terminal_value > 0
    assert result.discounted_terminal_value > 0
    assert result.enterprise_value > 0


def test_higher_growth_produces_higher_value() -> None:
    low_growth = dcf_valuation(
        DCFInputs(
            initial_fcf=100.0,
            growth_rate=0.05,
            discount_rate=0.10,
            forecast_years=5,
            terminal_growth_rate=0.03,
        )
    )
    high_growth = dcf_valuation(
        DCFInputs(
            initial_fcf=100.0,
            growth_rate=0.15,
            discount_rate=0.10,
            forecast_years=5,
            terminal_growth_rate=0.03,
        )
    )

    assert high_growth.enterprise_value > low_growth.enterprise_value


def test_higher_discount_rate_reduces_value() -> None:
    low_discount = dcf_valuation(
        DCFInputs(
            initial_fcf=100.0,
            growth_rate=0.10,
            discount_rate=0.08,
            forecast_years=5,
            terminal_growth_rate=0.03,
        )
    )
    high_discount = dcf_valuation(
        DCFInputs(
            initial_fcf=100.0,
            growth_rate=0.10,
            discount_rate=0.12,
            forecast_years=5,
            terminal_growth_rate=0.03,
        )
    )

    assert low_discount.enterprise_value > high_discount.enterprise_value


def test_reverse_dcf_recovers_known_growth_rate() -> None:
    true_growth = 0.12
    result = dcf_valuation(
        DCFInputs(
            initial_fcf=100.0,
            growth_rate=true_growth,
            discount_rate=0.10,
            forecast_years=5,
            terminal_growth_rate=0.03,
        )
    )

    implied_growth = reverse_dcf_implied_growth(
        target_enterprise_value=result.enterprise_value,
        initial_fcf=100.0,
        discount_rate=0.10,
        forecast_years=5,
        terminal_growth_rate=0.03,
    )

    assert abs(implied_growth - true_growth) < 1e-6


def test_invalid_dcf_inputs_raise_error() -> None:
    with pytest.raises(ValueError):
        DCFInputs(
            initial_fcf=-1.0,
            growth_rate=0.10,
            discount_rate=0.10,
            forecast_years=5,
            terminal_growth_rate=0.03,
        )
