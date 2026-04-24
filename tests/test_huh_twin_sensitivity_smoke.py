from huh_twin.sensitivity import one_way_sensitivity, two_way_sensitivity
from huh_twin.valuation import DCFInputs


def base_inputs() -> DCFInputs:
    return DCFInputs(
        initial_fcf=100.0,
        growth_rate=0.10,
        discount_rate=0.10,
        forecast_years=5,
        terminal_growth_rate=0.03,
    )


def test_one_way_sensitivity_returns_same_length_as_input_values() -> None:
    rows = one_way_sensitivity(
        base_inputs=base_inputs(),
        variable="growth_rate",
        values=[0.05, 0.10, 0.15],
    )

    assert len(rows) == 3


def test_growth_rate_sensitivity_is_monotone_increasing() -> None:
    rows = one_way_sensitivity(
        base_inputs=base_inputs(),
        variable="growth_rate",
        values=[0.05, 0.10, 0.15],
    )

    values = [row.enterprise_value for row in rows]
    assert values[0] < values[1] < values[2]


def test_discount_rate_sensitivity_is_monotone_decreasing() -> None:
    rows = one_way_sensitivity(
        base_inputs=base_inputs(),
        variable="discount_rate",
        values=[0.08, 0.10, 0.12],
    )

    values = [row.enterprise_value for row in rows]
    assert values[0] > values[1] > values[2]


def test_two_way_sensitivity_grid_size_matches_cartesian_product() -> None:
    rows = two_way_sensitivity(
        base_inputs=base_inputs(),
        x_name="growth_rate",
        x_values=[0.08, 0.10, 0.12],
        y_name="discount_rate",
        y_values=[0.09, 0.10],
    )

    assert len(rows) == 6


def test_pct_change_vs_base_is_zero_at_base_case() -> None:
    rows = two_way_sensitivity(
        base_inputs=base_inputs(),
        x_name="growth_rate",
        x_values=[0.10],
        y_name="discount_rate",
        y_values=[0.10],
    )

    assert len(rows) == 1
    assert abs(rows[0].pct_change_vs_base) < 1e-12
