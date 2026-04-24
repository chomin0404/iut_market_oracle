from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product

from huh_twin.valuation import DCFInputs, dcf_valuation


@dataclass(slots=True)
class SensitivityPoint:
    variable: str
    value: float
    enterprise_value: float
    pct_change_vs_base: float


@dataclass(slots=True)
class SensitivityGridPoint:
    x_name: str
    x_value: float
    y_name: str
    y_value: float
    enterprise_value: float
    pct_change_vs_base: float


def _replace_inputs(base: DCFInputs, **updates: float) -> DCFInputs:
    data = asdict(base)
    data.update(updates)
    return DCFInputs(**data)


def one_way_sensitivity(
    base_inputs: DCFInputs,
    variable: str,
    values: list[float],
) -> list[SensitivityPoint]:
    base_result = dcf_valuation(base_inputs)
    rows: list[SensitivityPoint] = []

    for value in values:
        scenario_inputs = _replace_inputs(base_inputs, **{variable: value})
        scenario_value = dcf_valuation(scenario_inputs).enterprise_value
        pct_change = (scenario_value / base_result.enterprise_value) - 1.0

        rows.append(
            SensitivityPoint(
                variable=variable,
                value=value,
                enterprise_value=scenario_value,
                pct_change_vs_base=pct_change,
            )
        )
    return rows


def two_way_sensitivity(
    base_inputs: DCFInputs,
    x_name: str,
    x_values: list[float],
    y_name: str,
    y_values: list[float],
) -> list[SensitivityGridPoint]:
    base_result = dcf_valuation(base_inputs)
    rows: list[SensitivityGridPoint] = []

    for x_value, y_value in product(x_values, y_values):
        scenario_inputs = _replace_inputs(base_inputs, **{x_name: x_value, y_name: y_value})
        scenario_value = dcf_valuation(scenario_inputs).enterprise_value
        pct_change = (scenario_value / base_result.enterprise_value) - 1.0

        rows.append(
            SensitivityGridPoint(
                x_name=x_name,
                x_value=x_value,
                y_name=y_name,
                y_value=y_value,
                enterprise_value=scenario_value,
                pct_change_vs_base=pct_change,
            )
        )
    return rows


def to_row_dicts_one_way(rows: list[SensitivityPoint]) -> list[dict[str, float | str]]:
    return [
        {
            "variable": row.variable,
            "value": row.value,
            "enterprise_value": row.enterprise_value,
            "pct_change_vs_base": row.pct_change_vs_base,
        }
        for row in rows
    ]


def to_row_dicts_two_way(rows: list[SensitivityGridPoint]) -> list[dict[str, float | str]]:
    return [
        {
            "x_name": row.x_name,
            "x_value": row.x_value,
            "y_name": row.y_name,
            "y_value": row.y_value,
            "enterprise_value": row.enterprise_value,
            "pct_change_vs_base": row.pct_change_vs_base,
        }
        for row in rows
    ]
