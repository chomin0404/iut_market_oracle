"""Load configs/valuation.yaml and execute the full sensitivity sweep.

YAML contract
-------------
base_case:
  initial_fcf: float
  growth_rate: float
  discount_rate: float
  forecast_years: int
  terminal_growth_rate: float

sensitivity:
  <variable>:          # one or more one-way sweep blocks
    values: [float, ...]

  two_way:             # optional; exactly one two-way block
    x_name: str
    x_values: [float, ...]
    y_name: str
    y_values: [float, ...]
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from huh_twin.sensitivity import (
    SensitivityGridPoint,
    SensitivityPoint,
    one_way_sensitivity,
    two_way_sensitivity,
)
from huh_twin.valuation import DCFInputs

_TWO_WAY_KEY = "two_way"


@dataclass(frozen=True)
class ValuationRunResult:
    base_inputs: DCFInputs
    one_way: dict[str, list[SensitivityPoint]]  # variable → sweep rows
    two_way: list[SensitivityGridPoint] | None  # None when absent from config


def load_valuation_config(path: Path) -> ValuationRunResult:
    """Parse *path* and run all sensitivity sweeps defined in the YAML.

    Parameters
    ----------
    path:
        Path to a valuation YAML file (see module docstring for schema).

    Returns
    -------
    ValuationRunResult
        Parsed base inputs and completed sensitivity results.
    """
    raw: dict = yaml.safe_load(path.read_text(encoding="utf-8"))

    base_inputs = DCFInputs(**raw["base_case"])

    one_way: dict[str, list[SensitivityPoint]] = {}
    two_way: list[SensitivityGridPoint] | None = None

    for key, spec in raw.get("sensitivity", {}).items():
        if key == _TWO_WAY_KEY:
            two_way = two_way_sensitivity(
                base_inputs=base_inputs,
                x_name=spec["x_name"],
                x_values=spec["x_values"],
                y_name=spec["y_name"],
                y_values=spec["y_values"],
            )
        else:
            one_way[key] = one_way_sensitivity(
                base_inputs=base_inputs,
                variable=key,
                values=spec["values"],
            )

    return ValuationRunResult(
        base_inputs=base_inputs,
        one_way=one_way,
        two_way=two_way,
    )
