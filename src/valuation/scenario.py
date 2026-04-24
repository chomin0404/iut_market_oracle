"""DCF valuation model with explicit assumptions and numerical sensitivity analysis.

Model
-----
For each forecast year t = 1 … n:

    Revenue_t = initial_revenue × (1 + g)^t
    EBIT_t    = Revenue_t × ebit_margin
    NOPAT_t   = EBIT_t × (1 − tax_rate)
    Capex_t   = EBIT_t × capex_rate          # net reinvestment proxy
    FCF_t     = NOPAT_t − Capex_t

Terminal value (Gordon Growth):

    FCF_n+1         = FCF_n × (1 + terminal_growth_rate)
    TerminalValue   = FCF_n+1 / (WACC − terminal_growth_rate)

Enterprise value:

    EV = Σ_{t=1}^{n} FCF_t / (1+WACC)^t
       + TerminalValue / (1+WACC)^n

Sensitivity analysis
---------------------
Each float parameter p is perturbed by ±DELTA_REL (relative step) and
the central difference ∂EV/∂p is stored in ScenarioResult.sensitivity.

    ∂EV/∂p ≈ [EV(p + δ) − EV(p − δ)] / (2δ)   where δ = |p| · DELTA_REL

Non-differentiable parameters (forecast_years) are excluded.

Note: floating-point precision accumulates across the sum of discount
factors. For large forecast_years (> 30), relative error may reach
O(10^-12). This is acceptable for valuation purposes.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from schemas import AssumptionSet, ScenarioResult

# Relative step size for numerical differentiation.
# Chosen as a compromise between truncation error (smaller δ is better)
# and cancellation error (larger δ is better). O(h²) central difference.
DELTA_REL: float = 1e-4

# Parameters excluded from sensitivity analysis (non-continuous).
_NON_DIFFERENTIABLE = frozenset(["forecast_years"])


# ---------------------------------------------------------------------------
# Core DCF engine
# ---------------------------------------------------------------------------


def _dcf(
    initial_revenue: float,
    revenue_growth: float,
    ebit_margin: float,
    tax_rate: float,
    capex_rate: float,
    discount_rate: float,
    terminal_growth_rate: float,
    forecast_years: int,
) -> float:
    """Compute enterprise value via DCF.  All rates are annual decimals.

    Raises
    ------
    ValueError
        If discount_rate <= terminal_growth_rate (Gordon Growth undefined).
    """
    if discount_rate <= terminal_growth_rate:
        raise ValueError(
            f"discount_rate ({discount_rate}) must be > terminal_growth_rate "
            f"({terminal_growth_rate}) for a finite terminal value."
        )
    if forecast_years < 1:
        raise ValueError("forecast_years must be >= 1")

    ev = 0.0
    fcf_last = 0.0

    for t in range(1, forecast_years + 1):
        revenue_t = initial_revenue * (1.0 + revenue_growth) ** t
        ebit_t = revenue_t * ebit_margin
        nopat_t = ebit_t * (1.0 - tax_rate)
        capex_t = ebit_t * capex_rate
        fcf_t = nopat_t - capex_t
        ev += fcf_t / (1.0 + discount_rate) ** t
        fcf_last = fcf_t

    terminal_fcf = fcf_last * (1.0 + terminal_growth_rate)
    terminal_value = terminal_fcf / (discount_rate - terminal_growth_rate)
    ev += terminal_value / (1.0 + discount_rate) ** forecast_years

    return ev


def _params_as_floats(assumption: AssumptionSet) -> dict[str, float]:
    """Extract only float-valued parameters (skip bool, str, int-as-int)."""
    out: dict[str, float] = {}
    for k, v in assumption.params.items():
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            out[k] = float(v)
    return out


def _ev_from_assumption(assumption: AssumptionSet) -> float:
    """Call _dcf using the full parameter dict from an AssumptionSet."""
    p = assumption.params
    return _dcf(
        initial_revenue=float(p["initial_revenue"]),
        revenue_growth=float(p["revenue_growth"]),
        ebit_margin=float(p["ebit_margin"]),
        tax_rate=float(p["tax_rate"]),
        capex_rate=float(p["capex_rate"]),
        discount_rate=float(p["discount_rate"]),
        terminal_growth_rate=float(p["terminal_growth_rate"]),
        forecast_years=int(p["forecast_years"]),
    )


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------


def _sensitivity(assumption: AssumptionSet, base_ev: float) -> dict[str, float]:
    """Central-difference ∂EV/∂p for each differentiable float parameter.

    Parameters
    ----------
    assumption:
        Base assumption set.
    base_ev:
        Pre-computed EV at the base parameters (avoids redundant evaluation).

    Returns
    -------
    dict[str, float]
        Mapping parameter name → ∂EV/∂p.
    """
    result: dict[str, float] = {}
    float_params = _params_as_floats(assumption)

    for param, base_val in float_params.items():
        if param in _NON_DIFFERENTIABLE:
            continue

        delta = abs(base_val) * DELTA_REL if base_val != 0.0 else DELTA_REL

        # Perturbed assumption sets (copy params dict, mutate one key)
        def _perturbed(shift: float) -> AssumptionSet:
            new_params = dict(assumption.params)
            new_params[param] = base_val + shift
            return AssumptionSet(
                name=assumption.name,
                version=assumption.version,
                params=new_params,
            )

        try:
            ev_up = _ev_from_assumption(_perturbed(+delta))
            ev_dn = _ev_from_assumption(_perturbed(-delta))
            result[param] = (ev_up - ev_dn) / (2.0 * delta)
        except ValueError:
            # Skip parameters whose perturbation violates model constraints
            # (e.g. terminal_growth_rate bumped above discount_rate).
            pass

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_scenario(
    assumption: AssumptionSet,
    output_path: str | None = None,
) -> ScenarioResult:
    """Run one DCF scenario and return a validated ScenarioResult.

    Parameters
    ----------
    assumption:
        Fully-specified AssumptionSet (loaded from YAML or constructed in code).
    output_path:
        Optional path where the result will be written (recorded in metadata
        only; caller is responsible for saving).

    Returns
    -------
    ScenarioResult
        Enterprise value in the same monetary unit as ``initial_revenue``,
        plus ∂EV/∂p sensitivity for each float parameter.
    """
    base_ev = _ev_from_assumption(assumption)
    sensitivity = _sensitivity(assumption, base_ev)

    return ScenarioResult(
        scenario_name=assumption.name,
        assumption_version=assumption.version,
        value=base_ev,
        unit="JPY millions",
        sensitivity=sensitivity,
        output_path=output_path,
    )


def load_assumption_yaml(path: str | Path) -> AssumptionSet:
    """Load an AssumptionSet from a YAML scenario file.

    Expected YAML keys: name, version, params, (optional) random_seed, description.
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return AssumptionSet(
        name=raw["name"],
        version=str(raw.get("version", "1.0")),
        params=raw["params"],
        random_seed=raw.get("random_seed"),
        description=raw.get("description", ""),
    )


def run_all_scenarios(
    scenario_dir: str | Path,
    pattern: str = "*.yaml",
) -> list[ScenarioResult]:
    """Load and run every YAML scenario file found in *scenario_dir*.

    Returns
    -------
    list[ScenarioResult]
        Results sorted by scenario name for deterministic ordering.
    """
    results: list[ScenarioResult] = []
    for yaml_path in sorted(Path(scenario_dir).glob(pattern)):
        assumption = load_assumption_yaml(yaml_path)
        results.append(run_scenario(assumption, output_path=str(yaml_path)))
    return results
