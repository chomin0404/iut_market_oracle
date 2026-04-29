"""Option-style exit value pricing for the Exit Strategy Engine (T900).

Pricing model
-------------
For each exit option with scenarios {s} and enterprise values {V_s}:

    payoff_s = max(V_s − floor, 0)                 [option-style truncation]
    pv_s     = payoff_s / (1 + r)^{t_expected}     [discount to present value]
    EV       = Σ_s p_s · pv_s                      [probability-weighted mean]

where:
    floor         : minimum deal value (ExitOption.floor_value)
    r             : annual discount rate (ExitOption.discount_rate)
    t_expected    : expected exit timing in years (ExitOption.timing_expected)
    p_s           : scenario probability (uniform 1/n if not supplied)

Sensitivity analysis
--------------------
Central-difference ∂EV/∂p for each differentiable scalar parameter:

    ∂EV/∂p ≈ [EV(p + δ) − EV(p − δ)] / (2δ)    δ = |p| · DELTA_REL

Parameters included: discount_rate, timing_expected, floor_value.

Note: the floor_value perturbation is clipped at 0 so the lower bound
never goes negative; sensitivity is computed only when the floor is
strictly above 0 (otherwise the downward perturbation is degenerate).
"""

from __future__ import annotations

from schemas import ExitOption, ExitValueSummary

# Relative step for central-difference (same convention as valuation module).
DELTA_REL: float = 1e-4

# Parameters included in sensitivity analysis.
_SENSITIVITY_PARAMS = frozenset(["discount_rate", "timing_expected", "floor_value"])


# ---------------------------------------------------------------------------
# Core arithmetic
# ---------------------------------------------------------------------------


def _payoff(value: float, floor: float) -> float:
    """Option-style payoff: max(V − floor, 0)."""
    return max(value - floor, 0.0)


def _pv(payoff: float, discount_rate: float, timing: float) -> float:
    """Discount a payoff to present value.

    PV = payoff / (1 + r)^t

    When t == 0 the denominator is 1 (no discounting).
    """
    return payoff / (1.0 + discount_rate) ** timing


def _compute_ev(
    values: dict[str, float],
    probs: dict[str, float],
    floor: float,
    discount_rate: float,
    timing: float,
) -> float:
    """Compute the probability-weighted expected present value.

    Parameters
    ----------
    values:
        scenario name → enterprise value.
    probs:
        scenario name → probability (must sum to 1.0).
    floor:
        Minimum deal value; payoff = max(V − floor, 0).
    discount_rate:
        Annual WACC.
    timing:
        Expected exit timing in years (used for discounting).
    """
    total = 0.0
    for name, v in values.items():
        p = probs[name]
        pf = _payoff(v, floor)
        total += p * _pv(pf, discount_rate, timing)
    return total


def _uniform_probs(scenario_names: list[str]) -> dict[str, float]:
    """Return equal probability for each scenario."""
    n = len(scenario_names)
    return {name: 1.0 / n for name in scenario_names}


# ---------------------------------------------------------------------------
# Sensitivity
# ---------------------------------------------------------------------------


def _sensitivity(
    option: ExitOption,
    probs: dict[str, float],
    base_ev: float,  # noqa: ARG001  — kept for API consistency with valuation module
) -> dict[str, float]:
    """Central-difference sensitivity ∂EV/∂p for key parameters.

    Parameters
    ----------
    option:
        Base exit option.
    probs:
        Scenario probabilities used for EV computation.
    base_ev:
        Pre-computed EV at base parameters (unused here; included for
        signature symmetry with the valuation module).

    Returns
    -------
    dict mapping parameter name → ∂EV/∂p.
    """
    result: dict[str, float] = {}
    values = option.value_by_scenario

    def _ev_at(floor: float, rate: float, timing: float) -> float:
        return _compute_ev(values, probs, floor, rate, timing)

    base_floor = option.floor_value
    base_rate = option.discount_rate
    base_timing = option.timing_expected

    # ∂EV/∂discount_rate
    delta_r = max(abs(base_rate) * DELTA_REL, DELTA_REL)
    result["discount_rate"] = (
        _ev_at(base_floor, base_rate + delta_r, base_timing)
        - _ev_at(base_floor, base_rate - delta_r, base_timing)
    ) / (2.0 * delta_r)

    # ∂EV/∂timing_expected  (only when timing > 0; otherwise PV is constant)
    if base_timing > 0.0:
        delta_t = max(abs(base_timing) * DELTA_REL, DELTA_REL)
        t_lo = max(base_timing - delta_t, 0.0)
        t_hi = base_timing + delta_t
        actual_delta = (t_hi - t_lo) / 2.0
        result["timing_expected"] = (
            _ev_at(base_floor, base_rate, t_hi) - _ev_at(base_floor, base_rate, t_lo)
        ) / (2.0 * actual_delta)

    # ∂EV/∂floor_value  (only when floor > 0 to avoid clipping issues)
    if base_floor > 0.0:
        delta_f = max(abs(base_floor) * DELTA_REL, DELTA_REL)
        result["floor_value"] = (
            _ev_at(base_floor + delta_f, base_rate, base_timing)
            - _ev_at(base_floor - delta_f, base_rate, base_timing)
        ) / (2.0 * delta_f)

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def price_option(
    option: ExitOption,
    scenario_probs: dict[str, float] | None = None,
) -> ExitValueSummary:
    """Price one exit option and return a validated ExitValueSummary.

    Parameters
    ----------
    option:
        Fully-specified exit option (loaded from config or constructed in code).
    scenario_probs:
        Optional mapping scenario name → probability.  Must sum to ~1.0 and
        cover all scenarios in option.value_by_scenario.
        If None, equal probability 1/n is assigned to each scenario.

    Returns
    -------
    ExitValueSummary
        Per-scenario payoffs and PVs, expected value, and sensitivity surface.

    Raises
    ------
    ValueError
        If scenario_probs keys do not match value_by_scenario keys, or
        if probabilities do not sum to ~1.0.
    """
    names = list(option.value_by_scenario.keys())

    if scenario_probs is None:
        probs = _uniform_probs(names)
    else:
        if set(scenario_probs.keys()) != set(names):
            raise ValueError(
                f"scenario_probs keys {set(scenario_probs.keys())} must match "
                f"value_by_scenario keys {set(names)}"
            )
        total = sum(scenario_probs.values())
        if not (0.999 <= total <= 1.001):
            raise ValueError(f"scenario_probs must sum to ~1.0, got {total:.6f}")
        probs = scenario_probs

    payoffs: dict[str, float] = {
        name: _payoff(v, option.floor_value) for name, v in option.value_by_scenario.items()
    }
    pvs: dict[str, float] = {
        name: _pv(pf, option.discount_rate, option.timing_expected) for name, pf in payoffs.items()
    }
    ev = sum(probs[name] * pvs[name] for name in names)
    sens = _sensitivity(option, probs, ev)

    return ExitValueSummary(
        option_name=option.name,
        exit_type=option.exit_type,
        scenario_payoffs=payoffs,
        scenario_pvs=pvs,
        expected_value=ev,
        sensitivity=sens,
    )


def price_all_options(
    options: list[ExitOption],
    scenario_probs: dict[str, float] | None = None,
) -> list[ExitValueSummary]:
    """Price all exit options and return summaries sorted by expected_value descending.

    Parameters
    ----------
    options:
        List of exit options (may have different scenario sets).
    scenario_probs:
        Applied uniformly to all options whose scenario keys match.
        Options with non-matching keys fall back to uniform probabilities.

    Returns
    -------
    list[ExitValueSummary]
        Sorted highest expected_value first for easy comparison.
    """
    results: list[ExitValueSummary] = []
    for opt in options:
        # Use provided probs only if the keys match; otherwise fall back.
        if scenario_probs is not None and set(scenario_probs.keys()) == set(
            opt.value_by_scenario.keys()
        ):
            probs: dict[str, float] | None = scenario_probs
        else:
            probs = None
        results.append(price_option(opt, probs))

    return sorted(results, key=lambda r: r.expected_value, reverse=True)
