"""Exit timing probability distribution and timing-adjusted pricing (T900).

Timing model
------------
Exit timing T is modelled as a triangular distribution:

    T ~ Triangular(a=earliest, c=expected, b=latest)

PDF:
              2(t − a)
    f(t) = ──────────────    for a ≤ t ≤ c
             (b − a)(c − a)

              2(b − t)
           = ──────────────   for c < t ≤ b
             (b − a)(b − c)

The triangular distribution is a natural choice when only the minimum,
mode, and maximum are known (analogous to PERT / three-point estimation).

Degenerate case: when a == b (point mass), all probability is placed
at t = a = c = b.

Discretisation
--------------
The continuous PDF is evaluated at n_steps evenly-spaced points over
[earliest, latest] and normalised so Σ P(t_k) = 1.0.  An extra guard
step is added at t=0 when earliest == 0 to avoid an empty grid.

Timing-adjusted pricing
-----------------------
Instead of discounting at timing_expected, the full timing distribution
gives a richer EV:

    EV_timing = Σ_k P(T=t_k) · Σ_s p_s · payoff_s / (1+r)^{t_k}

Sensitivity
-----------
∂EV_timing/∂r       : perturb discount_rate ± δ
∂EV_timing/∂t_mode  : perturb timing_expected (shifts the triangular mode)
                       Only valid when earliest < latest.
"""

from __future__ import annotations

import numpy as np
import scipy.stats as st

from exit.option_pricer import DELTA_REL, _payoff, _uniform_probs
from schemas import ExitOption, TimingDistribution

# Minimum number of discretisation steps.
_MIN_STEPS = 2


# ---------------------------------------------------------------------------
# Triangular distribution helpers
# ---------------------------------------------------------------------------


def _triangular_pdf(
    t: np.ndarray,
    earliest: float,
    expected: float,
    latest: float,
) -> np.ndarray:
    """Evaluate the triangular PDF at each point in t.

    Degenerate case (earliest == latest): returns a spike of 1.0 at the
    nearest grid point; 0.0 elsewhere.
    """
    if np.isclose(earliest, latest):
        # Point mass: assign 1.0 to the closest grid point.
        idx = int(np.argmin(np.abs(t - earliest)))
        pdf = np.zeros_like(t)
        pdf[idx] = 1.0
        return pdf

    c = (expected - earliest) / (latest - earliest)  # shape parameter ∈ [0, 1]
    return st.triang.pdf(t, c=c, loc=earliest, scale=latest - earliest)


# ---------------------------------------------------------------------------
# Public: build_timing_map
# ---------------------------------------------------------------------------


def build_timing_map(
    option: ExitOption,
    n_steps: int = 40,
) -> TimingDistribution:
    """Discretise the triangular exit-timing distribution into a probability map.

    Parameters
    ----------
    option:
        Exit option whose timing_earliest / timing_expected / timing_latest
        define the triangular distribution.
    n_steps:
        Number of discrete time points (>= 2).

    Returns
    -------
    TimingDistribution
        Normalised probability mass over n_steps evenly-spaced time points.

    Raises
    ------
    ValueError
        If n_steps < 2.
    """
    if n_steps < _MIN_STEPS:
        raise ValueError(f"n_steps must be >= {_MIN_STEPS}, got {n_steps}")

    a = option.timing_earliest
    b = option.timing_latest

    # Ensure the grid spans at least a minimal range.
    if np.isclose(a, b):
        t_grid = np.array([a, a])  # degenerate: two identical points
    else:
        t_grid = np.linspace(a, b, n_steps)

    raw_pdf = _triangular_pdf(t_grid, a, option.timing_expected, b)

    total = raw_pdf.sum()
    if total == 0.0:
        # Fallback: uniform distribution (should not occur with valid inputs).
        probs = np.full(len(t_grid), 1.0 / len(t_grid))
    else:
        probs = raw_pdf / total  # normalise so Σ P(t_k) = 1.0

    expected_t = float(np.dot(t_grid, probs))

    return TimingDistribution(
        option_name=option.name,
        time_steps=t_grid.tolist(),
        probabilities=probs.tolist(),
        expected_timing=expected_t,
    )


# ---------------------------------------------------------------------------
# Public: timing-adjusted pricing
# ---------------------------------------------------------------------------


def price_with_timing_map(
    option: ExitOption,
    timing: TimingDistribution,
    scenario_probs: dict[str, float] | None = None,
) -> float:
    """Compute EV using the full timing distribution rather than a point estimate.

    Model:
        EV_timing = Σ_k P(T=t_k) · Σ_s p_s · payoff_s / (1+r)^{t_k}

    Parameters
    ----------
    option:
        Exit option providing values, floor, and discount rate.
    timing:
        Discretised timing distribution built by build_timing_map.
    scenario_probs:
        Optional scenario probability dict.  Defaults to uniform.

    Returns
    -------
    float
        Timing-distribution-weighted expected present value.
    """
    names = list(option.value_by_scenario.keys())
    probs = scenario_probs if scenario_probs is not None else _uniform_probs(names)

    # Pre-compute payoffs (independent of timing step).
    payoffs = {name: _payoff(v, option.floor_value) for name, v in option.value_by_scenario.items()}

    # Expected payoff across scenarios (probability-weighted, before discounting).
    expected_payoff = sum(probs[name] * payoffs[name] for name in names)

    # Discount expected_payoff at each timing step and weight by P(T=t_k).
    ev = 0.0
    for t_k, p_k in zip(timing.time_steps, timing.probabilities):
        discount_factor = (1.0 + option.discount_rate) ** t_k
        ev += p_k * expected_payoff / discount_factor

    return ev


# ---------------------------------------------------------------------------
# Public: sensitivity of timing-adjusted EV
# ---------------------------------------------------------------------------


def timing_sensitivity(
    option: ExitOption,
    n_steps: int = 40,
    scenario_probs: dict[str, float] | None = None,
) -> dict[str, float]:
    """Central-difference sensitivity of EV_timing to discount_rate and timing mode.

    Parameters
    ----------
    option:
        Base exit option.
    n_steps:
        Discretisation steps for the timing map.
    scenario_probs:
        Scenario probability dict.  Defaults to uniform.

    Returns
    -------
    dict with keys:
        "discount_rate"    : ∂EV_timing/∂r
        "timing_expected"  : ∂EV_timing/∂t_mode  (only when earliest < latest)
    """

    def _ev_at(rate: float, t_mode: float) -> float:
        perturbed = ExitOption(
            name=option.name,
            exit_type=option.exit_type,
            timing_earliest=option.timing_earliest,
            timing_expected=max(t_mode, option.timing_earliest),
            timing_latest=max(option.timing_latest, t_mode),
            value_by_scenario=option.value_by_scenario,
            floor_value=option.floor_value,
            discount_rate=rate,
        )
        t_map = build_timing_map(perturbed, n_steps=n_steps)
        return price_with_timing_map(perturbed, t_map, scenario_probs)

    result: dict[str, float] = {}

    base_rate = option.discount_rate
    delta_r = max(abs(base_rate) * DELTA_REL, DELTA_REL)
    result["discount_rate"] = (
        _ev_at(base_rate + delta_r, option.timing_expected)
        - _ev_at(base_rate - delta_r, option.timing_expected)
    ) / (2.0 * delta_r)

    # Timing mode sensitivity only when the distribution is non-degenerate.
    if not np.isclose(option.timing_earliest, option.timing_latest):
        base_t = option.timing_expected
        delta_t = max(abs(base_t) * DELTA_REL, DELTA_REL)
        result["timing_expected"] = (
            _ev_at(base_rate, base_t + delta_t) - _ev_at(base_rate, base_t - delta_t)
        ) / (2.0 * delta_t)

    return result


# ---------------------------------------------------------------------------
# Public: compare multiple options
# ---------------------------------------------------------------------------


def compare_exit_options(
    options: list[ExitOption],
    n_steps: int = 40,
    scenario_probs: dict[str, float] | None = None,
) -> list[TimingDistribution]:
    """Build a timing distribution for each option (sorted by expected_timing).

    Returns
    -------
    list[TimingDistribution]
        Sorted by expected_timing ascending (earlier exits first).
    """
    maps = [build_timing_map(opt, n_steps=n_steps) for opt in options]
    return sorted(maps, key=lambda m: m.expected_timing)
