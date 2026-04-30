"""Viability condition analysis via Reverse DCF (T1700).

For each viability metric the engine solves:

    Implied growth rate (g*):
        The growth rate g* such that DCF(FCF, g*, wacc, g_T, T) = target_ev
        Solved via bisection (calls reverse_dcf_implied_growth).
        Returns None if bisection cannot bracket [low, high].

    Implied market share (s*):
        s* = required_fcf / (TAM · margin)
        where required_fcf = FCF needed to support target_ev at WACC.

    Implied FCF margin (m*):
        m* = required_fcf / (TAM · market_share)

All conditions report:
    gap = current_estimate − minimum_required   (≥ 0 means condition met)
    is_met = gap ≥ 0
"""

from __future__ import annotations

from schemas import BusinessUnit, MacroEnvironment, ViabilityCondition
from strategy_twin.moat import (
    composite_moat_score,
    moat_adjusted_terminal_growth,
    moat_adjusted_wacc,
)
from valuation.dcf import DCFInputs, dcf_valuation, reverse_dcf_implied_growth

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BISECT_LOW: float = -0.50
_BISECT_HIGH: float = 0.80
_MIN_FCF_RATIO: float = 0.01  # floor for required_fcf ratio


def _required_fcf_for_ev(
    target_ev: float,
    adj_wacc: float,
    adj_g_t: float,
    growth_rate: float,
    forecast_years: int,
) -> float:
    """Back-calculate the initial FCF needed to produce target_ev at given WACC.

    Uses a unit FCF DCF to find the multiplier:
        target_ev = multiplier · EV(FCF=1, ...)
        → FCF* = target_ev / multiplier
    """
    unit_result = dcf_valuation(
        DCFInputs(
            initial_fcf=1.0,
            growth_rate=growth_rate,
            discount_rate=adj_wacc,
            forecast_years=forecast_years,
            terminal_growth_rate=adj_g_t,
        )
    )
    multiplier = unit_result.enterprise_value
    if multiplier <= 0:
        return float("inf")
    return target_ev / multiplier


def compute_viability_conditions(
    primary_unit: BusinessUnit,
    macro: MacroEnvironment,
    target_ev: float,
    current_market_share: float = 0.05,
    current_fcf_margin: float = 0.15,
) -> tuple[list[ViabilityCondition], float | None]:
    """Compute viability conditions for primary business unit.

    Args:
        primary_unit:         Primary business unit (used for FCF, WACC, moat).
        macro:                Macro environment (TAM, risk-free rate, etc.).
        target_ev:            Target enterprise value (e.g. investment hurdle).
        current_market_share: Current or projected market share ∈ [0, 1].
        current_fcf_margin:   Current or projected FCF margin ∈ [0, 1].

    Returns:
        (conditions, implied_growth_rate)
        implied_growth_rate is None if bisection infeasible.
    """
    m = composite_moat_score(primary_unit.moat_scores)
    adj_wacc = moat_adjusted_wacc(primary_unit.discount_rate, m)
    adj_g_t = moat_adjusted_terminal_growth(
        primary_unit.terminal_growth_rate, adj_wacc, m
    )

    conditions: list[ViabilityCondition] = []

    # ── 1. Implied growth rate (g*) ──────────────────────────────────────
    implied_g: float | None = None
    try:
        implied_g = reverse_dcf_implied_growth(
            target_enterprise_value=target_ev,
            initial_fcf=primary_unit.initial_fcf,
            discount_rate=adj_wacc,
            forecast_years=primary_unit.forecast_years,
            terminal_growth_rate=adj_g_t,
            low=_BISECT_LOW,
            high=_BISECT_HIGH,
        )
    except ValueError:
        implied_g = None

    g_min = implied_g if implied_g is not None else float("nan")
    conditions.append(
        ViabilityCondition(
            metric="growth_rate",
            minimum_required=g_min,
            current_estimate=primary_unit.growth_rate,
            gap=primary_unit.growth_rate - g_min,
            is_met=(primary_unit.growth_rate >= g_min) if implied_g is not None else False,
            explanation=(
                f"Implied growth rate to reach EV={target_ev:.0f} at WACC={adj_wacc:.1%}"
                if implied_g is not None
                else "Bisection infeasible: target EV outside achievable range"
            ),
        )
    )

    # ── 2. Required FCF → implied market share ────────────────────────────
    required_fcf = _required_fcf_for_ev(
        target_ev, adj_wacc, adj_g_t, primary_unit.growth_rate, primary_unit.forecast_years
    )

    if current_fcf_margin > 0 and macro.tam > 0:
        implied_share = required_fcf / (macro.tam * current_fcf_margin)
    else:
        implied_share = float("inf")

    conditions.append(
        ViabilityCondition(
            metric="market_share",
            minimum_required=implied_share,
            current_estimate=current_market_share,
            gap=current_market_share - implied_share,
            is_met=current_market_share >= implied_share,
            explanation=(
                f"Market share needed so FCF≥{required_fcf:.0f} "
                f"at margin={current_fcf_margin:.1%}, TAM={macro.tam:.0f}"
            ),
        )
    )

    # ── 3. Required FCF → implied FCF margin ─────────────────────────────
    if current_market_share > 0 and macro.tam > 0:
        implied_margin = required_fcf / (macro.tam * current_market_share)
    else:
        implied_margin = float("inf")

    conditions.append(
        ViabilityCondition(
            metric="fcf_margin",
            minimum_required=implied_margin,
            current_estimate=current_fcf_margin,
            gap=current_fcf_margin - implied_margin,
            is_met=current_fcf_margin >= implied_margin,
            explanation=(
                f"FCF margin needed so FCF≥{required_fcf:.0f} "
                f"at share={current_market_share:.1%}, TAM={macro.tam:.0f}"
            ),
        )
    )

    return conditions, implied_g
