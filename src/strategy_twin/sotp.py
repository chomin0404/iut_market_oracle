"""Sum-of-the-Parts (SOTP) valuation engine (T1700).

For each business unit k:
    1. Compute composite moat score: m_k = Σ w_i s_i / Σ w_i
    2. Adjust WACC:            wacc_k = max(wacc − m_k·0.025, 0.04)
    3. Adjust terminal growth: g_T_k  = min(g_T + m_k·0.015, wacc_k − 0.005)
    4. Run DCF:                EV_k   = dcf_valuation(FCF_k, g_k, wacc_k, g_T_k, T)

SOTP:
    EV_total = Σ_k EV_k − net_debt
"""

from __future__ import annotations

from schemas import BusinessUnit, SOTPSegment
from strategy_twin.moat import (
    composite_moat_score,
    moat_adjusted_terminal_growth,
    moat_adjusted_wacc,
)
from valuation.dcf import DCFInputs, dcf_valuation


def value_unit(unit: BusinessUnit) -> SOTPSegment:
    """Moat-adjusted DCF for a single business unit.

    Args:
        unit: BusinessUnit with FCF, growth, WACC, moat scores, etc.

    Returns:
        SOTPSegment with enterprise value and adjusted parameters.
    """
    m = composite_moat_score(unit.moat_scores)
    adj_wacc = moat_adjusted_wacc(unit.discount_rate, m)
    adj_g_t = moat_adjusted_terminal_growth(unit.terminal_growth_rate, adj_wacc, m)

    inputs = DCFInputs(
        initial_fcf=unit.initial_fcf,
        growth_rate=unit.growth_rate,
        discount_rate=adj_wacc,
        forecast_years=unit.forecast_years,
        terminal_growth_rate=adj_g_t,
    )
    result = dcf_valuation(inputs)

    return SOTPSegment(
        unit_name=unit.name,
        enterprise_value=result.enterprise_value,
        moat_adjusted_wacc=adj_wacc,
        moat_adjusted_terminal_growth=adj_g_t,
        moat_composite_score=m,
    )


def sotp_valuation(
    units: list[BusinessUnit],
    net_debt: float = 0.0,
) -> tuple[list[SOTPSegment], float]:
    """SOTP: value each unit and aggregate.

    EV_total = Σ_k EV_k − net_debt

    Args:
        units:    List of business units to value independently.
        net_debt: Net debt to subtract from sum of parts (default 0).

    Returns:
        (segments, total_ev) where total_ev = Σ EV_k − net_debt.
    """
    segments = [value_unit(u) for u in units]
    total_ev = sum(s.enterprise_value for s in segments) - net_debt
    return segments, total_ev
