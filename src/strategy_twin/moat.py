"""Moat scoring and WACC/growth rate adjustments (T1700).

Composite moat score:
    m = Σ_i w_i·s_i / Σ_i w_i  ∈ [0, 1]

WACC adjustment (up to 250 bps reduction, 4% floor):
    wacc_adj = max(wacc_base − m · _MAX_WACC_REDUCTION, _MIN_WACC_FLOOR)

Terminal growth adjustment (up to 150 bps uplift, spread guard):
    g_T_adj = min(g_T_base + m · _MAX_GROWTH_UPLIFT, wacc_adj − _MIN_SPREAD)
"""

from __future__ import annotations

from schemas import MoatScore

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_WACC_REDUCTION: float = 0.025  # 250 bps
_MAX_GROWTH_UPLIFT: float = 0.015  # 150 bps
_MIN_WACC_FLOOR: float = 0.04  # 4%
_MIN_SPREAD: float = 0.005  # 50 bps spread guard


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def composite_moat_score(moat_scores: list[MoatScore]) -> float:
    """Weighted composite moat score ∈ [0, 1].

    m = Σ_i w_i·s_i / Σ_i w_i

    Returns 0.0 if moat_scores is empty.
    """
    if not moat_scores:
        return 0.0
    numerator = sum(ms.weight * ms.score for ms in moat_scores)
    denominator = sum(ms.weight for ms in moat_scores)
    return float(numerator / denominator)


def moat_adjusted_wacc(base_wacc: float, moat_score: float) -> float:
    """Reduce WACC by moat strength (capped at _MAX_WACC_REDUCTION).

    wacc_adj = max(base_wacc − moat_score · _MAX_WACC_REDUCTION, _MIN_WACC_FLOOR)

    Args:
        base_wacc:   WACC before moat adjustment (e.g. 0.10 for 10%)
        moat_score:  Composite moat score ∈ [0, 1]

    Returns:
        Adjusted WACC ≥ _MIN_WACC_FLOOR
    """
    return max(base_wacc - moat_score * _MAX_WACC_REDUCTION, _MIN_WACC_FLOOR)


def moat_adjusted_terminal_growth(
    base_g: float,
    adj_wacc: float,
    moat_score: float,
) -> float:
    """Uplift terminal growth by moat strength (capped by spread guard).

    g_T_adj = min(base_g + moat_score · _MAX_GROWTH_UPLIFT, adj_wacc − _MIN_SPREAD)

    Args:
        base_g:      Base terminal growth rate (e.g. 0.03 for 3%)
        adj_wacc:    Already-adjusted WACC (ensures spread guard is consistent)
        moat_score:  Composite moat score ∈ [0, 1]

    Returns:
        Adjusted terminal growth rate < adj_wacc
    """
    return min(base_g + moat_score * _MAX_GROWTH_UPLIFT, adj_wacc - _MIN_SPREAD)
