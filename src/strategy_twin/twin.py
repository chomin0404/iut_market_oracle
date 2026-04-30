"""Strategy Twin — decision engine for new-business viability (T1700).

Orchestrates:
  1. SOTP valuation  (moat-adjusted DCF per business unit)
  2. Black-Litterman posterior returns (market + investor views)
  3. Reverse DCF viability conditions (g*, s*, m*)
  4. Linear causal ATE from macro → FCF

Usage::

    config = StrategyTwinConfig(
        business_units=[...],
        macro=MacroEnvironment(...),
        views=[...],
        causal_edges=[...],
        target_ev=500_000,
    )
    twin = StrategyTwin(config)
    report = twin.run()
    print(report.verdict)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from schemas import (
    BLView,
    BusinessUnit,
    CausalEdge,
    MacroEnvironment,
    StrategyTwinReport,
)
from strategy_twin.black_litterman import black_litterman
from strategy_twin.causal import compute_all_effects
from strategy_twin.sotp import sotp_valuation
from strategy_twin.viability import compute_viability_conditions

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_CAUSAL_MACRO_NODES: list[str] = ["gdp_growth", "inflation", "risk_free_rate"]
_CAUSAL_FCF_NODE: str = "fcf"


@dataclass
class StrategyTwinConfig:
    """Configuration for the Strategy Twin.

    Attributes:
        business_units:       Business units to value (SOTP).
        macro:                Macro environment (TAM, rates, GDP).
        views:                Black-Litterman investor views.
        causal_edges:         DAG edges for causal inference.
        target_ev:            Target enterprise value for viability analysis.
        net_debt:             Net debt subtracted from SOTP total EV.
        current_market_share: Current market share for viability (∈ [0,1]).
        current_fcf_margin:   Current FCF margin for viability (∈ [0,1]).
        causal_causes:        Variable names used as causes in ATE analysis.
        causal_effects:       Variable names used as effects in ATE analysis.
    """

    business_units: list[BusinessUnit]
    macro: MacroEnvironment
    views: list[BLView] = field(default_factory=list)
    causal_edges: list[CausalEdge] = field(default_factory=list)
    target_ev: float = 0.0
    net_debt: float = 0.0
    current_market_share: float = 0.05
    current_fcf_margin: float = 0.15
    causal_causes: list[str] = field(default_factory=lambda: list(_CAUSAL_MACRO_NODES))
    causal_effects: list[str] = field(default_factory=lambda: [_CAUSAL_FCF_NODE])

    def __post_init__(self) -> None:
        if not self.business_units:
            raise ValueError("business_units must not be empty")
        if self.target_ev < 0:
            raise ValueError("target_ev must be non-negative")


# ---------------------------------------------------------------------------
# Strategy Twin
# ---------------------------------------------------------------------------


class StrategyTwin:
    """Decision engine for new-business viability.

    Args:
        config: StrategyTwinConfig with all inputs.
    """

    def __init__(self, config: StrategyTwinConfig) -> None:
        self._config = config

    def run(self) -> StrategyTwinReport:
        """Execute all analyses and return consolidated report.

        Steps:
            1. SOTP — moat-adjusted DCF per unit → total EV
            2. Black-Litterman — posterior returns from views
            3. Viability — reverse DCF conditions on primary unit
            4. Causal — ATE from macro nodes to FCF
            5. Verdict — plain-language summary
        """
        cfg = self._config

        # 1. SOTP
        segments, total_ev = sotp_valuation(cfg.business_units, net_debt=cfg.net_debt)

        # 2. Black-Litterman
        bl_result = black_litterman(cfg.business_units, cfg.views)

        # 3. Viability conditions (primary unit = first)
        primary = cfg.business_units[0]
        target = cfg.target_ev if cfg.target_ev > 0 else total_ev
        viability_conditions, implied_g = compute_viability_conditions(
            primary_unit=primary,
            macro=cfg.macro,
            target_ev=target,
            current_market_share=cfg.current_market_share,
            current_fcf_margin=cfg.current_fcf_margin,
        )

        # 4. Causal ATE
        causal_effects = compute_all_effects(
            cfg.causal_edges,
            causes=cfg.causal_causes,
            effects=cfg.causal_effects,
        )

        # 5. Verdict
        n_met = sum(1 for c in viability_conditions if c.is_met)
        n_total = len(viability_conditions)
        verdict = _build_verdict(n_met, n_total, total_ev, target, implied_g)

        return StrategyTwinReport(
            sotp_segments=segments,
            sotp_total_ev=total_ev,
            bl_result=bl_result,
            viability_conditions=viability_conditions,
            implied_growth_rate=implied_g,
            causal_effects=causal_effects,
            macro=cfg.macro,
            verdict=verdict,
        )


# ---------------------------------------------------------------------------
# Verdict builder
# ---------------------------------------------------------------------------


def _build_verdict(
    n_met: int,
    n_total: int,
    sotp_ev: float,
    target_ev: float,
    implied_g: float | None,
) -> str:
    """Build a concise plain-language verdict string."""
    parts: list[str] = []

    ratio = sotp_ev / target_ev if target_ev > 0 else float("nan")
    if ratio >= 1.0:
        parts.append(f"SOTP EV ({sotp_ev:,.0f}) meets target ({target_ev:,.0f}): PASS")
    else:
        parts.append(
            f"SOTP EV ({sotp_ev:,.0f}) below target ({target_ev:,.0f}): "
            f"gap {target_ev - sotp_ev:,.0f} ({1 - ratio:.1%})"
        )

    if implied_g is not None:
        parts.append(f"Implied growth rate: {implied_g:.2%}")
    else:
        parts.append("Implied growth rate: infeasible")

    parts.append(f"Viability conditions met: {n_met}/{n_total}")

    if n_met == n_total:
        parts.append("VERDICT: VIABLE")
    elif n_met >= n_total / 2:
        parts.append("VERDICT: CONDITIONALLY VIABLE — review unmet conditions")
    else:
        parts.append("VERDICT: NOT VIABLE — key conditions unmet")

    return " | ".join(parts)
