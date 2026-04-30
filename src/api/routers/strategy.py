"""FastAPI router for Strategy Twin (T1700).

Endpoint:
    POST /strategy/run  →  StrategyTwinReport
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from schemas import (
    BLView,
    BusinessUnit,
    CausalEdge,
    MacroEnvironment,
    StrategyTwinReport,
)
from strategy_twin.twin import StrategyTwin, StrategyTwinConfig

router = APIRouter()


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------


class StrategyRunRequest(BaseModel):
    """Request body for POST /strategy/run."""

    business_units: list[BusinessUnit] = Field(
        ...,
        min_length=1,
        description="Business units to value via moat-adjusted SOTP DCF",
    )
    macro: MacroEnvironment = Field(..., description="Macro environment parameters")
    views: list[BLView] = Field(
        default_factory=list,
        description="Black-Litterman investor views",
    )
    causal_edges: list[CausalEdge] = Field(
        default_factory=list,
        description="Causal DAG edges (linear SCM)",
    )
    target_ev: float = Field(
        default=0.0,
        ge=0.0,
        description="Target enterprise value for viability analysis (0 = use SOTP total)",
    )
    net_debt: float = Field(
        default=0.0,
        description="Net debt subtracted from SOTP total EV",
    )
    current_market_share: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Current market share for viability condition",
    )
    current_fcf_margin: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Current FCF margin for viability condition",
    )
    causal_causes: list[str] = Field(
        default_factory=lambda: ["gdp_growth", "inflation", "risk_free_rate"],
        description="Cause nodes for ATE analysis",
    )
    causal_effects: list[str] = Field(
        default_factory=lambda: ["fcf"],
        description="Effect nodes for ATE analysis",
    )


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post("/run", response_model=StrategyTwinReport)
def run_strategy_twin(request: StrategyRunRequest) -> StrategyTwinReport:
    """Run the Strategy Twin decision engine.

    Returns a consolidated report including:
    - SOTP valuation (moat-adjusted DCF per business unit)
    - Black-Litterman posterior returns
    - Reverse DCF viability conditions (g*, s*, m*)
    - Causal ATE estimates from macro → FCF
    - Plain-language verdict
    """
    try:
        config = StrategyTwinConfig(
            business_units=request.business_units,
            macro=request.macro,
            views=request.views,
            causal_edges=request.causal_edges,
            target_ev=request.target_ev,
            net_debt=request.net_debt,
            current_market_share=request.current_market_share,
            current_fcf_margin=request.current_fcf_margin,
            causal_causes=request.causal_causes,
            causal_effects=request.causal_effects,
        )
        twin = StrategyTwin(config)
        return twin.run()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
