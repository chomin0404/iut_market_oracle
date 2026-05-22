"""Request schemas for the Strategy Twin router (T1700)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from schemas import BLView, BusinessUnit, CausalEdge, MacroEnvironment


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
