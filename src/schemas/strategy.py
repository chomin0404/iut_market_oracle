"""T1700 Strategy Twin schemas."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field, model_validator


class MoatDimension(str, Enum):
    """Five canonical sources of competitive moat (Morningstar framework)."""

    COST_ADVANTAGE = "cost_advantage"
    SWITCHING_COSTS = "switching_costs"
    NETWORK_EFFECTS = "network_effects"
    INTANGIBLE_ASSETS = "intangible_assets"
    EFFICIENT_SCALE = "efficient_scale"


class MoatScore(BaseModel):
    """Strength assessment for one moat dimension.

    score: 0 = no moat, 0.5 = narrow moat, 1.0 = wide moat.
    weight: relative importance of this dimension (default 1.0).
    """

    dimension: MoatDimension
    score: float = Field(..., ge=0.0, le=1.0, description="Moat strength [0=none, 1=wide]")
    weight: float = Field(default=1.0, gt=0.0, description="Relative importance weight")


class BusinessUnit(BaseModel):
    """Single business segment for SOTP valuation.

    initial_fcf:           FCF₀ in consistent units (e.g. JPY millions).
    growth_rate:           CAGR estimate (decimal; may be negative).
    discount_rate:         Base WACC before moat adjustment (decimal).
    terminal_growth_rate:  Gordon long-run growth (decimal; must be < discount_rate).
    forecast_years:        Explicit DCF horizon.
    moat_scores:           Moat dimension assessments (empty = no adjustment).
    """

    name: str = Field(..., min_length=1)
    initial_fcf: float = Field(..., gt=0.0, description="Initial free cash flow (FCF₀)")
    growth_rate: float = Field(..., description="Revenue/FCF CAGR estimate (decimal)")
    discount_rate: float = Field(..., gt=0.0, description="Base WACC (decimal)")
    terminal_growth_rate: float = Field(
        default=0.03, description="Long-run Gordon growth rate (decimal)"
    )
    forecast_years: int = Field(default=5, ge=1)
    moat_scores: list[MoatScore] = Field(default_factory=list)


class MacroEnvironment(BaseModel):
    """Macro-economic context for Strategy Twin.

    tam: Total Addressable Market in the same unit as BusinessUnit.initial_fcf.
    """

    gdp_growth: float = Field(..., description="Real GDP growth rate (decimal, e.g. 0.02)")
    risk_free_rate: float = Field(..., ge=0.0, description="Risk-free rate (decimal)")
    market_risk_premium: float = Field(..., gt=0.0, description="Equity risk premium (decimal)")
    inflation: float = Field(..., ge=0.0, description="CPI inflation rate (decimal)")
    tam: float = Field(..., gt=0.0, description="Total Addressable Market (same unit as FCF)")


class BLView(BaseModel):
    """Single investor view for Black-Litterman (He-Litterman 1999).

    assets:          asset_name → pick weight
                     Absolute view: {A: 1.0}
                     Relative view: {A: 1.0, B: -1.0}  (long A, short B)
    expected_return: q_k — view's expected excess return (decimal).
    uncertainty:     Ω_kk — view variance (larger = less confident).
    """

    assets: dict[str, float] = Field(..., min_length=1)
    expected_return: float = Field(..., description="View's expected excess return (decimal)")
    uncertainty: float = Field(..., gt=0.0, description="View variance Ω_kk")


class CausalEdge(BaseModel):
    """Directed edge in a linear Structural Causal Model.

    coefficient: b_{effect ← cause}  — linear regression coefficient.
    """

    cause: str = Field(..., min_length=1)
    effect: str = Field(..., min_length=1)
    coefficient: float = Field(..., description="b_{effect ← cause}")

    @model_validator(mode="after")
    def no_self_loop(self) -> CausalEdge:
        if self.cause == self.effect:
            raise ValueError("Self-loops are not allowed (cause == effect)")
        return self


class ViabilityCondition(BaseModel):
    """One measurable condition the business must satisfy to reach its target EV.

    gap = current_estimate − minimum_required.
    is_met = True iff gap >= 0.
    """

    metric: str = Field(..., min_length=1, description="e.g. 'revenue_growth_rate'")
    minimum_required: float = Field(..., description="Minimum value for this condition")
    current_estimate: float = Field(..., description="Current estimated value")
    gap: float = Field(..., description="current_estimate − minimum_required (≥0 means met)")
    is_met: bool
    explanation: str = ""


class SOTPSegment(BaseModel):
    """SOTP valuation result for one business unit."""

    unit_name: str = Field(..., min_length=1)
    enterprise_value: float
    moat_adjusted_wacc: float = Field(..., description="WACC after moat reduction")
    moat_adjusted_terminal_growth: float = Field(..., description="g_T after moat uplift")
    moat_composite_score: float = Field(..., ge=0.0, le=1.0)


class BLResult(BaseModel):
    """Black-Litterman posterior return estimates per asset."""

    equilibrium_returns: dict[str, float] = Field(
        ..., description="Π = δ·Σ·w_mkt (CAPM equilibrium)"
    )
    posterior_returns: dict[str, float] = Field(
        ..., description="μ_BL: posterior mean after blending views"
    )
    posterior_std: dict[str, float] = Field(
        ..., description="sqrt(diag(M⁻¹)): posterior standard deviation"
    )
    market_weights: dict[str, float] = Field(..., description="Normalised FCF market weights")


class CausalEffect(BaseModel):
    """Total linear causal effect via all directed paths (ATE).

    total_effect = Σ_{paths p: cause→effect} Π_{(a→b)∈p} b_{b←a}
    """

    cause: str
    effect: str
    total_effect: float = Field(..., description="ATE = sum of path products")
    n_paths: int = Field(..., ge=0, description="Number of directed paths found")


class StrategyTwinReport(BaseModel):
    """Full output of one Strategy Twin run (T1700).

    sotp_segments:        Per-unit SOTP breakdown with moat adjustments.
    sotp_total_ev:        Sum(segment EVs) − net_debt.
    bl_result:            Black-Litterman posterior returns per segment.
    viability_conditions: Minimum conditions to reach the target EV.
    implied_growth_rate:  g* from reverse DCF (NaN serialised as null if infeasible).
    causal_effects:       All pairwise ATE in the causal graph (sorted by |ATE| desc).
    macro:                Macro snapshot used for this run.
    verdict:              One-line executive decision summary (Japanese).
    """

    sotp_segments: list[SOTPSegment]
    sotp_total_ev: float
    bl_result: BLResult
    viability_conditions: list[ViabilityCondition]
    implied_growth_rate: float | None = Field(
        None, description="g* from reverse DCF; None if infeasible"
    )
    causal_effects: list[CausalEffect]
    macro: MacroEnvironment
    verdict: str
    produced_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
