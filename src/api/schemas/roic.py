"""API DTO schemas for the ROIC evaluation endpoint."""

from __future__ import annotations

from pydantic import BaseModel, Field

from schemas.roic import ROICState

# ---------------------------------------------------------------------------
# Request / Response
# ---------------------------------------------------------------------------


class ROICEvaluateRequest(BaseModel):
    """Evidence for a single ROIC inference query."""

    industry_competitiveness: ROICState | None = Field(
        default=None, description="AUM scale / entry barriers"
    )
    management_efficiency: ROICState | None = Field(
        default=None, description="FRE margin, fee-earnings CAGR"
    )
    macro_environment: ROICState | None = Field(
        default=None, description="Interest-rate regime, deal-exit conditions"
    )
    pricing_power: ROICState | None = Field(
        default=None, description="Ability to maintain fee structures"
    )
    capital_allocation: ROICState | None = Field(
        default=None, description="Historical IRR, counter-cyclical investment track record"
    )
    equivalent_sample_size: float = Field(
        default=10.0, gt=0.0, description="Dirichlet prior strength (larger = more inertia)"
    )


class ROICEvaluateResponse(BaseModel):
    """Posterior distribution and grade for roic_level."""

    evidence: dict[str, str]
    roic_posterior: dict[str, float]
    expected_roic_score: float
    dominant_state: ROICState
    grade: str
