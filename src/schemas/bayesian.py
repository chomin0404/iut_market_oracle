"""T200 Bayesian Engine schemas."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator


class ClaimTag(str, Enum):
    """Epistemic status of a research claim."""

    PROVEN = "proven"
    HEURISTIC = "heuristic"
    EMPIRICAL = "empirical"
    TODO = "todo"


class EvidenceKind(str, Enum):
    """Kind of evidence submitted to the Bayesian engine."""

    OBSERVATION = "observation"
    EXPERT_PRIOR = "expert_prior"
    MARKET_DATA = "market_data"
    BACKTEST = "backtest"


class Evidence(BaseModel):
    """Single piece of evidence for Bayesian updating."""

    source: str = Field(..., min_length=1, description="Origin of the evidence")
    kind: EvidenceKind
    value: float = Field(..., description="Point estimate or likelihood ratio")
    weight: float = Field(default=1.0, gt=0.0, description="Relative credibility weight")
    tag: ClaimTag = ClaimTag.EMPIRICAL
    notes: str = ""

    @field_validator("value")
    @classmethod
    def finite_value(cls, v: float) -> float:
        import math

        if not math.isfinite(v):
            raise ValueError("value must be finite")
        return v


class PriorSpec(BaseModel):
    """Specification of a prior distribution."""

    distribution: str = Field(..., description="e.g. 'beta', 'normal', 'uniform'")
    params: dict[str, int | float] = Field(..., description="Distribution parameters")
    description: str = ""

    @field_validator("params")
    @classmethod
    def non_empty_params(cls, v: dict[str, int | float]) -> dict[str, float]:
        if not v:
            raise ValueError("params must not be empty")
        return {k: float(val) for k, val in v.items()}


class PosteriorSummary(BaseModel):
    """Output of one Bayesian update cycle."""

    mean: float
    variance: float = Field(..., ge=0.0)
    credible_interval_95: tuple[float, float]
    n_evidence: int = Field(..., ge=0)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def interval_ordered(self) -> PosteriorSummary:
        lo, hi = self.credible_interval_95
        if lo > hi:
            raise ValueError("credible_interval_95 lower bound must be <= upper bound")
        return self
