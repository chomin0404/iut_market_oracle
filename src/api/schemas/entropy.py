"""Request/response schemas for the entropy monitoring router."""

from __future__ import annotations

from pydantic import BaseModel, Field

from schemas import PosteriorSummary, PriorSpec


class EntropyRequest(BaseModel):
    posterior: PosteriorSummary
    prior: PriorSpec


class EntropyResponse(BaseModel):
    entropy: float


class KLRequest(BaseModel):
    posterior: PosteriorSummary
    prior: PriorSpec


class KLResponse(BaseModel):
    kl_divergence: float


class DetectRequest(BaseModel):
    posteriors: list[PosteriorSummary] = Field(..., min_length=1)
    prior: PriorSpec
    experiment_id: str
    kl_threshold: float = Field(default=0.5, gt=0.0)
    entropy_gradient_threshold: float = Field(default=0.1, gt=0.0)
    rolling_window: int = Field(default=3, ge=1)
