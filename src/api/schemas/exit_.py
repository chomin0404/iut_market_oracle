"""Request/response schemas for the exit strategy router."""

from __future__ import annotations

from pydantic import BaseModel, Field

from schemas import ExitOption, TimingDistribution


class PriceRequest(BaseModel):
    option: ExitOption
    scenario_probs: dict[str, float] | None = None


class PriceAllRequest(BaseModel):
    options: list[ExitOption]
    scenario_probs: dict[str, float] | None = None


class TimingMapRequest(BaseModel):
    option: ExitOption
    n_steps: int = Field(default=40, ge=2)


class PriceWithTimingRequest(BaseModel):
    option: ExitOption
    timing: TimingDistribution
    scenario_probs: dict[str, float] | None = None


class PriceWithTimingResponse(BaseModel):
    expected_value: float


class CompareRequest(BaseModel):
    options: list[ExitOption]
    n_steps: int = Field(default=40, ge=2)
    scenario_probs: dict[str, float] | None = None
