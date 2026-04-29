"""Exit Strategy endpoints: option pricing and timing distribution."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from exit.option_pricer import price_all_options, price_option
from exit.timing_map import build_timing_map, compare_exit_options, price_with_timing_map
from schemas import ExitOption, ExitValueSummary, TimingDistribution

router = APIRouter()


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/price", response_model=ExitValueSummary)
def price(req: PriceRequest) -> ExitValueSummary:
    """Price a single exit option (option-style payoff + sensitivity)."""
    try:
        return price_option(req.option, req.scenario_probs)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/price-all", response_model=list[ExitValueSummary])
def price_all(req: PriceAllRequest) -> list[ExitValueSummary]:
    """Price all exit options sorted by expected value descending."""
    try:
        return price_all_options(req.options, req.scenario_probs)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/timing-map", response_model=TimingDistribution)
def timing_map(req: TimingMapRequest) -> TimingDistribution:
    """Discretise the triangular exit-timing distribution into a probability map."""
    try:
        return build_timing_map(req.option, n_steps=req.n_steps)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/price-with-timing", response_model=PriceWithTimingResponse)
def price_timing(req: PriceWithTimingRequest) -> PriceWithTimingResponse:
    """Compute EV using the full timing distribution rather than a point estimate."""
    try:
        ev = price_with_timing_map(req.option, req.timing, req.scenario_probs)
        return PriceWithTimingResponse(expected_value=ev)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/compare", response_model=list[TimingDistribution])
def compare(req: CompareRequest) -> list[TimingDistribution]:
    """Build timing distributions for multiple options, sorted by expected_timing."""
    try:
        return compare_exit_options(
            req.options, n_steps=req.n_steps, scenario_probs=req.scenario_probs
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
