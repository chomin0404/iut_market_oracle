"""Exit Strategy endpoints: option pricing and timing distribution."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Body, HTTPException

from api.schemas.exit_ import (
    CompareRequest,
    PriceAllRequest,
    PriceRequest,
    PriceWithTimingRequest,
    PriceWithTimingResponse,
    TimingMapRequest,
)
from exit.option_pricer import price_all_options, price_option
from exit.timing_map import build_timing_map, compare_exit_options, price_with_timing_map
from schemas import ExitValueSummary, TimingDistribution

router = APIRouter()

_IPO_OPTION = {
    "name": "IPO exit",
    "exit_type": "ipo",
    "timing_earliest": 1.0,
    "timing_expected": 3.0,
    "timing_latest": 5.0,
    "value_by_scenario": {"bear": 80.0, "base": 150.0, "bull": 280.0},
    "floor_value": 0.0,
    "discount_rate": 0.10,
}
_MA_OPTION = {
    "name": "M&A exit",
    "exit_type": "ma",
    "timing_earliest": 0.5,
    "timing_expected": 2.0,
    "timing_latest": 4.0,
    "value_by_scenario": {"bear": 100.0, "base": 200.0, "bull": 350.0},
    "floor_value": 50.0,
    "discount_rate": 0.08,
}
_SCENARIO_PROBS = {"bear": 0.2, "base": 0.5, "bull": 0.3}

_PRICE_EXAMPLES = {
    "ipo_no_probs": {
        "summary": "IPO exit (equal scenario weights)",
        "value": {"option": _IPO_OPTION},
    },
    "ma_with_probs": {
        "summary": "M&A exit with scenario probabilities",
        "value": {"option": _MA_OPTION, "scenario_probs": _SCENARIO_PROBS},
    },
}
_PRICE_ALL_EXAMPLES = {
    "two_options": {
        "summary": "Compare IPO vs M&A",
        "value": {"options": [_IPO_OPTION, _MA_OPTION], "scenario_probs": _SCENARIO_PROBS},
    },
}
_TIMING_EXAMPLES = {
    "default": {
        "summary": "IPO timing map (40 steps)",
        "value": {"option": _IPO_OPTION},
    },
    "fine": {
        "summary": "IPO timing map (100 steps)",
        "value": {"option": _IPO_OPTION, "n_steps": 100},
    },
}
_COMPARE_EXAMPLES = {
    "two_options": {
        "summary": "Compare timing distributions for IPO vs M&A",
        "value": {"options": [_IPO_OPTION, _MA_OPTION]},
    },
}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/price", response_model=ExitValueSummary)
def price(
    req: Annotated[PriceRequest, Body(openapi_examples=_PRICE_EXAMPLES)],
) -> ExitValueSummary:
    """Price a single exit option (option-style payoff + sensitivity)."""
    try:
        return price_option(req.option, req.scenario_probs)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/price-all", response_model=list[ExitValueSummary])
def price_all(
    req: Annotated[PriceAllRequest, Body(openapi_examples=_PRICE_ALL_EXAMPLES)],
) -> list[ExitValueSummary]:
    """Price all exit options sorted by expected value descending."""
    try:
        return price_all_options(req.options, req.scenario_probs)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/timing-map", response_model=TimingDistribution)
def timing_map(
    req: Annotated[TimingMapRequest, Body(openapi_examples=_TIMING_EXAMPLES)],
) -> TimingDistribution:
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
def compare(
    req: Annotated[CompareRequest, Body(openapi_examples=_COMPARE_EXAMPLES)],
) -> list[TimingDistribution]:
    """Build timing distributions for multiple options, sorted by expected_timing."""
    try:
        return compare_exit_options(
            req.options, n_steps=req.n_steps, scenario_probs=req.scenario_probs
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
