"""Valuation endpoints: DCF scenario running and reverse DCF."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Body, HTTPException

from api.schemas.valuation import (
    DCFRequest,
    DCFResponse,
    ReverseDCFRequest,
    ReverseDCFResponse,
    RunAllRequest,
)
from schemas import AssumptionSet, ScenarioResult
from valuation.dcf import DCFInputs, dcf_valuation, reverse_dcf_implied_growth
from valuation.scenario import run_all_scenarios, run_scenario

router = APIRouter()

_DCF_EXAMPLES = {
    "saas_growth": {
        "summary": "SaaS startup (high growth)",
        "value": {
            "initial_fcf": 5.0,
            "growth_rate": 0.30,
            "discount_rate": 0.12,
            "forecast_years": 5,
            "terminal_growth_rate": 0.03,
        },
    },
    "mature_company": {
        "summary": "Mature company (stable growth)",
        "value": {
            "initial_fcf": 100.0,
            "growth_rate": 0.05,
            "discount_rate": 0.08,
            "forecast_years": 10,
            "terminal_growth_rate": 0.02,
        },
    },
}
_REVERSE_DCF_EXAMPLES = {
    "default": {
        "summary": "Implied growth for given market cap",
        "value": {
            "target_enterprise_value": 500.0,
            "initial_fcf": 20.0,
            "discount_rate": 0.10,
            "forecast_years": 5,
            "terminal_growth_rate": 0.025,
        },
    },
}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/scenario", response_model=ScenarioResult)
def run_scenario_endpoint(assumption: AssumptionSet) -> ScenarioResult:
    """Run a single DCF scenario from an AssumptionSet."""
    try:
        return run_scenario(assumption)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/scenarios/run-all", response_model=list[ScenarioResult])
def run_all_endpoint(req: RunAllRequest) -> list[ScenarioResult]:
    """Run all *.yaml scenario files found in scenario_dir."""
    try:
        return run_all_scenarios(req.scenario_dir)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/dcf", response_model=DCFResponse)
def dcf_endpoint(
    req: Annotated[DCFRequest, Body(openapi_examples=_DCF_EXAMPLES)],
) -> DCFResponse:
    """Run raw DCF valuation from explicit inputs."""
    try:
        inputs = DCFInputs(
            initial_fcf=req.initial_fcf,
            growth_rate=req.growth_rate,
            discount_rate=req.discount_rate,
            forecast_years=req.forecast_years,
            terminal_growth_rate=req.terminal_growth_rate,
        )
        result = dcf_valuation(inputs)
        return DCFResponse(
            projected_fcfs=result.projected_fcfs,
            discounted_fcfs=result.discounted_fcfs,
            terminal_value=result.terminal_value,
            discounted_terminal_value=result.discounted_terminal_value,
            enterprise_value=result.enterprise_value,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/dcf/reverse", response_model=ReverseDCFResponse)
def reverse_dcf_endpoint(
    req: Annotated[ReverseDCFRequest, Body(openapi_examples=_REVERSE_DCF_EXAMPLES)],
) -> ReverseDCFResponse:
    """Solve for the implied growth rate that produces the target enterprise value."""
    try:
        g = reverse_dcf_implied_growth(
            target_enterprise_value=req.target_enterprise_value,
            initial_fcf=req.initial_fcf,
            discount_rate=req.discount_rate,
            forecast_years=req.forecast_years,
            terminal_growth_rate=req.terminal_growth_rate,
        )
        return ReverseDCFResponse(implied_growth_rate=g)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
