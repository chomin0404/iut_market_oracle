"""Valuation endpoints: DCF scenario running and reverse DCF."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from schemas import AssumptionSet, ScenarioResult
from valuation.dcf import DCFInputs, dcf_valuation, reverse_dcf_implied_growth
from valuation.scenario import run_all_scenarios, run_scenario

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class DCFRequest(BaseModel):
    initial_fcf: float = Field(..., gt=0.0, description="Initial free cash flow")
    growth_rate: float = Field(..., description="Annual revenue growth rate (decimal)")
    discount_rate: float = Field(..., description="WACC (decimal)")
    forecast_years: int = Field(default=5, ge=1)
    terminal_growth_rate: float = Field(default=0.03, description="Gordon Growth rate (decimal)")


class DCFResponse(BaseModel):
    projected_fcfs: list[float]
    discounted_fcfs: list[float]
    terminal_value: float
    discounted_terminal_value: float
    enterprise_value: float


class ReverseDCFRequest(BaseModel):
    target_enterprise_value: float = Field(..., gt=0.0)
    initial_fcf: float = Field(..., gt=0.0)
    discount_rate: float
    forecast_years: int = Field(default=5, ge=1)
    terminal_growth_rate: float = Field(default=0.03)


class ReverseDCFResponse(BaseModel):
    implied_growth_rate: float


class RunAllRequest(BaseModel):
    scenario_dir: str = Field(
        default="configs/scenarios", description="Directory with *.yaml scenario files"
    )


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
def dcf_endpoint(req: DCFRequest) -> DCFResponse:
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
def reverse_dcf_endpoint(req: ReverseDCFRequest) -> ReverseDCFResponse:
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
