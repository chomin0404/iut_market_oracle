"""Request/response schemas for the valuation router."""

from __future__ import annotations

import pathlib

from pydantic import BaseModel, Field, field_validator


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

    @field_validator("scenario_dir")
    @classmethod
    def _no_traversal(cls, v: str) -> str:
        p = pathlib.PurePosixPath(v)
        if p.is_absolute() or ".." in p.parts:
            raise ValueError("scenario_dir must be a relative path within configs/")
        return v
