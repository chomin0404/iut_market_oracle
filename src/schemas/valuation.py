"""T400 Valuation / Scenario schemas."""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field, field_validator


class AssumptionSet(BaseModel):
    """Named, versioned set of modelling assumptions."""

    name: str = Field(..., min_length=1)
    version: str = "1.0"
    params: dict[str, float | str | bool] = Field(..., description="Scenario parameters")
    random_seed: int | None = None
    description: str = ""

    @field_validator("params")
    @classmethod
    def non_empty_params(cls, v: dict) -> dict:
        if not v:
            raise ValueError("params must not be empty")
        return v


class ScenarioResult(BaseModel):
    """Output produced by one valuation scenario run."""

    scenario_name: str = Field(..., min_length=1)
    assumption_version: str
    value: float
    unit: str = ""
    sensitivity: dict[str, float] = Field(
        default_factory=dict,
        description="Partial derivatives w.r.t. each assumption param",
    )
    produced_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    output_path: str | None = None
