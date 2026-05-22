"""Request/response schemas for the report generation router."""

from __future__ import annotations

from pydantic import BaseModel


class ReportRequest(BaseModel):
    scenario_dir: str = "configs/scenarios"
    reports_dir: str = "reports"
    experiments_root: str = "experiments"


class ReportResponse(BaseModel):
    artifacts: dict[str, str]
