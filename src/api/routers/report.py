"""Report generation endpoint."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from report import run_report

router = APIRouter()


class ReportRequest(BaseModel):
    scenario_dir: str = "configs/scenarios"
    reports_dir: str = "reports"
    experiments_root: str = "experiments"


class ReportResponse(BaseModel):
    artifacts: dict[str, str]


@router.post("/run", response_model=ReportResponse)
def run_report_endpoint(req: ReportRequest) -> ReportResponse:
    """Execute the full DCF report pipeline.

    Loads all *.yaml scenario files from scenario_dir, runs DCF + sensitivity,
    generates charts and a markdown summary, registers the experiment, and
    returns the paths of all generated artifacts.
    """
    try:
        artifacts = run_report(
            scenario_dir=req.scenario_dir,
            reports_dir=req.reports_dir,
            experiments_root=req.experiments_root,
        )
        return ReportResponse(artifacts={k: str(v) for k, v in artifacts.items()})
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
