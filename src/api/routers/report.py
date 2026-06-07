"""Report generation endpoint."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from api.schemas.report import ReportRequest, ReportResponse
from report import run_report

router = APIRouter()
_logger = logging.getLogger(__name__)


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
    except FileNotFoundError:
        _logger.exception("unexpected error")
        raise HTTPException(  # noqa: B904
            status_code=404, detail="Scenario directory or required file not found."
        )
    except ValueError as e:
        _logger.warning("%s", e, exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
