"""Bayesian update endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from bayesian.updater import update
from schemas import Evidence, PosteriorSummary, PriorSpec

router = APIRouter()


class UpdateRequest(BaseModel):
    prior: PriorSpec
    evidence: list[Evidence]


@router.post("/update", response_model=PosteriorSummary)
def bayesian_update(req: UpdateRequest) -> PosteriorSummary:
    """Bayesian conjugate update (beta or normal) given a prior and evidence list."""
    try:
        return update(req.prior, req.evidence)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
