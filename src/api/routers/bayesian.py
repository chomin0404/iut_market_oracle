"""Bayesian update endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Body, HTTPException

from api.schemas.bayesian import UpdateRequest
from bayesian.updater import update
from schemas import PosteriorSummary

router = APIRouter()

_BAYESIAN_EXAMPLES = {
    "beta_prior": {
        "summary": "Beta prior (conversion rate)",
        "value": {
            "prior": {"distribution": "beta", "params": {"alpha": 2.0, "beta": 18.0}},
            "evidence": [
                {"source": "obs_1", "kind": "observation", "value": 0.15, "weight": 1.0},
                {"source": "obs_2", "kind": "observation", "value": 0.20, "weight": 1.0},
            ],
        },
    },
    "normal_prior": {
        "summary": "Normal prior (return estimate)",
        "value": {
            "prior": {"distribution": "normal", "params": {"mu": 0.05, "sigma": 0.02}},
            "evidence": [
                {"source": "q1_return", "kind": "observation", "value": 0.07, "weight": 1.0},
            ],
        },
    },
}


@router.post("/update", response_model=PosteriorSummary)
def bayesian_update(
    req: Annotated[UpdateRequest, Body(openapi_examples=_BAYESIAN_EXAMPLES)],
) -> PosteriorSummary:
    """Bayesian conjugate update (beta or normal) given a prior and evidence list."""
    try:
        return update(req.prior, req.evidence)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
