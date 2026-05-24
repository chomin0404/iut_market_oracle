"""Experiment registry endpoints."""

from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException

from api.schemas.experiments import ExperimentCreateRequest, ExperimentUpdateRequest
from experiments.tracker import (
    create_experiment,
    list_experiments,
    load_experiment,
    update_experiment,
)
from schemas import ExperimentMeta

router = APIRouter()

_EXPERIMENTS_ROOT = os.environ.get("EXPERIMENTS_ROOT", "experiments")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("", response_model=ExperimentMeta, status_code=201)
def create(req: ExperimentCreateRequest) -> ExperimentMeta:
    """Create a new experiment and register it."""
    try:
        return create_experiment(
            title=req.title,
            config_path=req.config_path,
            result_path=req.result_path,
            note_path=req.note_path,
            random_seed=req.random_seed,
            tags=req.tags,
            summary=req.summary,
            experiments_root=_EXPERIMENTS_ROOT,
        )
    except (ValueError, OverflowError) as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("", response_model=list[ExperimentMeta])
def list_all() -> list[ExperimentMeta]:
    """List all experiments sorted by ID."""
    return list_experiments(_EXPERIMENTS_ROOT)


@router.get("/{exp_id}", response_model=ExperimentMeta)
def get(exp_id: str) -> ExperimentMeta:
    """Load a single experiment by ID."""
    try:
        return load_experiment(exp_id, _EXPERIMENTS_ROOT)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Experiment '{exp_id}' not found.")


@router.patch("/{exp_id}", response_model=ExperimentMeta)
def update(exp_id: str, req: ExperimentUpdateRequest) -> ExperimentMeta:
    """Update writable fields of an existing experiment."""
    fields = {k: v for k, v in req.model_dump().items() if v is not None}
    try:
        return update_experiment(exp_id, experiments_root=_EXPERIMENTS_ROOT, **fields)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
