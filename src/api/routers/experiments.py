"""Experiment registry endpoints."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from experiments.tracker import (
    create_experiment,
    list_experiments,
    load_experiment,
    update_experiment,
)
from schemas import ExperimentMeta

router = APIRouter()


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class ExperimentCreateRequest(BaseModel):
    title: str
    config_path: str
    result_path: str | None = None
    note_path: str | None = None
    random_seed: int | None = None
    tags: list[str] = []
    summary: str = ""
    experiments_root: str = "experiments"


class ExperimentUpdateRequest(BaseModel):
    result_path: str | None = None
    note_path: str | None = None
    summary: str | None = None
    tags: list[str] | None = None
    experiments_root: str = "experiments"


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
            experiments_root=req.experiments_root,
        )
    except (ValueError, OverflowError) as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("", response_model=list[ExperimentMeta])
def list_all(experiments_root: str = "experiments") -> list[ExperimentMeta]:
    """List all experiments sorted by ID."""
    return list_experiments(experiments_root)


@router.get("/{exp_id}", response_model=ExperimentMeta)
def get(exp_id: str, experiments_root: str = "experiments") -> ExperimentMeta:
    """Load a single experiment by ID."""
    try:
        return load_experiment(exp_id, experiments_root)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.patch("/{exp_id}", response_model=ExperimentMeta)
def update(exp_id: str, req: ExperimentUpdateRequest) -> ExperimentMeta:
    """Update writable fields of an existing experiment."""
    fields = {k: v for k, v in req.model_dump(exclude={"experiments_root"}).items() if v is not None}
    try:
        return update_experiment(exp_id, experiments_root=req.experiments_root, **fields)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
