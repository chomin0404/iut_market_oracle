"""Request schemas for the experiments registry router."""

from __future__ import annotations

from pydantic import BaseModel


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
