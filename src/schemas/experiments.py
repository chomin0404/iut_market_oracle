"""T500 Experiment Registry schemas."""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class ExperimentMeta(BaseModel):
    """Metadata envelope for a single reproducible experiment run."""

    experiment_id: str = Field(..., pattern=r"^exp-\d+$", description="e.g. 'exp-001'")
    title: str = Field(..., min_length=1)
    config_path: str = Field(..., description="Relative path to config file used")
    result_path: str | None = None
    note_path: str | None = None
    random_seed: int | None = None
    tags: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    summary: str = ""
