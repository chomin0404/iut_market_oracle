"""Request schemas for the model registry and generation router (T1400)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    idea: str = Field(..., min_length=1, description="Natural-language idea to formalise")
    domain: str | None = Field(None, description="Optional domain hint, e.g. 'finance'")


class RecommendRequest(BaseModel):
    description: str = Field(..., min_length=1, description="Problem or phenomenon to model")
    signals: list[str] | None = Field(
        None,
        description="Explicit problem characteristics, e.g. ['latent dynamics exist']",
    )
