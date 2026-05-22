"""Request schemas for the matroid router."""

from __future__ import annotations

from pydantic import BaseModel, Field

# Upper bound on n_assets: at n=200 the response contains 201 floats per series (~5 KB).
_N_ASSETS_MAX: int = 200


class LogConcavityRequest(BaseModel):
    n_assets: int = Field(
        ...,
        ge=1,
        le=_N_ASSETS_MAX,
        description="Number of ground elements in the matroid",
    )
    rank_weight: float = Field(
        default=0.8,
        gt=0.0,
        description="Multiplicative weight alpha per element in the independent set",
    )
    corank_weight: float = Field(
        default=1.2,
        gt=0.0,
        description="Multiplicative weight beta per element in the complement",
    )
