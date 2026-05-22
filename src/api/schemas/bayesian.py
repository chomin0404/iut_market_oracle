"""Request schemas for the Bayesian update router."""

from __future__ import annotations

from pydantic import BaseModel

from schemas import Evidence, PriorSpec


class UpdateRequest(BaseModel):
    prior: PriorSpec
    evidence: list[Evidence]
