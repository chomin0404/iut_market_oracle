"""Response schemas for the ModelForge router."""

from __future__ import annotations

from pydantic import BaseModel, Field

from schemas import VerificationReport


class AuditEntry(BaseModel):
    timestamp: str
    model_id: str
    event: str
    verification_overall: str | None = None


class TraceGraphResponse(BaseModel):
    nodes: list[dict] = Field(default_factory=list)
    edges: list[dict] = Field(default_factory=list)


class VerifyAllResponse(BaseModel):
    results: dict[str, VerificationReport]
