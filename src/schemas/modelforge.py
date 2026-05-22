"""ModelForge Traceability + Verification schemas."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class VerificationStatus(str, Enum):
    """Result level for one verification check or an overall report."""

    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


class VerificationCheck(BaseModel):
    """Single atomic check in a ModelForge verification run."""

    name: str = Field(..., min_length=1, description="Machine-readable check identifier")
    status: VerificationStatus
    message: str = Field(default="", description="Human-readable detail or empty on PASS")


class VerificationReport(BaseModel):
    """Static verification result for one model registry entry.

    overall: worst status across all checks (FAIL > WARN > PASS).
    registry_hash: SHA-256 of the YAML file content at verification time.
    """

    model_id: str = Field(..., min_length=1)
    registry_hash: str = Field(..., description="SHA-256 hex digest of the YAML content")
    checks: list[VerificationCheck]
    overall: VerificationStatus
    produced_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TraceNodeType(str, Enum):
    """Artifact type in the ModelForge traceability graph."""

    REGISTRY = "registry"  # configs/model_registry/<id>.yaml
    VERIFICATION = "verification"  # artifacts/modelforge/<id>/verification.json
    GENERATED_CODE = "generated_code"  # artifacts/modelforge/<id>/impl_skeleton.py
    AUDIT_ENTRY = "audit_entry"  # .claude/audit/modelforge.jsonl line


class TraceNode(BaseModel):
    """Node in the ModelForge directed acyclic traceability graph.

    node_id: unique stable identifier (SHA-256 prefix of content_hash + type).
    parent_ids: edges — this node was produced from these parent nodes.
    content_hash: SHA-256 of the artifact file at creation time.
    """

    node_id: str = Field(..., min_length=1)
    node_type: TraceNodeType
    model_id: str = Field(..., min_length=1)
    artifact_path: str = Field(..., description="Relative path from project root")
    content_hash: str = Field(..., description="SHA-256 hex digest of artifact content")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    parent_ids: list[str] = Field(default_factory=list)


class ForgeReport(BaseModel):
    """Full output of one ModelForge pipeline run for a single model.

    Connects YAML spec → verification → skeleton code → trace nodes → audit.
    """

    model_id: str = Field(..., min_length=1)
    registry_yaml_path: str
    verification: VerificationReport
    skeleton_code_path: str = Field(..., description="Path to generated impl_skeleton.py")
    trace_node_ids: list[str] = Field(..., description="Ordered node IDs created in this run")
    produced_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
