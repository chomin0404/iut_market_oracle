"""FastAPI router for ModelForge (ModelForge).

Endpoints:
    POST /forge/run/{model_id}       — full pipeline for one model or "all"
    POST /forge/verify/{model_id}    — static verification only
    GET  /forge/trace/{model_id}     — traceability DAG for one model
    GET  /forge/audit                — recent audit log entries
    GET  /forge/graph                — full traceability graph (nodes + edges)
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from models.forge import ModelForge
from models.trace import TraceGraph
from models.verifier import verify_all, verify_yaml_file
from schemas import ForgeReport, TraceNode, VerificationReport

router = APIRouter()

_REGISTRY_DIR = Path("configs") / "model_registry"
_AUDIT_LOG = Path(".claude") / "audit" / "modelforge.jsonl"


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/run/{model_id}", response_model=ForgeReport)
def forge_run(model_id: str) -> ForgeReport:
    """Run the full ModelForge pipeline for one model.

    Use model_id="all" to forge every registered model. When "all" is used,
    returns the report for the last model processed (use GET /forge/graph for the full picture).
    """
    try:
        forge = ModelForge()
        if model_id == "all":
            reports = forge.run_all()
            if not reports:
                raise HTTPException(status_code=404, detail="No models found in registry")
            return list(reports.values())[-1]
        return forge.run(model_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/verify/{model_id}", response_model=VerificationReport)
def forge_verify(model_id: str) -> VerificationReport:
    """Static verification only — no artifacts written.

    Use model_id="all" to verify every registered model (returns last report).
    """
    try:
        if model_id == "all":
            reports = verify_all(_REGISTRY_DIR)
            if not reports:
                raise HTTPException(status_code=404, detail="No models found in registry")
            return list(reports.values())[-1]
        yaml_path = _REGISTRY_DIR / f"{model_id}.yaml"
        return verify_yaml_file(yaml_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/trace/{model_id}", response_model=list[TraceNode])
def forge_trace(model_id: str) -> list[TraceNode]:
    """Return all TraceNodes for a given model_id."""
    graph = TraceGraph()
    nodes = graph.load_model(model_id)
    return nodes


@router.get("/graph", response_model=TraceGraphResponse)
def forge_graph() -> TraceGraphResponse:
    """Return the full traceability graph (all models)."""
    graph = TraceGraph()
    data = graph.to_dict()
    return TraceGraphResponse(nodes=data["nodes"], edges=data["edges"])


@router.get("/audit", response_model=list[AuditEntry])
def forge_audit(tail: int = 20) -> list[AuditEntry]:
    """Return the most recent ModelForge audit log entries."""
    if not _AUDIT_LOG.exists():
        return []
    lines = _AUDIT_LOG.read_text(encoding="utf-8").strip().splitlines()
    recent = lines[-tail:]
    entries: list[AuditEntry] = []
    for line in recent:
        try:
            record = json.loads(line)
            entries.append(
                AuditEntry(
                    timestamp=record.get("timestamp", ""),
                    model_id=record.get("model_id", ""),
                    event=record.get("event", ""),
                    verification_overall=record.get("verification_overall"),
                )
            )
        except (json.JSONDecodeError, KeyError):
            continue
    return entries
