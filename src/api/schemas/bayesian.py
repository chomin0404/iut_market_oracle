"""Request/response schemas for the Bayesian router."""

from __future__ import annotations

from pydantic import BaseModel, Field

from schemas import Evidence, PriorSpec


class UpdateRequest(BaseModel):
    prior: PriorSpec
    evidence: list[Evidence]


# ---------------------------------------------------------------------------
# Network-based inference schemas
# ---------------------------------------------------------------------------


class NodeDef(BaseModel):
    id: str = Field(..., description="Unique node identifier")
    states: list[str] = Field(..., min_length=2, description="Ordered state labels (>= 2)")


class EdgeDef(BaseModel):
    parent: str = Field(..., description="Parent node id")
    child: str = Field(..., description="Child node id")


class PriorDef(BaseModel):
    node: str = Field(..., description="Root node id (no parents)")
    probs: list[float] = Field(..., description="P(node) — one value per state, must sum to 1")


class CptRowDef(BaseModel):
    parents: list[str] = Field(
        ..., description="Parent state labels in the order parents were added via edges"
    )
    probs: list[float] = Field(
        ..., description="P(node | parents) — one value per child state, must sum to 1"
    )


class CptDef(BaseModel):
    node: str = Field(..., description="Conditional node id (has parents)")
    rows: list[CptRowDef] = Field(..., description="One row per parent-state combination")


class InferRequest(BaseModel):
    nodes: list[NodeDef] = Field(..., description="All nodes in the network")
    edges: list[EdgeDef] = Field(
        default_factory=list, description="Directed edges (parent -> child)"
    )
    priors: list[PriorDef] = Field(..., description="Prior distributions for root nodes")
    cpts: list[CptDef] = Field(
        default_factory=list, description="Conditional probability tables for non-root nodes"
    )
    query: str = Field(..., description="Node id to compute the posterior for")
    evidence: dict[str, str] = Field(
        default_factory=dict,
        description='Observed states, e.g. {"economy": "expansion"}',
    )


class InferResponse(BaseModel):
    query: str = Field(..., description="Queried node id")
    evidence: dict[str, str] = Field(..., description="Evidence used in inference")
    posterior: dict[str, float] = Field(
        ..., description="P(query | evidence) — state -> probability, sums to 1"
    )


# ---------------------------------------------------------------------------
# Named-network registry schemas
# ---------------------------------------------------------------------------


class NetworkInfo(BaseModel):
    name: str = Field(..., description="Registry key for the network")
    description: str = Field(..., description="Human-readable description")


class NamedInferRequest(BaseModel):
    query: str = Field(..., description="Node id to compute the posterior for")
    evidence: dict[str, str] = Field(
        default_factory=dict,
        description='Observed states, e.g. {"season": "summer"}',
    )
