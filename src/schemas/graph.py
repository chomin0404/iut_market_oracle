"""T300 Dependency / Skill Graph schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


class NodeMeta(BaseModel):
    """Metadata for a node in the skill/dependency graph."""

    node_id: str = Field(..., min_length=1)
    label: str = ""
    category: str = ""
    weight: float = Field(default=1.0, gt=0.0)
    attributes: dict[str, Any] = Field(default_factory=dict)


class EdgeMeta(BaseModel):
    """Directed edge in the dependency graph."""

    source: str = Field(..., min_length=1)
    target: str = Field(..., min_length=1)
    strength: float = Field(default=1.0, ge=0.0)
    label: str = ""

    @model_validator(mode="after")
    def no_self_loop(self) -> EdgeMeta:
        if self.source == self.target:
            raise ValueError("Self-loops are not allowed (source == target)")
        return self


class GraphInput(BaseModel):
    """Full graph payload for graph analysis module."""

    nodes: list[NodeMeta] = Field(..., min_length=1)
    edges: list[EdgeMeta] = Field(default_factory=list)

    @model_validator(mode="after")
    def edges_reference_existing_nodes(self) -> GraphInput:
        ids = {n.node_id for n in self.nodes}
        for e in self.edges:
            if e.source not in ids:
                raise ValueError(f"Edge source '{e.source}' not in node list")
            if e.target not in ids:
                raise ValueError(f"Edge target '{e.target}' not in node list")
        return self


class PortfolioMetrics(BaseModel):
    """Computed metrics from graph analysis."""

    basis_diversity: float = Field(..., ge=0.0, le=1.0)
    dependency_concentration: float = Field(..., ge=0.0)
    portfolio_score: float = Field(..., ge=0.0, le=1.0)
    node_count: int = Field(..., ge=1)
    edge_count: int = Field(..., ge=0)
    notes: str = ""
