"""Graph metrics endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Body, HTTPException

from graph.metrics import compute_all
from schemas import GraphInput, PortfolioMetrics

router = APIRouter()

_GRAPH_EXAMPLES = {
    "skill_graph": {
        "summary": "Skill dependency graph",
        "value": {
            "nodes": [
                {"node_id": "python", "label": "Python", "weight": 2.0},
                {"node_id": "stats", "label": "Statistics", "weight": 1.5},
                {"node_id": "ml", "label": "Machine Learning", "weight": 1.8},
            ],
            "edges": [
                {"source": "python", "target": "ml", "strength": 0.9},
                {"source": "stats", "target": "ml", "strength": 0.8},
            ],
        },
    },
    "supply_chain": {
        "summary": "Supply chain dependency graph",
        "value": {
            "nodes": [
                {"node_id": "supplier_a"},
                {"node_id": "supplier_b"},
                {"node_id": "assembly"},
                {"node_id": "distribution"},
            ],
            "edges": [
                {"source": "supplier_a", "target": "assembly"},
                {"source": "supplier_b", "target": "assembly"},
                {"source": "assembly", "target": "distribution"},
            ],
        },
    },
}


@router.post("/metrics", response_model=PortfolioMetrics)
def graph_metrics(
    graph: Annotated[GraphInput, Body(openapi_examples=_GRAPH_EXAMPLES)],
) -> PortfolioMetrics:
    """Compute basis diversity, dependency concentration, and portfolio score."""
    try:
        return compute_all(graph)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
