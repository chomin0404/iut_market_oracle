"""Graph metrics endpoints."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from graph.metrics import compute_all
from schemas import GraphInput, PortfolioMetrics

router = APIRouter()


@router.post("/metrics", response_model=PortfolioMetrics)
def graph_metrics(graph: GraphInput) -> PortfolioMetrics:
    """Compute basis diversity, dependency concentration, and portfolio score."""
    try:
        return compute_all(graph)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
