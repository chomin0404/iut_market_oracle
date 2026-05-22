"""Monte Carlo risk analysis API endpoints.

Routes (mounted at /risk):
    POST /risk/simulate    — Copula-based MC simulation with marginal distributions.
    POST /risk/boundary    — Exceedance curve and bootstrap confidence band.
"""

from __future__ import annotations

import uuid
from collections import OrderedDict
from typing import Annotated

import numpy as np
from fastapi import APIRouter, Body, HTTPException

from api.schemas.risk import (
    DEFAULT_ALPHA,
    BoundaryRequest,
    BoundaryResponse,
    ConfidenceBand,
    SimulateRequest,
    SimulateResponse,
    VariableSummary,
)
from core.risk_metrics import (
    compute_confidence_band,
    compute_es,
    compute_exceedance_curve,
    compute_var,
)
from core.simulator import MonteCarloSimulator

_simulator = MonteCarloSimulator()

router = APIRouter()

# Bounded LRU simulation cache: simulation_id -> np.ndarray (n_samples, n_vars)
# Oldest entries are evicted when the cache exceeds _CACHE_MAXSIZE.
# NOTE: This is an in-process dict — not shared across Uvicorn workers.
#       For multi-worker deployments, replace with Redis or a shared store.
_CACHE_MAXSIZE = 200
_SIMULATION_CACHE: OrderedDict[str, np.ndarray] = OrderedDict()


def _cache_put(sim_id: str, arr: np.ndarray) -> None:
    """Insert into the LRU cache, evicting the oldest entry if full."""
    _SIMULATION_CACHE[sim_id] = arr
    while len(_SIMULATION_CACHE) > _CACHE_MAXSIZE:
        _SIMULATION_CACHE.popitem(last=False)


def _cache_get(sim_id: str) -> np.ndarray | None:
    """Retrieve and refresh recency of a cached simulation."""
    if sim_id not in _SIMULATION_CACHE:
        return None
    _SIMULATION_CACHE.move_to_end(sim_id)
    return _SIMULATION_CACHE[sim_id]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/simulate",
    response_model=SimulateResponse,
    tags=["risk"],
    summary="Run copula-based Monte Carlo simulation",
    operation_id="risk_simulate",
)
def simulate(
    body: Annotated[
        SimulateRequest,
        Body(
            openapi_examples={
                "two_var_gaussian": {
                    "summary": "2-variable Gaussian copula (normal + lognormal)",
                    "value": {
                        "n_vars": 2,
                        "n_samples": 10000,
                        "distributions": [
                            {"name": "normal", "params": {"loc": 0.0, "scale": 1.0}},
                            {
                                "name": "lognormal",
                                "params": {"s": 0.8, "loc": 0.0, "scale": 1.0},
                            },
                        ],
                        "copula": {
                            "type": "gaussian",
                            "corr_matrix": [[1, 0.6], [0.6, 1]],
                        },
                        "seed": 42,
                    },
                }
            }
        ),
    ],
) -> SimulateResponse:
    """Run a copula-based Monte Carlo simulation.

    Generates *n_samples* draws from a multivariate distribution whose marginals
    are joined by the specified copula. The result is cached in memory under a UUID
    that can be passed to `POST /api/v1/risk/boundary`.

    Supported copulas: gaussian, student_t, clayton, independent.
    """
    copula_dict: dict = {"type": body.copula.type}
    if body.copula.corr_matrix is not None:
        copula_dict["corr_matrix"] = body.copula.corr_matrix
    if body.copula.df is not None:
        copula_dict["df"] = body.copula.df
    if body.copula.theta is not None:
        copula_dict["theta"] = body.copula.theta

    try:
        result = _simulator.simulate(
            n_vars=body.n_vars,
            n_samples=body.n_samples,
            distributions=[d.model_dump() for d in body.distributions],
            copula=copula_dict,
            seed=body.seed,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Cache as (n_samples, n_vars) for column access in /boundary
    sim_id = str(uuid.uuid4())
    _cache_put(sim_id, result.samples.T)

    col = result.samples[0]  # variable index 0, shape (n_samples,)
    return SimulateResponse(
        simulation_id=sim_id,
        n_samples=body.n_samples,
        summary=VariableSummary(
            mean=float(col.mean()),
            std=float(col.std()),
            var_95=compute_var(col, DEFAULT_ALPHA),
            es_95=compute_es(col, DEFAULT_ALPHA),
        ),
    )


@router.post(
    "/boundary",
    response_model=BoundaryResponse,
    tags=["risk"],
    summary="Compute exceedance curve and confidence band",
    operation_id="risk_boundary",
)
def risk_boundary(
    body: Annotated[
        BoundaryRequest,
        Body(
            openapi_examples={
                "from_simulation_id": {
                    "summary": "From stored simulation (use simulation_id from /simulate)",
                    "value": {
                        "simulation_id": "00000000-0000-0000-0000-000000000000",
                        "target_variable_index": 0,
                        "thresholds": [1.0, 2.0, 3.0, 4.0, 5.0],
                        "confidence_level": 0.95,
                        "bootstrap_n": 500,
                    },
                },
                "from_raw_samples": {
                    "summary": "From raw samples array",
                    "value": {
                        "samples": [0.5, 1.2, 2.3, 0.8, 3.1, 1.5],
                        "thresholds": [1.0, 2.0, 3.0],
                        "confidence_level": 0.95,
                        "bootstrap_n": 200,
                    },
                },
            }
        ),
    ],
) -> BoundaryResponse:
    """Compute exceedance probabilities and bootstrap confidence band.

    Either reference a stored `simulation_id` or pass raw `samples` directly.
    Returns the empirical P(X > t) for each threshold together with 95% CI
    bounds computed by non-parametric bootstrap.
    """
    if body.simulation_id is not None:
        arr = _cache_get(body.simulation_id)
        if arr is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"simulation_id {body.simulation_id!r} not found. "
                    "Run POST /risk/simulate first."
                ),
            )
        idx = body.target_variable_index
        if idx >= arr.shape[1]:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"target_variable_index={idx} out of range "
                    f"for simulation with n_vars={arr.shape[1]}."
                ),
            )
        col = arr[:, idx]
    else:
        col = np.asarray(body.samples, dtype=float)

    exc_probs = compute_exceedance_curve(col, body.thresholds)
    band = compute_confidence_band(
        col, body.thresholds, bootstrap_n=body.bootstrap_n, seed=body.bootstrap_seed
    )
    var = compute_var(col, body.confidence_level)
    es = compute_es(col, body.confidence_level)

    return BoundaryResponse(
        thresholds=body.thresholds,
        exceedance_probs=exc_probs,
        confidence_band=ConfidenceBand(**band),
        var_95=var,
        es_95=es,
    )
