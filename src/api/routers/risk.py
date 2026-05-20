"""Monte Carlo risk analysis API endpoints.

Routes (mounted at /api/v1):
    POST /api/v1/simulate        — Copula-based MC simulation with marginal distributions.
    POST /api/v1/risk/boundary   — Exceedance curve and bootstrap confidence band.
"""

from __future__ import annotations

import uuid
from typing import Annotated

import numpy as np
from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel, Field, model_validator

from core.risk_metrics import (
    compute_confidence_band,
    compute_es,
    compute_exceedance_curve,
    compute_var,
)
from core.simulator import simulate_gaussian_copula

router = APIRouter()

# In-memory simulation cache: simulation_id -> np.ndarray (n_samples, n_vars)
# No expiry — process lifetime cache per spec.
_SIMULATION_CACHE: dict[str, np.ndarray] = {}

MAX_N_SAMPLES = 100_000
DEFAULT_ALPHA = 0.95

# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class DistributionSpec(BaseModel):
    name: str = Field(
        ...,
        description=(
            "Distribution name. Supported: normal, lognormal, t, uniform, gev, expon, beta, gamma."
        ),
    )
    params: dict[str, float] = Field(
        default_factory=dict,
        description="scipy.stats keyword arguments (e.g. loc, scale, s, df, c).",
    )


class CopulaSpec(BaseModel):
    type: str = Field("gaussian", description="Copula type. Currently only 'gaussian'.")
    corr_matrix: list[list[float]] = Field(
        ..., description="Symmetric positive-definite correlation matrix (n_vars × n_vars)."
    )


class SimulateRequest(BaseModel):
    n_vars: int = Field(..., ge=1, le=20, description="Number of variables.")
    n_samples: int = Field(..., ge=100, le=MAX_N_SAMPLES, description="Number of MC samples.")
    distributions: list[DistributionSpec] = Field(
        ..., description="Marginal distribution for each variable (length must equal n_vars)."
    )
    copula: CopulaSpec
    seed: int | None = Field(None, description="Random seed for reproducibility.")

    @model_validator(mode="after")
    def _check_dimensions(self) -> SimulateRequest:
        if len(self.distributions) != self.n_vars:
            raise ValueError(
                f"distributions has {len(self.distributions)} entries but n_vars={self.n_vars}"
            )
        rows = len(self.copula.corr_matrix)
        if rows != self.n_vars:
            raise ValueError(f"corr_matrix has {rows} rows but n_vars={self.n_vars}")
        return self


class VariableSummary(BaseModel):
    mean: float
    std: float
    var_95: float = Field(..., description="95th-percentile VaR of variable index 0.")
    es_95: float = Field(..., description="95% Expected Shortfall of variable index 0.")


class SimulateResponse(BaseModel):
    simulation_id: str = Field(..., description="UUID for downstream /risk/boundary calls.")
    n_samples: int
    summary: VariableSummary = Field(..., description="Summary statistics for variable index 0.")


class BoundaryRequest(BaseModel):
    simulation_id: str | None = Field(None, description="ID returned by POST /api/v1/simulate.")
    samples: list[float] | None = Field(
        None, description="Raw samples (alternative to simulation_id)."
    )
    target_variable_index: int = Field(
        0, ge=0, description="Column index in the stored simulation array."
    )
    thresholds: list[float] = Field(..., min_length=1, description="Threshold values.")
    confidence_level: float = Field(
        DEFAULT_ALPHA, ge=0.5, lt=1.0, description="Confidence level for VaR/ES."
    )
    bootstrap_n: int = Field(500, ge=10, le=5000, description="Bootstrap resamples.")

    @model_validator(mode="after")
    def _check_source(self) -> BoundaryRequest:
        if self.simulation_id is None and self.samples is None:
            raise ValueError("Provide either simulation_id or samples.")
        return self


class ConfidenceBand(BaseModel):
    lower: list[float]
    upper: list[float]


class BoundaryResponse(BaseModel):
    thresholds: list[float]
    exceedance_probs: list[float]
    confidence_band: ConfidenceBand
    var_95: float
    es_95: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/simulate",
    response_model=SimulateResponse,
    tags=["risk"],
    summary="Run copula-based Monte Carlo simulation",
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
    are joined by a Gaussian copula. The result is cached in memory under a UUID
    that can be passed to `POST /api/v1/risk/boundary`.
    """
    if body.copula.type != "gaussian":
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported copula type: {body.copula.type!r}. Only 'gaussian' is supported.",
        )

    try:
        samples = simulate_gaussian_copula(
            n_vars=body.n_vars,
            n_samples=body.n_samples,
            distributions=[d.model_dump() for d in body.distributions],
            corr_matrix=body.copula.corr_matrix,
            seed=body.seed,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    sim_id = str(uuid.uuid4())
    _SIMULATION_CACHE[sim_id] = samples

    col = samples[:, 0]
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
    "/risk/boundary",
    response_model=BoundaryResponse,
    tags=["risk"],
    summary="Compute exceedance curve and confidence band",
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
        if body.simulation_id not in _SIMULATION_CACHE:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"simulation_id {body.simulation_id!r} not found. "
                    "Run POST /api/v1/simulate first."
                ),
            )
        arr = _SIMULATION_CACHE[body.simulation_id]
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
    band = compute_confidence_band(col, body.thresholds, bootstrap_n=body.bootstrap_n)
    var = compute_var(col, body.confidence_level)
    es = compute_es(col, body.confidence_level)

    return BoundaryResponse(
        thresholds=body.thresholds,
        exceedance_probs=exc_probs,
        confidence_band=ConfidenceBand(**band),
        var_95=var,
        es_95=es,
    )
