"""Monte Carlo risk analysis API endpoints.

Routes (mounted at /risk):
    POST /risk/simulate    — Copula-based MC simulation with marginal distributions.
    POST /risk/boundary    — Exceedance curve and bootstrap confidence band.
"""

from __future__ import annotations

import asyncio
import threading
import uuid
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Annotated

import numpy as np
from fastapi import APIRouter, Body, HTTPException

from api.schemas.risk import (
    DEFAULT_ALPHA,
    BoundaryRequest,
    BoundaryResponse,
    ConfidenceBand,
    CopulaSpec,
    GnssScenarioRequest,
    GnssScenarioResponse,
    GnssVariableResult,
    SensitivityRequest,
    SensitivityResponse,
    SimulateRequest,
    SimulateResponse,
    SweepParameter,
    TailRequest,
    TailResponse,
    TailStatEntry,
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
_CACHE_LOCK = threading.Lock()


def _cache_put(sim_id: str, arr: np.ndarray) -> None:
    """Insert into the LRU cache, evicting the oldest entry if full."""
    with _CACHE_LOCK:
        _SIMULATION_CACHE[sim_id] = arr
        while len(_SIMULATION_CACHE) > _CACHE_MAXSIZE:
            _SIMULATION_CACHE.popitem(last=False)


def _cache_get(sim_id: str) -> np.ndarray | None:
    """Retrieve and refresh recency of a cached simulation."""
    with _CACHE_LOCK:
        if sim_id not in _SIMULATION_CACHE:
            return None
        _SIMULATION_CACHE.move_to_end(sim_id)
        return _SIMULATION_CACHE[sim_id]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _build_copula_dict(spec: CopulaSpec) -> dict:
    """Convert a CopulaSpec into the dict expected by MonteCarloSimulator.simulate()."""
    d: dict = {"type": spec.type}
    if spec.corr_matrix is not None:
        d["corr_matrix"] = spec.corr_matrix
    if spec.df is not None:
        d["df"] = spec.df
    if spec.theta is not None:
        d["theta"] = spec.theta
    return d


def _resolve_samples(
    body_sim_id: str | None, body_samples: list[float] | None, target_idx: int
) -> np.ndarray:
    """Resolve sample array from either a cached simulation_id or raw samples list.

    Raises HTTPException 404 / 400 on failure.
    """
    if body_sim_id is not None:
        arr = _cache_get(body_sim_id)
        if arr is None:
            raise HTTPException(
                status_code=404,
                detail=(f"simulation_id {body_sim_id!r} not found. Run POST /risk/simulate first."),
            )
        if target_idx >= arr.shape[1]:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"target_variable_index={target_idx} out of range "
                    f"for simulation with n_vars={arr.shape[1]}."
                ),
            )
        return arr[:, target_idx]
    # raw samples path (body_samples is guaranteed non-None by schema validator)
    return np.asarray(body_samples, dtype=float)


# ---------------------------------------------------------------------------
# Sensitivity helpers
# ---------------------------------------------------------------------------

# Maps risk_metric name → (function, alpha)
_RISK_METRIC_FN: dict[str, tuple[Callable[[np.ndarray, float], float], float]] = {
    "var_95": (compute_var, 0.95),
    "es_95": (compute_es, 0.95),
    "var_99": (compute_var, 0.99),
}

# Maximum number of threads for sensitivity sweep parallelism.
_MAX_SENSITIVITY_WORKERS = 8


def _apply_sweep(base: SimulateRequest, sweep: SweepParameter, value: float) -> SimulateRequest:
    """Return a new SimulateRequest with one parameter replaced by *value*."""
    d = base.model_dump()
    target = sweep.target
    param = sweep.param_name

    if target == "copula":
        if param == "corr_matrix_off_diagonal":
            n = base.n_vars
            d["copula"]["corr_matrix"] = [
                [1.0 if i == j else value for j in range(n)] for i in range(n)
            ]
        elif param in ("df", "theta"):
            d["copula"][param] = value
        else:
            raise ValueError(
                f"Unknown copula param_name {param!r}. "
                "Expected: corr_matrix_off_diagonal, df, theta."
            )
    elif target.startswith("distribution_"):
        raw_idx = target[len("distribution_") :]
        try:
            idx = int(raw_idx)
        except ValueError:
            raise ValueError(f"Invalid target {target!r}. Expected format: distribution_{{int}}.")
        if idx >= len(d["distributions"]):
            raise ValueError(
                f"target index {idx} out of range for {len(d['distributions'])} distributions."
            )
        d["distributions"][idx]["params"][param] = value
    else:
        raise ValueError(f"Unknown target {target!r}. Expected 'copula' or 'distribution_{{i}}'.")

    return SimulateRequest.model_validate(d)


def _run_single_sweep(
    base_config: SimulateRequest,
    sweep: SweepParameter,
    val: float,
    target_idx: int,
    fn: Callable[[np.ndarray, float], float],
    alpha: float,
) -> float:
    """Execute one sweep step and return the risk metric value.

    Raises ValueError for invalid parameters (propagated to the caller).
    Thread-safe: _simulator.simulate() uses only local NumPy state.
    """
    swept = _apply_sweep(base_config, sweep, val)
    if target_idx >= swept.n_vars:
        raise ValueError(
            f"target_variable_index={target_idx} out of range for n_vars={swept.n_vars}."
        )
    result = _simulator.simulate(
        n_vars=swept.n_vars,
        n_samples=swept.n_samples,
        distributions=[d.model_dump() for d in swept.distributions],
        copula=_build_copula_dict(swept.copula),
        seed=swept.seed,
    )
    return fn(result.samples[target_idx], alpha)


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
async def simulate(
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
    def _run() -> SimulateResponse:
        try:
            result = _simulator.simulate(
                n_vars=body.n_vars,
                n_samples=body.n_samples,
                distributions=[d.model_dump() for d in body.distributions],
                copula=_build_copula_dict(body.copula),
                seed=body.seed,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        sim_id = str(uuid.uuid4())
        _cache_put(sim_id, result.samples.T)

        col = result.samples[0]
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

    return await asyncio.to_thread(_run)


@router.post(
    "/boundary",
    response_model=BoundaryResponse,
    tags=["risk"],
    summary="Compute exceedance curve and confidence band",
    operation_id="risk_boundary",
)
async def risk_boundary(
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
    col = _resolve_samples(body.simulation_id, body.samples, body.target_variable_index)

    def _run() -> BoundaryResponse:
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

    return await asyncio.to_thread(_run)


@router.post(
    "/tail",
    response_model=TailResponse,
    tags=["risk"],
    summary="Tail risk (VaR and ES) at multiple confidence levels",
    operation_id="risk_tail",
)
async def risk_tail(
    body: Annotated[
        TailRequest,
        Body(
            openapi_examples={
                "from_simulation_id": {
                    "summary": "Multi-alpha tail from stored simulation",
                    "value": {
                        "simulation_id": "00000000-0000-0000-0000-000000000000",
                        "target_variable_index": 0,
                        "alphas": [0.90, 0.95, 0.99],
                    },
                },
                "from_raw_samples": {
                    "summary": "Multi-alpha tail from raw samples",
                    "value": {
                        "samples": [0.5, 1.2, 2.3, 0.8, 3.1, 1.5, 2.8, 0.4, 1.9, 3.7],
                        "alphas": [0.90, 0.95, 0.99],
                    },
                },
            }
        ),
    ],
) -> TailResponse:
    """Compute VaR and Expected Shortfall at multiple confidence levels.

    Either reference a stored ``simulation_id`` or pass raw ``samples`` directly.
    Returns one ``TailStatEntry`` per alpha, sorted in ascending order of alpha.

    This endpoint is equivalent to calling ``POST /risk/boundary`` with explicit thresholds
    but is optimised for the common case of needing VaR/ES at several confidence levels at once.
    """
    col = _resolve_samples(body.simulation_id, body.samples, body.target_variable_index)

    def _run() -> TailResponse:
        stats = [
            TailStatEntry(alpha=a, var=compute_var(col, a), es=compute_es(col, a))
            for a in sorted(body.alphas)
        ]
        return TailResponse(tail_stats=stats, n_samples=len(col))

    return await asyncio.to_thread(_run)


@router.post(
    "/gnss_scenario",
    response_model=GnssScenarioResponse,
    tags=["risk"],
    summary="GNSS attack scenario tail risk analysis (VaR/ES per variable)",
    operation_id="risk_gnss_scenario",
)
async def risk_gnss_scenario(
    body: Annotated[
        GnssScenarioRequest,
        Body(
            openapi_examples={
                "spoofing_student_t": {
                    "summary": "Spoofing scenario — 3-variable Student-T copula",
                    "value": {
                        "scenario_name": "multipath_and_spoofing",
                        "variables": [
                            {
                                "name": "position_error_m",
                                "distribution": {
                                    "name": "lognormal",
                                    "params": {"s": 0.8, "loc": 0.0, "scale": 2.0},
                                },
                            },
                            {
                                "name": "cn0_degradation_db",
                                "distribution": {
                                    "name": "normal",
                                    "params": {"loc": 3.0, "scale": 1.5},
                                },
                            },
                            {
                                "name": "time_error_ns",
                                "distribution": {
                                    "name": "gev",
                                    "params": {"c": 0.2, "loc": 10.0, "scale": 5.0},
                                },
                            },
                        ],
                        "copula": {
                            "type": "student_t",
                            "corr_matrix": [
                                [1.0, 0.7, 0.4],
                                [0.7, 1.0, 0.3],
                                [0.4, 0.3, 1.0],
                            ],
                            "df": 4.0,
                        },
                        "n_samples": 10000,
                        "seed": 42,
                        "alphas": [0.90, 0.95, 0.99],
                    },
                },
                "independent_baseline": {
                    "summary": "Baseline — independent Gaussian copula",
                    "value": {
                        "scenario_name": "nominal_baseline",
                        "variables": [
                            {
                                "name": "position_error_m",
                                "distribution": {
                                    "name": "lognormal",
                                    "params": {"s": 0.3, "loc": 0.0, "scale": 1.0},
                                },
                            },
                            {
                                "name": "cn0_degradation_db",
                                "distribution": {
                                    "name": "normal",
                                    "params": {"loc": 0.5, "scale": 0.5},
                                },
                            },
                        ],
                        "copula": {"type": "independent"},
                        "n_samples": 5000,
                        "seed": 0,
                        "alphas": [0.95, 0.99],
                    },
                },
            }
        ),
    ],
) -> GnssScenarioResponse:
    """Run a GNSS attack scenario risk analysis.

    Models named GNSS observables (e.g. ``position_error_m``, ``cn0_degradation_db``,
    ``time_error_ns``) as a multivariate distribution joined by the specified copula.
    Returns VaR and ES at each requested confidence level for every variable.

    Typical copula choice for spoofing scenarios: ``student_t`` with low df (heavy tails)
    and high off-diagonal correlation to capture simultaneous signal degradation.
    """
    def _run() -> GnssScenarioResponse:
        try:
            result = _simulator.simulate(
                n_vars=len(body.variables),
                n_samples=body.n_samples,
                distributions=[v.distribution.model_dump() for v in body.variables],
                copula=_build_copula_dict(body.copula),
                seed=body.seed,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        sorted_alphas = sorted(body.alphas)
        per_variable = [
            GnssVariableResult(
                name=var_spec.name,
                mean=float(col.mean()),
                std=float(col.std()),
                tail_stats=[
                    TailStatEntry(alpha=a, var=compute_var(col, a), es=compute_es(col, a))
                    for a in sorted_alphas
                ],
            )
            for var_spec, col in zip(body.variables, result.samples)
        ]
        return GnssScenarioResponse(
            scenario_name=body.scenario_name,
            n_samples=body.n_samples,
            per_variable=per_variable,
        )

    return await asyncio.to_thread(_run)


@router.post(
    "/sensitivity",
    response_model=SensitivityResponse,
    tags=["risk"],
    summary="Parameter sensitivity analysis for VaR/ES",
    operation_id="risk_sensitivity",
)
async def risk_sensitivity(
    body: Annotated[
        SensitivityRequest,
        Body(
            openapi_examples={
                "scale_sweep": {
                    "summary": "Sweep distribution scale from 1 to 3 (VaR increases)",
                    "value": {
                        "base_config": {
                            "n_vars": 2,
                            "n_samples": 5000,
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
                        "sweep_parameter": {
                            "target": "distribution_0",
                            "param_name": "scale",
                            "values": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
                        },
                        "risk_metric": "var_95",
                        "target_variable_index": 0,
                    },
                },
                "corr_sweep": {
                    "summary": "Sweep Gaussian copula correlation from 0 to 0.95",
                    "value": {
                        "base_config": {
                            "n_vars": 2,
                            "n_samples": 5000,
                            "distributions": [
                                {"name": "normal", "params": {"loc": 0.0, "scale": 1.0}},
                                {
                                    "name": "lognormal",
                                    "params": {"s": 0.8, "loc": 0.0, "scale": 1.0},
                                },
                            ],
                            "copula": {
                                "type": "gaussian",
                                "corr_matrix": [[1, 0.0], [0.0, 1]],
                            },
                            "seed": 42,
                        },
                        "sweep_parameter": {
                            "target": "copula",
                            "param_name": "corr_matrix_off_diagonal",
                            "values": [0.0, 0.2, 0.4, 0.6, 0.8, 0.95],
                        },
                        "risk_metric": "var_95",
                        "target_variable_index": 0,
                    },
                },
            }
        ),
    ],
) -> SensitivityResponse:
    """Sweep a single parameter and compute VaR or ES at each value.

    For each value in ``sweep_parameter.values``, the base simulation config is
    cloned with that parameter replaced and the simulation is re-run. The risk
    metric is then evaluated on ``target_variable_index``.

    **Note on Gaussian copula:** The marginal of variable 0 in a Gaussian copula
    is independent of the correlation structure (Cholesky column ordering). Use a
    ``distribution_{i}`` sweep to observe monotone VaR changes with fixed seed.
    """
    fn, alpha = _RISK_METRIC_FN[body.risk_metric]
    sweep = body.sweep_parameter
    idx = body.target_variable_index

    def _run() -> SensitivityResponse:
        _one_step = partial(
            _run_single_sweep, body.base_config, sweep, target_idx=idx, fn=fn, alpha=alpha
        )
        n_workers = min(len(sweep.values), _MAX_SENSITIVITY_WORKERS)
        try:
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                risk_values: list[float] = list(executor.map(_one_step, sweep.values))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        max_val = max(risk_values)
        min_val = min(risk_values)
        abs_max = abs(max_val)
        sensitivity_index = (max_val - min_val) / abs_max if abs_max > 0 else 0.0
        most_sensitive_at = sweep.values[risk_values.index(max_val)]

        return SensitivityResponse(
            parameter_values=list(sweep.values),
            risk_values=risk_values,
            sensitivity_index=sensitivity_index,
            most_sensitive_at=most_sensitive_at,
        )

    return await asyncio.to_thread(_run)
