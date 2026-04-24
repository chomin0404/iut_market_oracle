"""Digital Twin endpoints: Monte Carlo simulation, Bayesian calibration,
regime-switching price simulation, and market evolution (T800 / T1100)."""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from schemas import (
    DigitalTwinState,
    MarketEvolutionResult,
    PosteriorSummary,
    PriorSpec,
    RegimeSwitchResult,
    SimulationResult,
)
from twin.calibrator import calibrate
from twin.regime_simulator import simulate_market_evolution, simulate_regime_switching
from twin.simulator import DEFAULT_DT, simulate

router = APIRouter()

# Maximum n_steps accepted by T1100 endpoints; keeps JSON response ≤ ~150 KB
_N_STEPS_MAX: int = 5000


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class SimulateRequest(BaseModel):
    initial_state: DigitalTwinState
    horizon: int = Field(..., ge=1, description="Number of forward time steps")
    n_samples: int = Field(..., ge=1, description="Number of Monte Carlo trajectories")
    process_noise_std: float = Field(..., ge=0.0, description="Isotropic process noise σ")
    random_seed: int = Field(..., description="RNG seed for reproducibility")
    dt: float = Field(default=DEFAULT_DT, gt=0.0, description="Time step in years")
    transition_matrix: list[list[float]] | None = Field(
        default=None,
        description="(d × d) state transition matrix F. Uses local linear trend if omitted.",
    )


class CalibrateRequest(BaseModel):
    observations: list[float]
    prior: PriorSpec
    experiment_id: str
    obs_precision: float = Field(default=1.0, gt=0.0, description="Precision τ = 1/σ_obs²")


class CalibrateResponse(BaseModel):
    posterior: PosteriorSummary
    state: DigitalTwinState


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/simulate", response_model=SimulationResult)
def simulate_endpoint(req: SimulateRequest) -> SimulationResult:
    """Run Monte Carlo forward simulation from an initial DigitalTwinState."""
    rng = np.random.default_rng(req.random_seed)
    F: np.ndarray | None = None
    if req.transition_matrix is not None:
        F = np.array(req.transition_matrix, dtype=float)
    try:
        return simulate(
            initial_state=req.initial_state,
            horizon=req.horizon,
            n_samples=req.n_samples,
            process_noise_std=req.process_noise_std,
            rng=rng,
            dt=req.dt,
            transition_matrix=F,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/calibrate", response_model=CalibrateResponse)
def calibrate_endpoint(req: CalibrateRequest) -> CalibrateResponse:
    """Estimate a latent parameter from observations via Normal-Normal conjugate updating."""
    try:
        posterior, state = calibrate(
            observations=req.observations,
            prior=req.prior,
            experiment_id=req.experiment_id,
            obs_precision=req.obs_precision,
        )
        return CalibrateResponse(posterior=posterior, state=state)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------------------------------------------------------------
# T1100  Regime-switching and market evolution request / response models
# ---------------------------------------------------------------------------


class RegimeSimulateRequest(BaseModel):
    n_steps: int = Field(..., ge=1, le=_N_STEPS_MAX, description="Number of time steps")
    initial_price: float = Field(default=100.0, gt=0.0, description="Starting asset price")
    p_stay_normal: float = Field(default=0.95, gt=0.0, lt=1.0, description="P(regime=0 | prev=0)")
    p_stay_volatile: float = Field(default=0.90, gt=0.0, lt=1.0, description="P(regime=1 | prev=1)")
    random_seed: int = Field(..., description="RNG seed for reproducibility")


class RegimeSimulateSummaryResponse(BaseModel):
    """Compact summary of a regime-switching simulation (no full price series)."""

    n_steps: int
    final_price: float
    min_price: float
    max_price: float
    regime_0_fraction: float = Field(..., description="Fraction of steps in normal regime")
    regime_1_fraction: float = Field(..., description="Fraction of steps in volatile regime")
    regime_switch_count: int = Field(..., description="Number of regime transitions")


class MarketEvolveRequest(BaseModel):
    n_steps: int = Field(..., ge=1, le=_N_STEPS_MAX, description="Number of time steps")
    gamma_alpha: float = Field(default=2.0, gt=0.0, description="Gamma shape parameter alpha")
    gamma_beta: float = Field(
        default=1.0, gt=0.0, description="Gamma rate parameter beta (scale = 1/beta)"
    )
    random_seed: int = Field(..., description="RNG seed for reproducibility")


# ---------------------------------------------------------------------------
# T1100  Endpoints
# ---------------------------------------------------------------------------


@router.post("/regime-simulate", response_model=RegimeSwitchResult)
def regime_simulate(req: RegimeSimulateRequest) -> RegimeSwitchResult:
    """Run a 2-state Markov regime-switching price simulation.

    Returns the full price and regime series (length = n_steps).
    Use ``/twin/regime-simulate/summary`` for a compact response.
    """
    try:
        return simulate_regime_switching(
            n_steps=req.n_steps,
            initial_price=req.initial_price,
            p_stay_normal=req.p_stay_normal,
            p_stay_volatile=req.p_stay_volatile,
            rng=np.random.default_rng(req.random_seed),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/regime-simulate/summary", response_model=RegimeSimulateSummaryResponse)
def regime_simulate_summary(req: RegimeSimulateRequest) -> RegimeSimulateSummaryResponse:
    """Run a regime-switching simulation and return a compact summary.

    Computes the same simulation as ``/twin/regime-simulate`` but returns only
    aggregate statistics (final/min/max price, regime fractions, switch count).
    """
    try:
        result = simulate_regime_switching(
            n_steps=req.n_steps,
            initial_price=req.initial_price,
            p_stay_normal=req.p_stay_normal,
            p_stay_volatile=req.p_stay_volatile,
            rng=np.random.default_rng(req.random_seed),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    prices = result.prices
    regimes = result.regimes
    n = result.n_steps
    n_volatile = sum(r == 1 for r in regimes)
    switch_count = sum(1 for t in range(1, n) if regimes[t] != regimes[t - 1])

    return RegimeSimulateSummaryResponse(
        n_steps=n,
        final_price=prices[-1],
        min_price=min(prices),
        max_price=max(prices),
        regime_0_fraction=(n - n_volatile) / n,
        regime_1_fraction=n_volatile / n,
        regime_switch_count=switch_count,
    )


@router.post("/market-evolve", response_model=MarketEvolutionResult)
def market_evolve(req: MarketEvolveRequest) -> MarketEvolutionResult:
    """Simulate market size evolution via Gamma-Poisson mixture and sigmoid adoption.

    Returns new_customers, cumulative_base, sigmoid_factor, and market_capture
    series of length n_steps.
    """
    try:
        return simulate_market_evolution(
            n_steps=req.n_steps,
            gamma_alpha=req.gamma_alpha,
            gamma_beta=req.gamma_beta,
            rng=np.random.default_rng(req.random_seed),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
