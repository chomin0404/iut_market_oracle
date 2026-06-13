"""Request/response schemas for the Digital Twin router (T800 / T1100)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from schemas import DigitalTwinState, PosteriorSummary, PriorSpec
from twin.simulator import DEFAULT_DT

# Maximum n_steps accepted by T1100 endpoints; keeps JSON response ≤ ~150 KB
_N_STEPS_MAX: int = 5000
# Upper bounds for SimulateRequest to prevent OOM
_HORIZON_MAX: int = 5_000
_N_SAMPLES_MAX: int = 10_000


class SimulateRequest(BaseModel):
    initial_state: DigitalTwinState
    horizon: int = Field(..., ge=1, le=_HORIZON_MAX, description="Number of forward time steps")
    n_samples: int = Field(
        ..., ge=1, le=_N_SAMPLES_MAX, description="Number of Monte Carlo trajectories"
    )
    process_noise_std: float = Field(..., ge=0.0, description="Isotropic process noise σ")
    random_seed: int = Field(..., description="RNG seed for reproducibility")
    dt: float = Field(default=DEFAULT_DT, gt=0.0, description="Time step in years")
    transition_matrix: list[list[float]] | None = Field(
        default=None,
        description="(d × d) state transition matrix F. Uses local linear trend if omitted.",
    )


_OBSERVATIONS_MAX: int = 10_000  # prevents large payload DoS


class CalibrateRequest(BaseModel):
    observations: list[float] = Field(..., min_length=1, max_length=_OBSERVATIONS_MAX)
    prior: PriorSpec
    experiment_id: str
    obs_precision: float = Field(default=1.0, gt=0.0, description="Precision τ = 1/σ_obs²")


class CalibrateResponse(BaseModel):
    posterior: PosteriorSummary
    state: DigitalTwinState


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
