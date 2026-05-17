"""Pydantic schemas for the growth-fragmentation drone swarm module.

Mathematical context
--------------------
Growth-fragmentation equation (Bertoin 2006):

    ∂_t n(t,x) + ∂_x(τ(x) n(t,x)) = -κ(x) n(t,x)
        + ∫_x^∞ p(x,y) κ(y) n(t,y) dy

where x > 0 is the capability score of a drone sub-swarm.

Assumptions enforced here
--------------------------
[A2] Loss: x₁ + x₂ = β · x_parent, β ∈ (0, 1).
[A5] Initial distribution: Truncated Student-t(ν ≥ 3), x > 0.
     ν ≥ 3 ensures finite 2nd moment (required for W₂ distance).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_STUDENT_T_DF: int = 3  # minimum df for finite 2nd moment
MIN_FRAG_RATE: float = 1e-6  # guard against zero total rate in Gillespie
MIN_LOSS_EFF: float = 1e-3  # β must be strictly positive
MAX_LOSS_EFF: float = 1.0 - 1e-6  # β must be strictly below 1


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class FragConfig(BaseModel):
    """Full configuration for a growth-fragmentation simulation run.

    Parameters
    ----------
    n_particles : int
        Initial number of drone sub-swarms N.
    n_particles_max : int
        Hard cap on particle count (Gillespie safety).
    T : float
        Simulation end time.
    seed : int
        Random seed for reproducibility.
    x_min : float
        Absorption boundary.  Particles with x < x_min are removed.
    x_max : float
        Upper truncation for PDE grid and initial sampler.
    tau_coef : float
        Growth rate coefficient a in τ(x) = a·x.
    kappa_0 : float
        Baseline fragmentation rate κ₀.
    alpha : float
        Fragmentation exponent α in κ(x) = κ₀ · x^α.
    loss_efficiency : float
        β ∈ (0, 1): x₁ + x₂ = β · x_parent  [A2].
    daughter_kernel : str
        Daughter size distribution.  Currently only ``"uniform"`` supported.
    init_dist : str
        Initial particle distribution family.
    init_loc : float
        Location parameter μ for initial distribution.
    init_scale : float
        Scale parameter σ for initial distribution (σ > 0).
    init_df : int
        Degrees of freedom ν ≥ 3 for Truncated Student-t  [A5].
    pde_grid_size : int
        Number of grid cells M for PDE eigenanalysis.
    target_mean : float
        Mean of the Gaussian target distribution μ* (for W₂ cost).
    target_std : float
        Std of the Gaussian target distribution μ* (for W₂ cost).
    """

    n_particles: int = Field(default=500, ge=2)
    n_particles_max: int = Field(default=5000, ge=10)
    T: float = Field(default=20.0, gt=0.0)
    seed: int = Field(default=42)
    x_min: float = Field(default=0.01, gt=0.0)
    x_max: float = Field(default=100.0)
    tau_coef: float = Field(default=0.05, ge=0.0, description="a in τ(x)=a·x")
    kappa_0: float = Field(default=1.0, gt=0.0, description="Baseline fragmentation rate κ₀")
    alpha: float = Field(default=0.0, ge=0.0, description="Exponent in κ(x)=κ₀·x^α")
    loss_efficiency: float = Field(default=0.9, description="β: x₁+x₂=β·x_parent ∈ (0,1)  [A2]")
    daughter_kernel: Literal["uniform"] = "uniform"
    init_dist: Literal["truncated_t", "gamma", "uniform_box"] = "truncated_t"
    init_loc: float = Field(default=10.0, gt=0.0)
    init_scale: float = Field(default=3.0, gt=0.0)
    init_df: int = Field(
        default=MIN_STUDENT_T_DF,
        ge=MIN_STUDENT_T_DF,
        description="Student-t df ≥ 3 (finite 2nd moment)  [A5]",
    )
    pde_grid_size: int = Field(default=200, ge=20)
    target_mean: float = Field(default=5.0, gt=0.0, description="μ* mean for W₂ cost")
    target_std: float = Field(default=2.0, gt=0.0, description="μ* std for W₂ cost")

    @field_validator("loss_efficiency")
    @classmethod
    def _check_beta(cls, v: float) -> float:
        if not (MIN_LOSS_EFF <= v <= MAX_LOSS_EFF):
            raise ValueError(
                f"loss_efficiency must be in ({MIN_LOSS_EFF}, {MAX_LOSS_EFF}), got {v}"
            )
        return v

    @model_validator(mode="after")
    def _check_x_bounds(self) -> FragConfig:
        if self.x_min >= self.x_max:
            raise ValueError(f"x_min={self.x_min} must be < x_max={self.x_max}")
        if self.n_particles >= self.n_particles_max:
            raise ValueError("n_particles must be < n_particles_max")
        return self


# ---------------------------------------------------------------------------
# Simulation snapshots
# ---------------------------------------------------------------------------


class ParticleSnapshot(BaseModel):
    """State of the swarm at a single Gillespie event."""

    time: float
    sizes: list[float]  # capability scores of all sub-swarms
    n_particles: int
    event: Literal["fragmentation", "absorption", "initial"]


# ---------------------------------------------------------------------------
# Eigenanalysis result
# ---------------------------------------------------------------------------


class EigenResult(BaseModel):
    """Result of the GFE spectral analysis.

    Attributes
    ----------
    malthus_lambda : float
        Dominant real eigenvalue λ of the GFE operator.
        For constant κ₀, no growth (τ=0), and no loss (β=1):
        λ = κ₀  (all particles fragment at the same rate, binary split doubles count).
    eigenfunction_x : list[float]
        Grid points x₁, …, x_M.
    eigenfunction_phi : list[float]
        Corresponding eigenvector φ(x) ≥ 0, L¹-normalized.
    converged : bool
        Whether the dominant eigenvalue is well-separated from the second.
    spectral_gap : float
        |λ₁ − λ₂| / |λ₁|  (relative gap).
    """

    malthus_lambda: float
    eigenfunction_x: list[float]
    eigenfunction_phi: list[float]
    converged: bool
    spectral_gap: float


# ---------------------------------------------------------------------------
# Full simulation result
# ---------------------------------------------------------------------------


class FragResult(BaseModel):
    """Output of a complete growth-fragmentation simulation.

    Traceability fields: run_id, config, seed, timestamp.
    """

    run_id: str = Field(default_factory=lambda: str(uuid4()))
    config: FragConfig
    trajectory: list[ParticleSnapshot]
    eigen: EigenResult
    cost_w2: float = Field(description="W₂²(μ_N(T), μ*)")
    score_components: dict[str, float] = Field(
        description="{'coverage_loss': float, 'n_fragments': int, 'mean_size': float}"
    )
    reasons: list[str] = Field(description="Fragmentation event log")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
