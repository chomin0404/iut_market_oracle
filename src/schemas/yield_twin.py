"""T1600 Process Yield Twin schemas."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class FactorSpec(BaseModel):
    """Specification of one process factor (input variable).

    Values are assumed to lie in [low, high] in physical units.
    The twin normalises to [0, 1] internally for all GP computations.
    """

    name: str = Field(..., min_length=1)
    low: float = Field(..., description="Lower bound of factor range (physical units)")
    high: float = Field(..., description="Upper bound of factor range (physical units)")

    @model_validator(mode="after")
    def low_lt_high(self) -> FactorSpec:
        if not self.low < self.high:
            raise ValueError(f"low ({self.low}) must be strictly less than high ({self.high})")
        return self


class ExperimentPoint(BaseModel):
    """Single design point with optional observed yield.

    factors:   Mapping of factor name → value in physical units.
    yield_obs: Observed yield ∈ [0, 1] (e.g. fraction of conforming parts),
               or None if the experiment has not yet been run.
    """

    factors: dict[str, float]
    yield_obs: float | None = Field(None, ge=0.0, le=1.0)


class DOERecommendation(BaseModel):
    """Next experiment recommended by the Process Yield Twin.

    factors:            Recommended factor settings in physical units.
    expected_improvement: EI(x*) from the GP surrogate [yield units].
    d_leverage:         φ(x*)ᵀ M⁻¹ φ(x*) — D-optimal leverage score.
    fusion_score:       Weighted combination of normalised EI and d_leverage.
    predicted_yield:    GP posterior mean at x* (in [0, 1]).
    predicted_std:      GP posterior standard deviation at x*.
    acquisition_mode:   "doe_explore" | "fused" | "ei_exploit"
    n_observations:     Number of observations available when recommendation was made.
    """

    factors: dict[str, float]
    expected_improvement: float = Field(..., ge=0.0)
    d_leverage: float = Field(..., ge=0.0)
    fusion_score: float = Field(..., ge=0.0)
    predicted_yield: float = Field(..., ge=0.0, le=1.0)
    predicted_std: float = Field(..., ge=0.0)
    acquisition_mode: Literal["doe_explore", "fused", "ei_exploit"]
    n_observations: int = Field(..., ge=0)


class YieldTwinReport(BaseModel):
    """Full optimisation report from the Process Yield Twin (T1600).

    n_observations:      Number of completed experiments.
    best_yield_observed: Maximum observed yield so far (None if no data).
    best_factors:        Factor settings that achieved best_yield_observed.
    surrogate_loocv_r2:  GP leave-one-out cross-validated R² (None if < 3 obs).
    recommendation:      Next experiment to run.
    gp_hyperparams:      Fitted GP hyperparameters:
                           signal_var, noise_var, length_scale_<name> per factor.
    factor_specs:        Factor definitions used for this run.
    """

    n_observations: int = Field(..., ge=0)
    best_yield_observed: float | None = Field(None, ge=0.0, le=1.0)
    best_factors: dict[str, float] | None = None
    surrogate_loocv_r2: float | None = Field(
        None, description="GP LOO cross-validated R² (None when < 3 observations)"
    )
    recommendation: DOERecommendation
    gp_hyperparams: dict[str, float] = Field(
        default_factory=dict,
        description="signal_var, noise_var, length_scale_<factor_name>",
    )
    factor_specs: list[FactorSpec]
    produced_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
