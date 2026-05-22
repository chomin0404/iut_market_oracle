"""Request schemas for the Process Yield Twin router (T1600)."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from schemas import ExperimentPoint, FactorSpec

_N_CANDIDATES_MAX: int = 5000


class YieldTwinRequest(BaseModel):
    """Request body shared by /recommend and /report.

    factor_specs:   Definitions of process factors (name, low, high in physical units).
    observations:   Past experiment results. Points without yield_obs are ignored.
    random_seed:    RNG seed for Latin-Hypercube sampling and GP initialisation.
    n_candidates:   LHS candidate set size (10 – 5000).
    gp_n_restarts:  GP hyperparameter optimisation restarts.
    ei_xi:          Exploration bonus ξ for Expected Improvement.
    """

    factor_specs: list[FactorSpec] = Field(
        ...,
        min_length=1,
        description="Factor definitions (name, low, high) — at least 1 required",
    )
    observations: list[ExperimentPoint] = Field(
        default_factory=list,
        description="Past experiments. Points without yield_obs are skipped.",
    )
    random_seed: int = Field(default=42, description="RNG seed for reproducibility")
    n_candidates: int = Field(
        default=2000,
        ge=2,
        le=_N_CANDIDATES_MAX,
        description="Latin-Hypercube candidate set size",
    )
    gp_n_restarts: int = Field(
        default=5,
        ge=1,
        le=50,
        description="GP hyperparameter optimisation random restarts",
    )
    ei_xi: float = Field(
        default=0.01,
        gt=0.0,
        description="Exploration bonus ξ for Expected Improvement",
    )

    @model_validator(mode="after")
    def _validate_factor_names(self) -> YieldTwinRequest:
        names = [fs.name for fs in self.factor_specs]
        if len(names) != len(set(names)):
            raise ValueError("factor_specs must have unique names")
        observed_names = {k for obs in self.observations for k in obs.factors}
        unknown = observed_names - set(names)
        if unknown:
            raise ValueError(f"observations reference unknown factor names: {unknown}")
        return self
