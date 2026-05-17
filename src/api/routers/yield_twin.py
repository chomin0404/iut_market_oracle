"""Process Yield Twin API — surrogate + Bayesian optimisation + D-Optimal DOE (T1600).

Endpoints:
    POST /yield-twin/recommend   — one-shot: fit on observations, return next recommendation
    POST /yield-twin/report      — full report including GP diagnostics and factor specs
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, model_validator

from schemas import DOERecommendation, ExperimentPoint, FactorSpec, YieldTwinReport
from yield_twin.twin import YieldTwinConfig, recommend_next_experiment

router = APIRouter()

# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


def _build_report(req: YieldTwinRequest) -> YieldTwinReport:
    """Shared logic: fit twin, return full report."""
    try:
        config = YieldTwinConfig(
            factor_specs=req.factor_specs,
            n_candidates=req.n_candidates,
            gp_n_restarts=req.gp_n_restarts,
            ei_xi=req.ei_xi,
            random_seed=req.random_seed,
        )
        _ = config  # validated above; use convenience function below
        return recommend_next_experiment(
            factor_specs=req.factor_specs,
            observations=req.observations,
            random_seed=req.random_seed,
            n_candidates=req.n_candidates,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/recommend", response_model=DOERecommendation)
def recommend(req: YieldTwinRequest) -> DOERecommendation:
    """Recommend the next experiment to run (T1600).

    Fits a GP surrogate (ARD RBF) on past observations and fuses
    Expected Improvement (EI) with D-Optimal leverage to select the
    most informative next design point.

    ### Acquisition modes

    | Phase | Condition | Strategy |
    |---|---|---|
    | `doe_explore`  | n < n_min        | D-optimal dominates — fill design space |
    | `fused`        | n_min ≤ n < 2·n_min | Weighted blend of EI and D-leverage |
    | `ei_exploit`   | n ≥ 2·n_min      | EI dominates — exploit surrogate |

    where n_min = max(2·d + 1, 5) and d = number of factors.

    ### Factor normalisation
    All factors are normalised to [0, 1] internally.
    Returned `factors` are in original physical units.
    """
    report = _build_report(req)
    return report.recommendation


@router.post("/report", response_model=YieldTwinReport)
def report(req: YieldTwinRequest) -> YieldTwinReport:
    """Full Process Yield Twin optimisation report (T1600).

    Returns the next-experiment recommendation plus:
    - Best yield observed and corresponding factor settings
    - GP surrogate LOO cross-validated R² (None when < 3 observations)
    - GP hyperparameters: signal variance, noise variance, ARD length scales
    - Factor specs for downstream traceability

    ### GP surrogate
    ARD RBF kernel: k(x, x') = σ² exp(−Σ (xᵢ − x'ᵢ)² / (2 lᵢ²))
    Hyperparameters optimised by log-marginal-likelihood maximisation.
    """
    return _build_report(req)
