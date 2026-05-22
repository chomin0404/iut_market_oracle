"""Process Yield Twin API — surrogate + Bayesian optimisation + D-Optimal DOE (T1600).

Endpoints:
    POST /yield-twin/recommend   — one-shot: fit on observations, return next recommendation
    POST /yield-twin/report      — full report including GP diagnostics and factor specs
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.schemas.yield_twin import YieldTwinRequest
from schemas import DOERecommendation, YieldTwinReport
from yield_twin.twin import YieldTwinConfig, recommend_next_experiment

router = APIRouter()


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
