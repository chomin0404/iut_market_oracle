"""Matroid combinatorics endpoints: log-concavity analysis (T1200)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from matroid.log_concavity import compute_log_concave_weights
from schemas import MatroidLogConcavityResult

router = APIRouter()

# Upper bound on n_assets: at n=200 the response contains 201 floats per series (~5 KB).
# Larger values are academically interesting but rarely needed in portfolio analysis.
_N_ASSETS_MAX: int = 200


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------


class LogConcavityRequest(BaseModel):
    n_assets: int = Field(
        ...,
        ge=1,
        le=_N_ASSETS_MAX,
        description="Number of ground elements in the matroid",
    )
    rank_weight: float = Field(
        default=0.8,
        gt=0.0,
        description="Multiplicative weight alpha per element in the independent set",
    )
    corank_weight: float = Field(
        default=1.2,
        gt=0.0,
        description="Multiplicative weight beta per element in the complement",
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/log-concavity", response_model=MatroidLogConcavityResult)
def log_concavity(req: LogConcavityRequest) -> MatroidLogConcavityResult:
    """Compute log-concave subset-size weights for a matroid rank-generating polynomial.

    Returns the normalised probability mass b_k = C(n,k)·alpha^k·beta^(n-k),
    log probabilities ln(b_k), log-concavity checks b_k² ≥ b_{k-1}·b_{k+1},
    and the aggregate ``is_log_concave`` flag.

    The result is always log-concave for valid alpha, beta > 0 (binomial PMF
    property), consistent with June Huh's theorem for graphic matroids.
    """
    try:
        return compute_log_concave_weights(
            n_assets=req.n_assets,
            rank_weight=req.rank_weight,
            corank_weight=req.corank_weight,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
