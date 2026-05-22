"""Matroid combinatorics endpoints: log-concavity analysis (T1200)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.schemas.matroid import LogConcavityRequest
from matroid.log_concavity import compute_log_concave_weights
from schemas import MatroidLogConcavityResult

router = APIRouter()


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
