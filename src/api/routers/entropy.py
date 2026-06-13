"""Entropy monitoring and regime-change detection endpoints."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Body, HTTPException

from api.schemas.entropy import (
    DetectRequest,
    EntropyRequest,
    EntropyResponse,
    KLRequest,
    KLResponse,
)
from entropy.monitor import compute_entropy, compute_kl, entropy_rate
from schemas import AlertType, EntropyAlert, EntropyReport

router = APIRouter()
_logger = logging.getLogger(__name__)

_ENTROPY_EXAMPLES = {
    "normal_posterior": {
        "summary": "Normal posterior vs Normal prior",
        "value": {
            "posterior": {
                "mean": 0.08,
                "variance": 0.0004,
                "credible_interval_95": [0.04, 0.12],
                "n_evidence": 10,
                "updated_at": "2025-01-01T00:00:00Z",
            },
            "prior": {"distribution": "normal", "params": {"mu": 0.05, "sigma": 0.02}},
        },
    },
}

_KL_EXAMPLES = {
    "normal_shift": {
        "summary": "KL divergence — posterior shifted from prior",
        "value": {
            "posterior": {
                "mean": 0.13,
                "variance": 0.0002,
                "credible_interval_95": [0.10, 0.16],
                "n_evidence": 20,
                "updated_at": "2025-01-01T00:00:00Z",
            },
            "prior": {"distribution": "normal", "params": {"mu": 0.05, "sigma": 0.02}},
        },
    },
}

_DETECT_EXAMPLES = {
    "regime_shift": {
        "summary": "5-step sequence with regime shift at step 3",
        "value": {
            "posteriors": [
                {
                    "mean": 0.05 + i * 0.02,
                    "variance": 0.0004,
                    "credible_interval_95": [0.01 + i * 0.02, 0.09 + i * 0.02],
                    "n_evidence": i + 1,
                    "updated_at": f"2025-01-0{i + 1}T00:00:00Z",
                }
                for i in range(5)
            ],
            "prior": {"distribution": "normal", "params": {"mu": 0.05, "sigma": 0.02}},
            "experiment_id": "exp-demo",
            "kl_threshold": 0.5,
            "entropy_gradient_threshold": 0.1,
            "rolling_window": 2,
        },
    },
}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/entropy",
    response_model=EntropyResponse,
    summary="Shannon entropy of a posterior",
    description=(
        "Compute H(posterior) in nats for a single `PosteriorSummary`. "
        "For a Normal posterior N(μ, σ²), H = 0.5·ln(2πeσ²). "
        "For a Beta posterior Beta(α, β), H = ln B(α,β) − (α−1)ψ(α) − (β−1)ψ(β) + (α+β−2)ψ(α+β)."
    ),
)
def compute_entropy_endpoint(
    req: Annotated[EntropyRequest, Body(openapi_examples=_ENTROPY_EXAMPLES)],  # type: ignore[arg-type]
) -> EntropyResponse:
    """Compute Shannon entropy of a single posterior (nats)."""
    try:
        h = compute_entropy(req.posterior, req.prior)
        return EntropyResponse(entropy=h)
    except ValueError as e:
        _logger.warning("%s", e, exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))


@router.post(
    "/kl",
    response_model=KLResponse,
    summary="KL divergence from prior",
    description=(
        "Compute KL(posterior ‖ prior) in nats. "
        "For Normal: KL = 0.5·[(σ_p/σ_q)² + (μ_q−μ_p)²/σ_q² − 1 + ln(σ_q/σ_p)]. "
        "Returns 0 when posterior equals prior."
    ),
)
def compute_kl_endpoint(
    req: Annotated[KLRequest, Body(openapi_examples=_KL_EXAMPLES)],  # type: ignore[arg-type]
) -> KLResponse:
    """Compute KL divergence KL(posterior || prior) in nats."""
    try:
        kl = compute_kl(req.posterior, req.prior)
        return KLResponse(kl_divergence=kl)
    except (ValueError, KeyError) as e:
        _logger.warning("%s", e, exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))


@router.post(
    "/detect",
    response_model=EntropyReport,
    summary="Full entropy detection pipeline",
    description=(
        "Run the full entropy monitoring pipeline over a sequence of `PosteriorSummary` objects "
        "(Bayesian posteriors through time). Computes:\n\n"
        "- **entropy_series**: H(p_t) for each step t\n"
        "- **kl_series**: KL(p_t ‖ prior) for each step t\n"
        "- **entropy_rate_series**: rolling ΔH over `rolling_window` steps\n\n"
        "Fires **KL_THRESHOLD** alerts when KL > `kl_threshold`, "
        "and **ENTROPY_GRADIENT** alerts when |ΔH| > `entropy_gradient_threshold`."
    ),
)
def detect(
    req: Annotated[DetectRequest, Body(openapi_examples=_DETECT_EXAMPLES)],  # type: ignore[arg-type]
) -> EntropyReport:
    """Run full entropy detection pipeline and return an EntropyReport.

    Fires KL_THRESHOLD alerts when KL > kl_threshold and
    ENTROPY_GRADIENT alerts when |ΔH| > entropy_gradient_threshold.
    """
    try:
        h_series = [compute_entropy(p, req.prior) for p in req.posteriors]
        kl_series = [compute_kl(p, req.prior) for p in req.posteriors]
        rate_series = entropy_rate(h_series, req.rolling_window)

        alerts: list[EntropyAlert] = []

        for step, kl_val in enumerate(kl_series):
            if kl_val > req.kl_threshold:
                alerts.append(
                    EntropyAlert(
                        experiment_id=req.experiment_id,
                        triggered_at=step,
                        alert_type=AlertType.KL_THRESHOLD,
                        metric_value=kl_val,
                        threshold=req.kl_threshold,
                        message=(
                            f"KL {kl_val:.4f} > threshold {req.kl_threshold:.4f} at step {step}"
                        ),
                    )
                )

        step_offset = req.rolling_window
        for i, rate_val in enumerate(rate_series):
            if abs(rate_val) > req.entropy_gradient_threshold:
                step = step_offset + i
                alerts.append(
                    EntropyAlert(
                        experiment_id=req.experiment_id,
                        triggered_at=step,
                        alert_type=AlertType.ENTROPY_GRADIENT,
                        metric_value=rate_val,
                        threshold=req.entropy_gradient_threshold,
                        message=(
                            f"|ΔH| {abs(rate_val):.4f} > threshold "
                            f"{req.entropy_gradient_threshold:.4f} at step {step}"
                        ),
                    )
                )

        alerts.sort(key=lambda a: a.triggered_at)

        return EntropyReport(
            experiment_id=req.experiment_id,
            entropy_series=h_series,
            kl_series=kl_series,
            entropy_rate_series=rate_series,
            alerts=alerts,
        )
    except (ValueError, KeyError) as e:
        _logger.warning("%s", e, exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
