"""Entropy monitoring and regime-change detection endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from entropy.monitor import compute_entropy, compute_kl, entropy_rate
from schemas import AlertType, EntropyAlert, EntropyReport, PosteriorSummary, PriorSpec

router = APIRouter()


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class EntropyRequest(BaseModel):
    posterior: PosteriorSummary
    prior: PriorSpec


class EntropyResponse(BaseModel):
    entropy: float


class KLRequest(BaseModel):
    posterior: PosteriorSummary
    prior: PriorSpec


class KLResponse(BaseModel):
    kl_divergence: float


class DetectRequest(BaseModel):
    posteriors: list[PosteriorSummary] = Field(..., min_length=1)
    prior: PriorSpec
    experiment_id: str
    kl_threshold: float = Field(default=0.5, gt=0.0)
    entropy_gradient_threshold: float = Field(default=0.1, gt=0.0)
    rolling_window: int = Field(default=3, ge=1)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/entropy", response_model=EntropyResponse)
def compute_entropy_endpoint(req: EntropyRequest) -> EntropyResponse:
    """Compute Shannon entropy of a single posterior (nats)."""
    try:
        h = compute_entropy(req.posterior, req.prior)
        return EntropyResponse(entropy=h)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/kl", response_model=KLResponse)
def compute_kl_endpoint(req: KLRequest) -> KLResponse:
    """Compute KL divergence KL(posterior || prior) in nats."""
    try:
        kl = compute_kl(req.posterior, req.prior)
        return KLResponse(kl_divergence=kl)
    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/detect", response_model=EntropyReport)
def detect(req: DetectRequest) -> EntropyReport:
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
                            f"KL {kl_val:.4f} > threshold {req.kl_threshold:.4f}"
                            f" at step {step}"
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
        raise HTTPException(status_code=400, detail=str(e))
