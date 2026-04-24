"""Parameter calibration for the Digital Twin via Bayesian updating (T800).

Given a sequence of scalar observations (e.g., observed annual growth rates),
the calibrator estimates the underlying mean parameter μ using the
Normal-Normal conjugate updater from src/bayesian/updater.py.

Calibration model
-----------------
Let μ be the true underlying parameter (e.g., mean growth rate).

Prior:
    μ ~ N(μ₀, σ₀²)
    encoded as PriorSpec(distribution="normal", params={"mu": μ₀, "sigma": σ₀})

Likelihood per observation y_i:
    y_i ~ N(μ, σ_obs²),  precision τ_obs = 1/σ_obs²
    (caller supplies obs_precision = τ_obs)

After observing y_1, …, y_n (Normal-Normal conjugate):
    τ_n  = 1/σ₀² + n · τ_obs
    μ_n  = (μ₀/σ₀² + τ_obs · Σ y_i) / τ_n
    σ_n² = 1/τ_n

The calibrated DigitalTwinState embeds the result as:
    state_vector   = [μ_n]
    state_labels   = ["mu"]
    param_snapshot = {"mu": μ_n, "sigma": σ_n}
    step           = n  (number of observations consumed)
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

from bayesian.updater import update
from schemas import (
    DigitalTwinState,
    Evidence,
    EvidenceKind,
    PosteriorSummary,
    PriorSpec,
)


def calibrate(
    observations: list[float],
    prior: PriorSpec,
    experiment_id: str,
    obs_precision: float = 1.0,
) -> tuple[PosteriorSummary, DigitalTwinState]:
    """Estimate a latent parameter from observations via Bayesian updating.

    Parameters
    ----------
    observations:
        Scalar time series (e.g., period growth rates).  Each value is treated
        as an independent draw from N(μ, 1/obs_precision).
        An empty list returns the prior predictive summary unchanged.
    prior:
        Normal prior over parameter μ.
        Must satisfy distribution == "normal" with keys "mu" and "sigma".
    experiment_id:
        Experiment identifier matching r"^exp-\\d{3}$".
    obs_precision:
        Precision τ_obs = 1/σ_obs² applied uniformly to every observation.
        Larger values express more trust in the observations relative to the prior.

    Returns
    -------
    posterior : PosteriorSummary
        Posterior statistics after incorporating all observations.
    state : DigitalTwinState
        Single-dimensional state snapshot reflecting the calibrated parameter.

    Raises
    ------
    ValueError
        If prior.distribution != "normal" or obs_precision <= 0.
    """
    if prior.distribution.lower() != "normal":
        raise ValueError(
            f"calibrate requires distribution='normal', got '{prior.distribution}'"
        )
    if obs_precision <= 0.0:
        raise ValueError(f"obs_precision must be > 0, got {obs_precision}")

    evidence_list: list[Evidence] = [
        Evidence(
            source="calibration_observation",
            kind=EvidenceKind.OBSERVATION,
            value=obs,
            weight=obs_precision,
        )
        for obs in observations
    ]

    posterior = update(prior, evidence_list)
    posterior_sigma = math.sqrt(posterior.variance)

    state = DigitalTwinState(
        experiment_id=experiment_id,
        state_vector=[posterior.mean],
        state_labels=["mu"],
        param_snapshot={
            "mu": posterior.mean,
            "sigma": posterior_sigma,
        },
        step=len(observations),
        timestamp=datetime.now(UTC),
    )

    return posterior, state
