"""Posterior entropy monitoring for the Entropy Layer (T1000).

Supported posterior families
-----------------------------
Normal  (from T200 Normal-Normal conjugate):
    H = 0.5 * ln(2πe σ²)

Beta    (recovered from PosteriorSummary via method-of-moments):
    H = ln B(α,β) − (α−1)ψ(α) − (β−1)ψ(β) + (α+β−2)ψ(α+β)
    (computed via scipy.stats.beta.entropy)

KL divergence
-------------
Normal  KL(q||p):  ln(σ_p/σ_q) + (σ_q² + (μ_q−μ_p)²) / (2σ_p²) − 0.5
Beta    KL(q||p):  closed-form via log-gamma and digamma
    (computed via scipy.stats.beta; formula in _kl_beta)

Entropy rate
------------
Rolling mean of first differences ΔH_t = H_t − H_{t−1} over a
configurable window.  Output length = len(entropy_series) − window.
Returns an empty list when fewer than 2 observations are available.
"""

from __future__ import annotations

import math

import numpy as np
import scipy.stats as st
from scipy.special import betaln, digamma

from schemas import PosteriorSummary, PriorSpec

# ---------------------------------------------------------------------------
# Family detection
# ---------------------------------------------------------------------------

_NORMAL_FAMILY = frozenset(["normal", "gaussian"])
_BETA_FAMILY = frozenset(["beta"])


def _is_normal(prior: PriorSpec) -> bool:
    return prior.distribution.lower() in _NORMAL_FAMILY


def _is_beta(prior: PriorSpec) -> bool:
    return prior.distribution.lower() in _BETA_FAMILY


# ---------------------------------------------------------------------------
# Entropy — Normal
# ---------------------------------------------------------------------------

_LOG_2PIE: float = math.log(2.0 * math.pi * math.e)


def entropy_normal(variance: float) -> float:
    """Shannon differential entropy of a Normal(μ, σ²) in nats.

    H = 0.5 * ln(2πe σ²)

    Parameters
    ----------
    variance:
        Posterior variance σ².  Must be strictly positive.

    Raises
    ------
    ValueError
        If variance <= 0.
    """
    if variance <= 0.0:
        raise ValueError(f"variance must be strictly positive, got {variance}")
    return 0.5 * (_LOG_2PIE + math.log(variance))


# ---------------------------------------------------------------------------
# Entropy — Beta
# ---------------------------------------------------------------------------


def _beta_params_from_posterior(posterior: PosteriorSummary) -> tuple[float, float]:
    """Recover Beta(α, β) parameters via method-of-moments from a PosteriorSummary.

    Given μ = α/(α+β) and σ² = αβ / ((α+β)²(α+β+1)):
        n = μ(1−μ)/σ² − 1   (total concentration α+β)
        α = μ · n
        β = (1−μ) · n

    Raises
    ------
    ValueError
        If μ is not in (0,1) or the implied concentration n <= 0.
    """
    mu = posterior.mean
    var = posterior.variance

    if not (0.0 < mu < 1.0):
        raise ValueError(
            f"Beta method-of-moments requires mean in (0,1), got {mu}"
        )
    if var <= 0.0:
        raise ValueError(f"variance must be strictly positive, got {var}")

    denom = mu * (1.0 - mu)
    if var >= denom:
        raise ValueError(
            f"variance ({var}) must be < mean*(1-mean) ({denom}) for Beta MoM"
        )

    n = denom / var - 1.0
    if n <= 0.0:
        raise ValueError(f"Implied Beta concentration n={n:.4f} <= 0")

    alpha = mu * n
    beta = (1.0 - mu) * n
    return alpha, beta


def entropy_beta(alpha: float, beta: float) -> float:
    """Shannon differential entropy of a Beta(α,β) in nats.

    Uses scipy.stats.beta.entropy which returns nats by default.
    """
    return float(st.beta.entropy(alpha, beta))


# ---------------------------------------------------------------------------
# Unified entropy dispatcher
# ---------------------------------------------------------------------------


def compute_entropy(posterior: PosteriorSummary, prior: PriorSpec) -> float:
    """Compute Shannon entropy of the posterior in nats.

    Dispatches to the appropriate closed-form formula based on
    prior.distribution.

    Parameters
    ----------
    posterior:
        Current posterior summary.
    prior:
        Prior specification used to identify the conjugate family.

    Returns
    -------
    float
        Entropy in nats.

    Raises
    ------
    ValueError
        For unsupported distribution families or invalid parameters.
    """
    if _is_normal(prior):
        return entropy_normal(posterior.variance)
    if _is_beta(prior):
        alpha, beta = _beta_params_from_posterior(posterior)
        return entropy_beta(alpha, beta)
    raise ValueError(
        f"Unsupported distribution family '{prior.distribution}'. "
        "Supported: normal, gaussian, beta."
    )


# ---------------------------------------------------------------------------
# KL divergence — Normal KL(posterior || prior)
# ---------------------------------------------------------------------------


def kl_normal(
    mu_q: float,
    var_q: float,
    mu_p: float,
    var_p: float,
) -> float:
    """KL divergence KL(q || p) for Normal distributions in nats.

    KL = ln(σ_p/σ_q) + (σ_q² + (μ_q − μ_p)²) / (2σ_p²) − 0.5

    Parameters
    ----------
    mu_q, var_q:
        Mean and variance of q (posterior).
    mu_p, var_p:
        Mean and variance of p (prior reference).

    Raises
    ------
    ValueError
        If either variance is not strictly positive.
    """
    if var_q <= 0.0 or var_p <= 0.0:
        raise ValueError(
            f"Both variances must be strictly positive; "
            f"got var_q={var_q}, var_p={var_p}"
        )
    return (
        0.5 * math.log(var_p / var_q)
        + (var_q + (mu_q - mu_p) ** 2) / (2.0 * var_p)
        - 0.5
    )


# ---------------------------------------------------------------------------
# KL divergence — Beta KL(posterior || prior)
# ---------------------------------------------------------------------------


def kl_beta(
    alpha_q: float,
    beta_q: float,
    alpha_p: float,
    beta_p: float,
) -> float:
    """KL divergence KL(q || p) for Beta distributions in nats.

    KL = ln B(α_p, β_p) − ln B(α_q, β_q)
         + (α_q − α_p) ψ(α_q)
         + (β_q − β_p) ψ(β_q)
         + (α_p − α_q + β_p − β_q) ψ(α_q + β_q)

    where B(·) is the Beta function and ψ is the digamma function.
    """
    return float(
        betaln(alpha_p, beta_p)
        - betaln(alpha_q, beta_q)
        + (alpha_q - alpha_p) * digamma(alpha_q)
        + (beta_q - beta_p) * digamma(beta_q)
        + (alpha_p - alpha_q + beta_p - beta_q) * digamma(alpha_q + beta_q)
    )


# ---------------------------------------------------------------------------
# Unified KL dispatcher
# ---------------------------------------------------------------------------


def compute_kl(
    posterior: PosteriorSummary,
    prior: PriorSpec,
) -> float:
    """KL divergence KL(posterior || prior) in nats.

    Parameters
    ----------
    posterior:
        Current posterior summary (q).
    prior:
        Prior specification (p).  The prior mean and variance are
        extracted from prior.params keys:
            Normal: "mean" (default 0.0), "std" or "variance"
            Beta:   "alpha", "beta"

    Returns
    -------
    float
        KL divergence in nats.

    Raises
    ------
    ValueError
        For unsupported families or missing required params keys.
    KeyError
        If required param keys are absent from prior.params.
    """
    if _is_normal(prior):
        mu_p = float(prior.params.get("mean", 0.0))
        if "std" in prior.params:
            var_p = float(prior.params["std"]) ** 2
        elif "variance" in prior.params:
            var_p = float(prior.params["variance"])
        else:
            raise KeyError(
                "Normal prior must have 'std' or 'variance' in params; "
                f"got keys {list(prior.params.keys())}"
            )
        return kl_normal(posterior.mean, posterior.variance, mu_p, var_p)

    if _is_beta(prior):
        alpha_p = float(prior.params["alpha"])
        beta_p = float(prior.params["beta"])
        alpha_q, beta_q = _beta_params_from_posterior(posterior)
        return kl_beta(alpha_q, beta_q, alpha_p, beta_p)

    raise ValueError(
        f"Unsupported distribution family '{prior.distribution}'. "
        "Supported: normal, gaussian, beta."
    )


# ---------------------------------------------------------------------------
# Entropy rate (rolling window)
# ---------------------------------------------------------------------------


def entropy_rate(
    entropy_series: list[float],
    window: int,
) -> list[float]:
    """Rolling mean of first differences ΔH_t = H_t − H_{t−1}.

    Parameters
    ----------
    entropy_series:
        Sequence of entropy values H_0, H_1, …, H_T.
    window:
        Rolling window length.  Must be >= 1.

    Returns
    -------
    list[float]
        Rolling-mean ΔH values.  Length = max(0, len(entropy_series) − window).
        Returns an empty list when fewer than 2 entropy values are supplied.

    Raises
    ------
    ValueError
        If window < 1.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if len(entropy_series) < 2:
        return []

    diffs = np.diff(np.array(entropy_series, dtype=float))
    # Convolve with a uniform kernel of length `window` (valid mode).
    if len(diffs) < window:
        # Not enough differences for even one full window — return raw diffs.
        return diffs.tolist()

    kernel = np.ones(window, dtype=float) / window
    return np.convolve(diffs, kernel, mode="valid").tolist()
