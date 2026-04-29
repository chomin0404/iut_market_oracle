"""Wasserstein-2 distance for posterior distribution monitoring (Villani 2003, 2008).

Mathematical basis
------------------
2-Wasserstein distance (Villani, "Optimal Transport: Old and New", 2008)
    W₂(μ, ν)² = inf_{γ ∈ Γ(μ,ν)} ∫∫ |x − y|² dγ(x, y)

    The infimum is over all couplings γ whose marginals are μ and ν.

1-D closed form (Villani, Theorem 2.18)
    For any two distributions on ℝ with quantile functions Q_μ, Q_ν:

        W₂(μ, ν)² = ∫₀¹ (Q_μ(t) − Q_ν(t))² dt

    This follows from the Hoeffding–Fréchet coupling: the optimal transport
    plan on the real line is the monotone (quantile) coupling.

Normal closed form
    For μ = N(μ₁, σ₁²) and ν = N(μ₂, σ₂²), Q(t) = μ + σ Φ⁻¹(t), so:

        W₂² = (μ₁ − μ₂)² + (σ₁ − σ₂)²

    Derivation:
        ∫₀¹ [(μ₁ + σ₁ Φ⁻¹(t)) − (μ₂ + σ₂ Φ⁻¹(t))]² dt
        = (μ₁−μ₂)² + 2(μ₁−μ₂)(σ₁−σ₂) ∫₀¹ Φ⁻¹(t) dt + (σ₁−σ₂)² ∫₀¹ (Φ⁻¹(t))² dt
        = (μ₁−μ₂)² + 0 + (σ₁−σ₂)²   [∫Φ⁻¹(t)dt = E[Z] = 0, ∫(Φ⁻¹(t))²dt = Var[Z] = 1]

Beta numerical form
    No closed form exists.  The quantile coupling integral is evaluated by
    adaptive Gauss-Kronrod quadrature (scipy.integrate.quad, limit=_W2_QUAD_LIMIT):

        W₂(Beta(α₁,β₁), Beta(α₂,β₂))² ≈ ∫₀¹ (Q_{α₁,β₁}(t) − Q_{α₂,β₂}(t))² dt

    The Beta PPF is smooth on (0, 1) except when α < 1 or β < 1, where the
    density diverges at 0 or 1.  The integrand remains square-integrable and
    scipy.integrate.quad handles this via adaptive subdivision.

Advantages over KL divergence for regime monitoring
    • Symmetric:   W₂(μ,ν) = W₂(ν,μ)  (KL is asymmetric)
    • Metric:      W₂ satisfies the triangle inequality
    • Finite:      W₂ is finite for any two distributions with finite 2nd moments
    • Geometric:   W₂ measures displacement in the underlying space, not
                   information-theoretic divergence — natural for tracking
                   how far a belief has shifted from its reference position

Reference
---------
Villani C. (2003). "Topics in Optimal Transportation." AMS.
Villani C. (2008). "Optimal Transport: Old and New." Springer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import scipy.integrate as si
import scipy.stats as st

from entropy.monitor import _beta_params_from_posterior, _is_beta, _is_normal
from schemas import PosteriorSummary, PriorSpec

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_W2_QUAD_LIMIT: int = 200  # scipy.integrate.quad max subdivision count for Beta


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------


@dataclass
class W2Result:
    """Wasserstein-2 distance between a posterior and a reference distribution.

    Attributes
    ----------
    distance : float
        W₂(posterior, reference) ≥ 0.  Zero iff both distributions are equal.
    squared : float
        W₂² = distance².  The natural cost in optimal transport theory.
    family : str
        Distribution family used: ``"normal"`` or ``"beta"``.
    """

    distance: float
    squared: float
    family: str


# ---------------------------------------------------------------------------
# Core: Normal — closed-form
# ---------------------------------------------------------------------------


def w2_normal(
    mu_q: float,
    var_q: float,
    mu_p: float,
    var_p: float,
) -> float:
    """Wasserstein-2 distance between two Normal distributions.

    W₂(N(μ_q, σ_q²), N(μ_p, σ_p²)) = √[(μ_q − μ_p)² + (σ_q − σ_p)²]

    Parameters
    ----------
    mu_q, var_q : float
        Mean and variance of q (e.g. posterior).
    mu_p, var_p : float
        Mean and variance of p (e.g. prior reference).

    Returns
    -------
    float
        W₂ distance ≥ 0.

    Raises
    ------
    ValueError
        If either variance is not strictly positive.
    """
    if var_q <= 0.0:
        raise ValueError(f"var_q must be strictly positive, got {var_q}")
    if var_p <= 0.0:
        raise ValueError(f"var_p must be strictly positive, got {var_p}")

    sigma_q = math.sqrt(var_q)
    sigma_p = math.sqrt(var_p)
    squared = (mu_q - mu_p) ** 2 + (sigma_q - sigma_p) ** 2
    return math.sqrt(squared)


def w2_normal_squared(
    mu_q: float,
    var_q: float,
    mu_p: float,
    var_p: float,
) -> float:
    """Wasserstein-2 distance *squared* between two Normal distributions.

    W₂²(N(μ_q, σ_q²), N(μ_p, σ_p²)) = (μ_q − μ_p)² + (σ_q − σ_p)²

    Cheaper to compute than w2_normal when only the squared value is needed.
    Raises ValueError for non-positive variances.
    """
    if var_q <= 0.0:
        raise ValueError(f"var_q must be strictly positive, got {var_q}")
    if var_p <= 0.0:
        raise ValueError(f"var_p must be strictly positive, got {var_p}")

    sigma_q = math.sqrt(var_q)
    sigma_p = math.sqrt(var_p)
    return (mu_q - mu_p) ** 2 + (sigma_q - sigma_p) ** 2


# ---------------------------------------------------------------------------
# Core: Beta — numerical quantile coupling
# ---------------------------------------------------------------------------


def w2_beta(
    alpha_q: float,
    beta_q: float,
    alpha_p: float,
    beta_p: float,
) -> float:
    """Wasserstein-2 distance between two Beta distributions.

    Computed via the 1-D quantile coupling formula:

        W₂² = ∫₀¹ (Q_{α_q,β_q}(t) − Q_{α_p,β_p}(t))² dt

    Numerically integrated with scipy.integrate.quad (adaptive Gauss-Kronrod).

    Parameters
    ----------
    alpha_q, beta_q : float
        Shape parameters of q (e.g. posterior Beta).  Both must be > 0.
    alpha_p, beta_p : float
        Shape parameters of p (e.g. prior reference Beta).  Both must be > 0.

    Returns
    -------
    float
        W₂ distance ≥ 0.

    Raises
    ------
    ValueError
        If any shape parameter is not strictly positive.
    """
    for name, val in [
        ("alpha_q", alpha_q),
        ("beta_q", beta_q),
        ("alpha_p", alpha_p),
        ("beta_p", beta_p),
    ]:
        if val <= 0.0:
            raise ValueError(f"{name} must be strictly positive, got {val}")

    dist_q = st.beta(alpha_q, beta_q)
    dist_p = st.beta(alpha_p, beta_p)

    def _integrand(t: float) -> float:
        return (dist_q.ppf(t) - dist_p.ppf(t)) ** 2

    squared, _ = si.quad(_integrand, 0.0, 1.0, limit=_W2_QUAD_LIMIT)
    return math.sqrt(max(0.0, squared))


def w2_beta_squared(
    alpha_q: float,
    beta_q: float,
    alpha_p: float,
    beta_p: float,
) -> float:
    """Wasserstein-2 distance *squared* between two Beta distributions.

    Same as ``w2_beta`` but returns W₂² directly (avoids one sqrt).
    Raises ValueError for non-positive shape parameters.
    """
    for name, val in [
        ("alpha_q", alpha_q),
        ("beta_q", beta_q),
        ("alpha_p", alpha_p),
        ("beta_p", beta_p),
    ]:
        if val <= 0.0:
            raise ValueError(f"{name} must be strictly positive, got {val}")

    dist_q = st.beta(alpha_q, beta_q)
    dist_p = st.beta(alpha_p, beta_p)

    def _integrand(t: float) -> float:
        return (dist_q.ppf(t) - dist_p.ppf(t)) ** 2

    squared, _ = si.quad(_integrand, 0.0, 1.0, limit=_W2_QUAD_LIMIT)
    return max(0.0, squared)


# ---------------------------------------------------------------------------
# Unified dispatcher
# ---------------------------------------------------------------------------


def _prior_normal_params(prior: PriorSpec) -> tuple[float, float]:
    """Extract (mu_p, var_p) from a Normal PriorSpec."""
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
    return mu_p, var_p


def w2_posterior(
    posterior: PosteriorSummary,
    prior: PriorSpec,
) -> W2Result:
    """Wasserstein-2 distance between a posterior and its prior reference.

    Dispatches to the appropriate formula based on ``prior.distribution``.

    Parameters
    ----------
    posterior : PosteriorSummary
        Current posterior summary (q).
    prior : PriorSpec
        Prior specification (p), used as the reference distribution.

    Returns
    -------
    W2Result
        Contains ``distance``, ``squared``, and ``family``.

    Raises
    ------
    ValueError
        For unsupported distribution families or invalid parameters.
    KeyError
        If required param keys are absent from prior.params.
    """
    if _is_normal(prior):
        mu_p, var_p = _prior_normal_params(prior)
        sq = w2_normal_squared(posterior.mean, posterior.variance, mu_p, var_p)
        return W2Result(distance=math.sqrt(sq), squared=sq, family="normal")

    if _is_beta(prior):
        alpha_p = float(prior.params["alpha"])
        beta_p = float(prior.params["beta"])
        alpha_q, beta_q = _beta_params_from_posterior(posterior)
        sq = w2_beta_squared(alpha_q, beta_q, alpha_p, beta_p)
        return W2Result(distance=math.sqrt(sq), squared=sq, family="beta")

    raise ValueError(
        f"Unsupported distribution family '{prior.distribution}'. "
        "Supported: normal, gaussian, beta."
    )


# ---------------------------------------------------------------------------
# Batch / time-series helpers
# ---------------------------------------------------------------------------


def w2_series(
    posteriors: list[PosteriorSummary],
    prior: PriorSpec,
) -> list[float]:
    """Compute W₂(posterior_t, prior) for each time step t.

    Parameters
    ----------
    posteriors : list[PosteriorSummary]
        Ordered sequence of posteriors.
    prior : PriorSpec
        Fixed reference distribution (prior).

    Returns
    -------
    list[float]
        W₂ values, one per posterior.  Same length as ``posteriors``.
        Returns an empty list if ``posteriors`` is empty.
    """
    return [w2_posterior(p, prior).distance for p in posteriors]


def w2_rolling_mean(
    w2_vals: list[float],
    window: int,
) -> list[float]:
    """Rolling mean of W₂ values over a sliding window.

    Parameters
    ----------
    w2_vals : list[float]
        W₂ distance series W₂_0, W₂_1, …, W₂_T.
    window : int
        Rolling window length.  Must be ≥ 1.

    Returns
    -------
    list[float]
        Rolling-mean values.  Length = max(0, len(w2_vals) − window + 1).
        Returns an empty list when len(w2_vals) < window.

    Raises
    ------
    ValueError
        If window < 1.

    Notes
    -----
    Unlike ``entropy_rate`` (which differences before smoothing), this
    function smooths the raw W₂ series directly.  This preserves the
    monotone regime-shift signal without introducing sign ambiguity.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if len(w2_vals) < window:
        return []

    result: list[float] = []
    for i in range(len(w2_vals) - window + 1):
        result.append(sum(w2_vals[i : i + window]) / window)
    return result
