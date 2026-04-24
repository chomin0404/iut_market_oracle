"""Bayesian update engine supporting Beta-Binomial and Normal-Normal conjugates.

Supported distributions
-----------------------
beta
    Conjugate update for proportions (default rates, market share, ...).

    Prior  : Beta(α₀, β₀)
    Update : each Evidence contributes
               α += value × weight
               β += (1 − value) × weight
    where value ∈ [0, 1] is the observed proportion and weight is the
    effective sample size.
    Posterior : Beta(α_n, β_n)
    Mean      : α_n / (α_n + β_n)
    Variance  : α_n β_n / [(α_n + β_n)² (α_n + β_n + 1)]

normal
    Conjugate update with precision-weighted observations.

    Prior  : N(μ₀, σ₀²)       params: mu, sigma
    Update : each Evidence contributes precision τ_i = weight
             (caller sets weight = 1/σ_likelihood² when known)
    Posterior precision : τ_n = 1/σ₀² + Σ τ_i
    Posterior mean      : μ_n = (μ₀/σ₀² + Σ(τ_i · v_i)) / τ_n
    Posterior variance  : σ_n² = 1/τ_n
"""

from __future__ import annotations

import math

import scipy.stats as st

from schemas import Evidence, PosteriorSummary, PriorSpec

_SUPPORTED = frozenset(["beta", "normal"])


def update(
    prior: PriorSpec,
    evidence_list: list[Evidence],
) -> PosteriorSummary:
    """Return the posterior summary after incorporating all evidence.

    Parameters
    ----------
    prior:
        Prior distribution specification (see schemas.PriorSpec).
    evidence_list:
        Ordered list of evidence items.  An empty list returns the prior
        predictive summary without any update.

    Raises
    ------
    ValueError
        If the distribution is unsupported or evidence values are
        incompatible with the prior type.
    """
    dist = prior.distribution.lower()
    if dist not in _SUPPORTED:
        raise ValueError(
            f"Unsupported distribution '{prior.distribution}'. Supported: {sorted(_SUPPORTED)}"
        )

    if dist == "beta":
        return _update_beta(prior, evidence_list)
    return _update_normal(prior, evidence_list)


# ---------------------------------------------------------------------------
# Beta-Binomial conjugate
# ---------------------------------------------------------------------------


def _update_beta(prior: PriorSpec, evidence_list: list[Evidence]) -> PosteriorSummary:
    alpha = prior.params["alpha"]
    beta = prior.params["beta"]

    if alpha <= 0 or beta <= 0:
        raise ValueError("Beta prior requires alpha > 0 and beta > 0")

    for ev in evidence_list:
        v = ev.value
        w = ev.weight
        if not (0.0 <= v <= 1.0):
            raise ValueError(
                f"Beta update requires evidence value ∈ [0, 1], got {v} (source='{ev.source}')"
            )
        alpha += v * w
        beta += (1.0 - v) * w

    n = alpha + beta
    mean = alpha / n
    variance = (alpha * beta) / (n**2 * (n + 1))
    lo, hi = st.beta.ppf([0.025, 0.975], alpha, beta)

    return PosteriorSummary(
        mean=mean,
        variance=variance,
        credible_interval_95=(float(lo), float(hi)),
        n_evidence=len(evidence_list),
    )


# ---------------------------------------------------------------------------
# Normal-Normal conjugate (precision weighting)
# ---------------------------------------------------------------------------


def _update_normal(prior: PriorSpec, evidence_list: list[Evidence]) -> PosteriorSummary:
    mu0 = prior.params["mu"]
    sigma0 = prior.params["sigma"]

    if sigma0 <= 0:
        raise ValueError("Normal prior requires sigma > 0")

    tau0 = 1.0 / sigma0**2  # prior precision
    tau_n = tau0
    weighted_sum = mu0 * tau0  # τ₀ · μ₀

    for ev in evidence_list:
        tau_i = ev.weight  # caller supplies precision (1/σ_likelihood²)
        if tau_i <= 0:
            raise ValueError(
                f"Evidence weight (precision) must be > 0, got {tau_i} (source='{ev.source}')"
            )
        if not math.isfinite(ev.value):
            raise ValueError(
                f"Evidence value must be finite, got {ev.value} (source='{ev.source}')"
            )
        tau_n += tau_i
        weighted_sum += tau_i * ev.value

    sigma_n = math.sqrt(1.0 / tau_n)
    mu_n = weighted_sum / tau_n
    lo, hi = st.norm.ppf([0.025, 0.975], loc=mu_n, scale=sigma_n)

    return PosteriorSummary(
        mean=mu_n,
        variance=1.0 / tau_n,
        credible_interval_95=(float(lo), float(hi)),
        n_evidence=len(evidence_list),
    )
