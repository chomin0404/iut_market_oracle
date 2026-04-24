"""Log-concavity computation for matroid rank-generating polynomials (T1200).

Model
-----
Given n ground elements, define the rank-generating polynomial coefficients:

    b_k = C(n, k) · alpha^k · beta^(n-k),   k = 0, 1, …, n

where:
    alpha  (rank_weight)   — multiplicative weight per element in the independent set
    beta   (corank_weight) — multiplicative weight per element in the complement

After normalisation:
    p_k = b_k / Σ_j b_j

This gives a PMF over subset sizes, equivalent to Binomial(n, p) with
    p = alpha / (alpha + beta)

Log-concavity condition (per interior index k = 1, …, n-1):
    b_k² ≥ b_{k-1} · b_{k+1}

Log-concavity is guaranteed here because the unnormalised b_k are the terms of
(alpha + beta)^n via the binomial theorem; any binomial PMF is log-concave.
June Huh's theorem extends this guarantee to arbitrary matroids.

Floating-point note: for n > ~60, C(n, k) overflows float64 before weighting.
The implementation uses log-space arithmetic to avoid this, then exponentiates.
"""

from __future__ import annotations

import math

import numpy as np

from schemas import MatroidLogConcavityResult

# Small constant to avoid log(0) when computing log-probabilities
_LOG_EPSILON: float = 1e-10


def compute_log_concave_weights(
    n_assets: int,
    rank_weight: float = 0.8,
    corank_weight: float = 1.2,
) -> MatroidLogConcavityResult:
    """Compute log-concave subset-size weights for a rank-generating polynomial.

    Uses log-space arithmetic to avoid overflow for large n_assets.

    Parameters
    ----------
    n_assets:
        Number of ground elements (>= 1).
    rank_weight:
        Multiplicative weight alpha per element in the independent set (> 0).
        Controls how smaller subsets are weighted relative to larger ones.
    corank_weight:
        Multiplicative weight beta per element in the complement (> 0).

    Returns
    -------
    MatroidLogConcavityResult

    Raises
    ------
    ValueError
        If n_assets < 1, rank_weight <= 0, or corank_weight <= 0.
    """
    if n_assets < 1:
        raise ValueError(f"n_assets must be >= 1, got {n_assets}")
    if rank_weight <= 0.0:
        raise ValueError(f"rank_weight must be > 0, got {rank_weight}")
    if corank_weight <= 0.0:
        raise ValueError(f"corank_weight must be > 0, got {corank_weight}")

    log_alpha = math.log(rank_weight)
    log_beta = math.log(corank_weight)

    # Compute log(b_k) = log(C(n,k)) + k*log(alpha) + (n-k)*log(beta)
    # math.lgamma gives log(n!) without overflow
    log_b = np.array(
        [
            math.lgamma(n_assets + 1)
            - math.lgamma(k + 1)
            - math.lgamma(n_assets - k + 1)
            + k * log_alpha
            + (n_assets - k) * log_beta
            for k in range(n_assets + 1)
        ],
        dtype=float,
    )

    # Numerically stable normalisation via log-sum-exp
    log_b_max = log_b.max()
    log_sum = log_b_max + math.log(np.sum(np.exp(log_b - log_b_max)))
    log_p = log_b - log_sum  # log(normalised probability mass)
    p_k = np.exp(log_p)  # normalised probability mass

    # Log-concavity check: b_k² >= b_{k-1} * b_{k+1}
    # Equivalent in log-space: 2*log_b[k] >= log_b[k-1] + log_b[k+1]
    checks: list[bool] = [2.0 * log_b[k] >= log_b[k - 1] + log_b[k + 1] for k in range(1, n_assets)]

    return MatroidLogConcavityResult(
        n_assets=n_assets,
        rank_weight=rank_weight,
        corank_weight=corank_weight,
        subset_sizes=list(range(n_assets + 1)),
        probability_mass=p_k.tolist(),
        log_probability=(log_p).tolist(),
        log_concavity_checks=checks,
        is_log_concave=all(checks),
    )


def plot_log_concavity(result: MatroidLogConcavityResult):  # -> matplotlib.figure.Figure
    """Return a matplotlib Figure visualising the log-concavity result.

    Two panels:
        Top: probability mass b_k (bar) with log-concavity violations marked.
        Bottom: log probability ln(b_k) — a concave curve confirms log-concavity.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    k = result.subset_sizes
    p = result.probability_mass
    lp = result.log_probability

    # Identify violation indices (k=1…n-1 where check failed)
    violation_k = [k[i + 1] for i, ok in enumerate(result.log_concavity_checks) if not ok]
    violation_p = [p[i + 1] for i, ok in enumerate(result.log_concavity_checks) if not ok]

    fig, (ax_mass, ax_log) = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [3, 2]}
    )

    ax_mass.bar(k, p, color="indigo", alpha=0.75, label="Probability mass $b_k$")
    if violation_k:
        ax_mass.scatter(
            violation_k,
            violation_p,
            color="red",
            zorder=5,
            label="Log-concavity violation",
        )
    status = "log-concave ✓" if result.is_log_concave else "NOT log-concave ✗"
    ax_mass.set_ylabel("Probability mass $b_k$")
    ax_mass.set_title(
        f"Matroid rank-generating polynomial — {status}\n"
        f"n={result.n_assets},  α={result.rank_weight},  β={result.corank_weight}"
    )
    ax_mass.legend(fontsize=8)

    ax_log.plot(
        k,
        lp,
        color="darkorange",
        linewidth=1.5,
        marker="o",
        markersize=4,
        linestyle="--",
        label="$\\ln(b_k)$",
    )
    ax_log.set_xlabel("Subset size $k$")
    ax_log.set_ylabel("$\\ln(b_k)$")
    ax_log.legend(fontsize=8)

    fig.tight_layout()
    return fig
