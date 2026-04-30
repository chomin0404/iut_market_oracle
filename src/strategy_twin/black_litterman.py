"""Black-Litterman posterior return estimation (T1700).

He-Litterman (1999) formulation:

    Prior (market equilibrium):
        Π = δ · Σ · w_mkt          (implied excess returns)

    Posterior precision:
        M = (τΣ)⁻¹ + Pᵀ Ω⁻¹ P

    Posterior mean:
        μ_BL = M⁻¹ [(τΣ)⁻¹ Π + Pᵀ Ω⁻¹ q]

    Posterior variance (diagonal):
        Σ_BL = M⁻¹

Covariance construction:
    σ_i = |growth_rate_i| · 0.5 + 0.05   (heuristic from FCF volatility)
    Σ_{ij} = σ_i · σ_j · ρ               i ≠ j
    Σ_{ii} = σ_i²

with uniform cross-correlation ρ = 0.30.
"""

from __future__ import annotations

import numpy as np

from schemas import BLResult, BLView, BusinessUnit

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BL_DELTA: float = 2.5  # risk-aversion coefficient
_BL_TAU: float = 0.05  # confidence scale on prior (τ·Σ)
_BL_RHO: float = 0.30  # uniform off-diagonal correlation

_EPS: float = 1e-9


# ---------------------------------------------------------------------------
# Covariance helper
# ---------------------------------------------------------------------------


def _build_covariance(units: list[BusinessUnit]) -> np.ndarray:
    """Build covariance matrix from FCF growth rates.

    σ_i = |g_i| · 0.5 + 0.05
    Σ_{ij} = σ_i · σ_j · ρ  (i ≠ j),  Σ_{ii} = σ_i²
    """
    sigma = np.array([abs(u.growth_rate) * 0.5 + 0.05 for u in units])
    cov = np.outer(sigma, sigma) * _BL_RHO
    np.fill_diagonal(cov, sigma**2)
    return cov


# ---------------------------------------------------------------------------
# Black-Litterman
# ---------------------------------------------------------------------------


def black_litterman(
    units: list[BusinessUnit],
    views: list[BLView],
) -> BLResult:
    """Compute Black-Litterman posterior returns.

    Args:
        units: Business units that define assets, weights, and FCF covariance.
        views: Investor views (P·μ = q with uncertainty Ω).

    Returns:
        BLResult with equilibrium returns, posterior returns, and posterior std.
    """
    names = [u.name for u in units]
    n = len(units)

    # Market-cap weights (equal if no size info; use initial_fcf as proxy)
    total_fcf = sum(u.initial_fcf for u in units)
    w_mkt = np.array([u.initial_fcf / total_fcf for u in units])

    Sigma = _build_covariance(units)

    # Equilibrium excess returns: Π = δ · Σ · w_mkt
    pi = _BL_DELTA * Sigma @ w_mkt  # (n,)

    # Build P ∈ ℝ^{k×n}, q ∈ ℝ^k, Ω = diag(uncertainty²)
    k = len(views)
    P = np.zeros((k, n))
    q = np.zeros(k)
    omega_diag = np.zeros(k)

    name_to_idx = {name: i for i, name in enumerate(names)}

    for row, view in enumerate(views):
        total_weight = sum(abs(w) for w in view.assets.values())
        if total_weight < _EPS:
            continue
        for asset_name, weight in view.assets.items():
            if asset_name in name_to_idx:
                P[row, name_to_idx[asset_name]] = weight / total_weight
        q[row] = view.expected_return
        omega_diag[row] = view.uncertainty**2

    # Posterior precision: M = (τΣ)⁻¹ + Pᵀ Ω⁻¹ P
    tau_Sigma_inv = np.linalg.inv(_BL_TAU * Sigma + _EPS * np.eye(n))

    if k > 0:
        Omega_inv = np.diag(1.0 / np.maximum(omega_diag, _EPS))
        M = tau_Sigma_inv + P.T @ Omega_inv @ P
        rhs = tau_Sigma_inv @ pi + P.T @ Omega_inv @ q
    else:
        M = tau_Sigma_inv
        rhs = tau_Sigma_inv @ pi

    M_inv = np.linalg.inv(M + _EPS * np.eye(n))
    mu_bl = M_inv @ rhs  # posterior mean
    sigma_bl = np.sqrt(np.maximum(np.diag(M_inv), 0.0))  # posterior std

    return BLResult(
        equilibrium_returns={names[i]: float(pi[i]) for i in range(n)},
        posterior_returns={names[i]: float(mu_bl[i]) for i in range(n)},
        posterior_std={names[i]: float(sigma_bl[i]) for i in range(n)},
        market_weights={names[i]: float(w_mkt[i]) for i in range(n)},
    )
