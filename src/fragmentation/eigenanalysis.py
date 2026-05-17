"""Malthus parameter and eigenfunction estimation for the GFE.

Method: Gillespie Monte Carlo regression
-----------------------------------------
The Malthus parameter λ controls the long-time exponential growth of particle
number in the growth-fragmentation process:

    N(t) ~ C · exp(λ · t)   as t → ∞  (before absorption dominates)

Estimation procedure:
1. Run a reference Gillespie simulation (control u=0, independent seed).
2. Record (t_k, N(t_k)) at each fragmentation event.
3. Fit linear regression of log(N(t)) vs t on the second half of the trajectory
   (to avoid initial transient).
4. λ̂ = slope of the regression.

Eigenfunction φ(x) ≈ empirical density of particle sizes at final time,
computed as a normalized histogram.

Why not matrix eigenanalysis?
------------------------------
The discrete GFE operator A is upper triangular (gains flow from large x to
small x, i.e., A_{ij} for j > i).  Upper triangular matrices have eigenvalues
equal to their diagonal elements.  For constant κ₀, the diagonal is -κ₀,
giving all eigenvalues = -κ₀ regardless of the gain term.  This is a
mathematical artifact of the truncated discrete approximation and does not
reflect the true Malthus parameter of the unbounded GFE.

The Gillespie-based estimator directly measures the empirical growth rate of
the stochastic process and is consistent as N → ∞ (Norris 1999).

Reference test case
--------------------
τ(x) = 0,  κ(x) = κ₀ (constant),  β → 1 (near no-loss):
  Each event: 1 particle → 2 daughters → N doubles at rate κ₀
  → Ṅ = κ₀ N  →  λ = κ₀
"""

from __future__ import annotations

import numpy as np

from fragmentation.kernels import (
    apply_growth,
    frag_rates_array,
    sample_daughters,
    sample_initial_particles,
)
from fragmentation.schemas import EigenResult, FragConfig

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MIN_TOTAL_RATE: float = 1e-12
_MIN_EVENTS_FOR_REGRESSION: int = 10  # require at least this many events
_CONVERGENCE_R2_THRESHOLD: float = 0.70  # R² threshold for "converged"
_EIGEN_SEED_XOR: int = 0xDEAD_BEEF  # XOR mask to get independent seed


# ---------------------------------------------------------------------------
# Malthus parameter estimator
# ---------------------------------------------------------------------------


def estimate_malthus(config: FragConfig) -> EigenResult:
    """Estimate Malthus parameter λ and eigenfunction φ(x) via Gillespie MC.

    Parameters
    ----------
    config : FragConfig
        Simulation configuration.

    Returns
    -------
    EigenResult
        malthus_lambda : slope of log(N(t)) linear regression (λ̂).
        eigenfunction_x : bin centers of empirical density histogram.
        eigenfunction_phi : normalized histogram of particle sizes at final t.
        converged : True if R² of log-linear fit ≥ threshold.
        spectral_gap : R² of the log-linear fit (proxy for fit quality).

    Notes
    -----
    Uses a different seed (config.seed XOR _EIGEN_SEED_XOR) from the main
    simulation to ensure the eigenanalysis and main run are independent.
    """
    # Use a different seed so eigenanalysis and main simulation are independent
    eigen_seed = config.seed ^ _EIGEN_SEED_XOR
    rng = np.random.default_rng(eigen_seed)

    sizes = sample_initial_particles(config, rng)

    times: list[float] = [0.0]
    counts: list[int] = [len(sizes)]
    t = 0.0

    # --- Minimal Gillespie loop (no trajectory storage needed) ---
    while t < config.T and 0 < len(sizes) < config.n_particles_max:
        rates = frag_rates_array(sizes, config, control_u=0.0)
        total_rate = float(rates.sum())
        if total_rate < _MIN_TOTAL_RATE:
            break

        dt = rng.exponential(1.0 / total_rate)
        if t + dt > config.T:
            sizes = apply_growth(sizes, config.T - t, config)
            break

        sizes = apply_growth(sizes, dt, config)
        t += dt

        probs = rates / total_rate
        idx = int(rng.choice(len(sizes), p=probs))
        x_parent = float(sizes[idx])
        x1, x2 = sample_daughters(x_parent, config, rng)

        sizes = np.delete(sizes, idx)
        new_particles = [x for x in (x1, x2) if x >= config.x_min]
        if new_particles:
            sizes = np.concatenate([sizes, new_particles])

        times.append(t)
        counts.append(len(sizes))

    # --- Fit λ from log(N) regression ---
    malthus_lambda, converged, r2 = _fit_malthus(
        np.array(times), np.array(counts, dtype=float), config
    )

    # --- Eigenfunction from empirical density ---
    eigenfunction_x, eigenfunction_phi = _empirical_density(sizes, config)

    return EigenResult(
        malthus_lambda=malthus_lambda,
        eigenfunction_x=eigenfunction_x,
        eigenfunction_phi=eigenfunction_phi,
        converged=converged,
        spectral_gap=r2,
    )


# ---------------------------------------------------------------------------
# Helper: log-linear regression
# ---------------------------------------------------------------------------


def _fit_malthus(
    times: np.ndarray,
    counts: np.ndarray,
    config: FragConfig,
) -> tuple[float, bool, float]:
    """Fit λ̂ from log(N(t)) linear regression on second half of trajectory.

    Returns
    -------
    (malthus_lambda, converged, r2)
    """
    n_events = len(times)
    if n_events < _MIN_EVENTS_FOR_REGRESSION:
        # Fallback: analytical estimate for constant κ, binary split
        return float(config.kappa_0), False, 0.0

    log_counts = np.log(np.maximum(counts, 1.0))

    # Use second half to skip initial transient
    half = n_events // 2
    t_fit = times[half:]
    lc_fit = log_counts[half:]

    t_span = float(t_fit[-1] - t_fit[0])
    if t_span < 1e-10 or len(t_fit) < 2:
        return float(config.kappa_0), False, 0.0

    coeffs = np.polyfit(t_fit, lc_fit, 1)
    slope = float(coeffs[0])

    # R² for fit quality assessment
    fitted = np.polyval(coeffs, t_fit)
    ss_res = float(np.sum((lc_fit - fitted) ** 2))
    ss_tot = float(np.sum((lc_fit - lc_fit.mean()) ** 2))
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)
    r2 = float(np.clip(r2, 0.0, 1.0))

    converged = r2 >= _CONVERGENCE_R2_THRESHOLD
    return slope, converged, r2


# ---------------------------------------------------------------------------
# Helper: empirical density histogram
# ---------------------------------------------------------------------------


def _empirical_density(
    sizes: np.ndarray,
    config: FragConfig,
) -> tuple[list[float], list[float]]:
    """Compute normalized histogram of particle sizes as eigenfunction proxy.

    Returns (bin_centers, density) each of length pde_grid_size.
    """
    n_bins = config.pde_grid_size
    edges = np.linspace(config.x_min, config.x_max, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    if len(sizes) == 0:
        phi = np.zeros(n_bins)
    else:
        phi, _ = np.histogram(sizes, bins=edges, density=True)
        phi = phi.astype(float)

    return centers.tolist(), phi.tolist()
