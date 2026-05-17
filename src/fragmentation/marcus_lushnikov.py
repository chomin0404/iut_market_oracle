"""Marcus-Lushnikov process: exact Gillespie simulation of the GFE.

Mathematical basis
------------------
The Marcus-Lushnikov process is the finite-particle stochastic counterpart
of the growth-fragmentation equation.  As N → ∞, the empirical measure

    μ_N(t) = (1/N) Σᵢ δ_{xᵢ(t)}

converges to the solution n(t,·) of the GFE (Norris 1999).

Gillespie algorithm (exact CTMC)
---------------------------------
State:  sorted list of capability scores {xᵢ}_{i=1}^N(t)

At each step:
1. Compute rates:  rᵢ = κ(xᵢ, u)  for each particle i.
2. Total rate:     R = Σᵢ rᵢ.
3. Time to event:  Δt ~ Exp(R)  (exact, memoryless).
4. Apply growth:   xᵢ ← xᵢ · exp(tau_coef · Δt)  (exact for linear τ).
5. Select particle i* with probability rᵢ/R.
6. Fragment i*: sample daughters (x₁, x₂) from daughter kernel.
7. Remove i*; add daughters if xⱼ ≥ x_min  [A6].
8. Record event in log.

Assumptions
-----------
[A1] Binary split only.
[A2] x₁ + x₂ = β · x_parent < x_parent  (loss, β ∈ (0,1)).
[A4] Control updated at each fragmentation event (discrete event-driven).
[A6] Absorption boundary at x_min.

Wasserstein-2 cost
------------------
After simulation, W₂²(μ_N(T), μ*) is estimated via the sorted-quantile
estimator for 1D measures:

    W₂² ≈ (1/N) Σₖ (x_(k) − y_k)²

where x_(k) are the sorted particle sizes and y_k = Q_{μ*}(k/N) are
the corresponding quantiles of the Gaussian target μ*.
"""

from __future__ import annotations

import numpy as np
import scipy.stats as st

from fragmentation.kernels import (
    apply_growth,
    frag_rates_array,
    sample_daughters,
    sample_initial_particles,
)
from fragmentation.schemas import (
    EigenResult,
    FragConfig,
    FragResult,
    ParticleSnapshot,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MIN_TOTAL_RATE: float = 1e-12  # guard: stop if all rates are negligible
_MAX_REASONS_LOG: int = 500  # cap on stored event strings (memory guard)


# ---------------------------------------------------------------------------
# Wasserstein-2 estimator (1D empirical vs. Gaussian target)
# ---------------------------------------------------------------------------


def w2_squared_empirical_gaussian(
    sizes: np.ndarray,
    target_mean: float,
    target_std: float,
) -> float:
    """Estimate W₂²(μ_N, μ*) for empirical μ_N vs. Gaussian μ* = N(μ*, σ*²).

    Sorted-quantile estimator for 1D:
        W₂² ≈ (1/N) Σₖ (x_(k) − Q_{μ*}(k/(N+1)))²

    Parameters
    ----------
    sizes : np.ndarray
        Empirical particle sizes (unsorted), shape (N,).
    target_mean : float
        Mean of the Gaussian target distribution.
    target_std : float
        Std of the Gaussian target distribution.

    Returns
    -------
    float
        Estimated W₂² ≥ 0.
    """
    n = len(sizes)
    if n == 0:
        return float("inf")

    sorted_x = np.sort(sizes)
    # Quantile levels: avoid 0 and 1 to prevent ±inf from ppf
    levels = (np.arange(1, n + 1)) / (n + 1)
    target_quantiles = st.norm.ppf(levels, loc=target_mean, scale=target_std)

    return float(np.mean((sorted_x - target_quantiles) ** 2))


# ---------------------------------------------------------------------------
# Gillespie simulator
# ---------------------------------------------------------------------------


def simulate(
    config: FragConfig,
    eigen: EigenResult,
    control_u: float = 0.0,
) -> FragResult:
    """Run the Marcus-Lushnikov Gillespie simulation.

    Parameters
    ----------
    config : FragConfig
        Full simulation configuration.
    eigen : EigenResult
        Pre-computed eigenanalysis result (embedded in FragResult).
    control_u : float
        Uniform control input u ≥ 0 applied to all fragmentation rates.
        u = 0: uncontrolled (baseline).

    Returns
    -------
    FragResult
        Full result including trajectory, eigenanalysis, W₂ cost, and log.
    """
    rng = np.random.default_rng(config.seed)

    # -----------------------------------------------------------------------
    # Initialise particles
    # -----------------------------------------------------------------------
    sizes: np.ndarray = sample_initial_particles(config, rng)

    trajectory: list[ParticleSnapshot] = [
        ParticleSnapshot(
            time=0.0,
            sizes=sizes.tolist(),
            n_particles=len(sizes),
            event="initial",
        )
    ]
    reasons: list[str] = []
    t = 0.0

    # -----------------------------------------------------------------------
    # Gillespie main loop
    # -----------------------------------------------------------------------
    while t < config.T:
        if len(sizes) == 0:
            break

        rates = frag_rates_array(sizes, config, control_u)
        total_rate = float(rates.sum())

        if total_rate < _MIN_TOTAL_RATE:
            break

        # --- Time to next event ---
        dt = rng.exponential(1.0 / total_rate)
        if t + dt > config.T:
            # Apply final growth up to T and stop
            sizes = apply_growth(sizes, config.T - t, config)
            break

        # --- Apply growth to all particles (exact for linear τ) ---
        sizes = apply_growth(sizes, dt, config)
        t += dt

        # --- Select fragmenting particle ---
        probs = rates / total_rate
        idx = int(rng.choice(len(sizes), p=probs))
        x_parent = float(sizes[idx])

        # --- Sample daughters ---
        x1, x2 = sample_daughters(x_parent, config, rng)

        # --- Update particle list ---
        sizes = np.delete(sizes, idx)
        new_particles = []
        if x1 >= config.x_min:
            new_particles.append(x1)
        if x2 >= config.x_min:
            new_particles.append(x2)
        if new_particles:
            sizes = np.concatenate([sizes, new_particles])

        # --- Log event ---
        if len(reasons) < _MAX_REASONS_LOG:
            reasons.append(
                f"t={t:.4f}: split {x_parent:.4f} → {x1:.4f}+{x2:.4f}"
                f" (loss={1 - config.loss_efficiency:.2f})"
            )

        # --- Record snapshot (every event) ---
        trajectory.append(
            ParticleSnapshot(
                time=t,
                sizes=sizes.tolist(),
                n_particles=len(sizes),
                event="fragmentation" if new_particles else "absorption",
            )
        )

        # --- Safety cap (prevent memory explosion) ---
        if len(sizes) >= config.n_particles_max:
            break

    # -----------------------------------------------------------------------
    # Compute W₂² cost at final state
    # -----------------------------------------------------------------------
    cost_w2 = w2_squared_empirical_gaussian(sizes, config.target_mean, config.target_std)

    score_components: dict[str, float] = {
        "coverage_loss": cost_w2,
        "n_fragments": float(len(sizes)),
        "mean_size": float(sizes.mean()) if len(sizes) > 0 else 0.0,
        "control_u": control_u,
    }

    return FragResult(
        config=config,
        trajectory=trajectory,
        eigen=eigen,
        cost_w2=cost_w2,
        score_components=score_components,
        reasons=reasons,
    )
