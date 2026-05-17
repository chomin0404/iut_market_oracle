"""Fragmentation kernels, growth rates, and initial particle samplers.

Mathematical basis
------------------
Growth rate (linear):
    τ(x) = a · x     →  exact ODE solution: x(t + Δt) = x(t) · exp(a · Δt)

Fragmentation rate (power-law):
    κ(x) = κ₀ · x^α    [κ₀ > 0, α ≥ 0]

Daughter kernel (uniform binary split with loss):
    p(x, y) = 2 / (β · y)  for 0 ≤ x ≤ β · y     [A1, A2]

    Normalization: ∫₀^y p(x, y) dx = 2  (two daughters per event)
    Loss:          x₁ + x₂ = β · y < y   for β ∈ (0, 1)

Initial distribution (Truncated Student-t, x > 0):
    Provides heavy-tailed robustness while preserving finite 2nd moments.
    Requires df ≥ 3  →  E[x²] < ∞  →  W₂ distance is finite  [A5].

Assumptions
-----------
[A1] Binary split only.
[A2] x₁ + x₂ = β · x_parent, β ∈ (0, 1).
[A5] Initial distribution: Truncated Student-t(ν ≥ 3), x > 0.
"""

from __future__ import annotations

import numpy as np

from fragmentation.schemas import FragConfig

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MIN_PARTICLE_SIZE: float = 1e-12  # guard against numerical underflow


# ---------------------------------------------------------------------------
# Growth rate τ(x)
# ---------------------------------------------------------------------------


def growth_rate(x: float, config: FragConfig) -> float:
    """Linear growth rate τ(x) = tau_coef · x.

    Parameters
    ----------
    x : float
        Current capability score (x > 0).
    config : FragConfig
        Simulation configuration.

    Returns
    -------
    float
        τ(x) ≥ 0.
    """
    return config.tau_coef * x


def apply_growth(sizes: np.ndarray, dt: float, config: FragConfig) -> np.ndarray:
    """Apply linear growth τ(x)=a·x over interval Δt using exact ODE solution.

    Exact solution of ẋ = a·x:  x(t+Δt) = x(t) · exp(a·Δt)

    Parameters
    ----------
    sizes : np.ndarray
        Current capability scores, shape (N,).
    dt : float
        Time interval Δt ≥ 0.
    config : FragConfig
        Simulation configuration.

    Returns
    -------
    np.ndarray
        Updated sizes after growth, shape (N,).
    """
    if config.tau_coef == 0.0:
        return sizes.copy()
    return sizes * np.exp(config.tau_coef * dt)


# ---------------------------------------------------------------------------
# Fragmentation rate κ(x, u)
# ---------------------------------------------------------------------------


def frag_rate(x: float, config: FragConfig, control_u: float = 0.0) -> float:
    """Fragmentation rate κ(x, u) = κ₀ · x^α · (1 + u).

    The control input u modulates the baseline rate multiplicatively.
    u = 0 corresponds to the uncontrolled (natural) fragmentation.

    Parameters
    ----------
    x : float
        Capability score (x > 0).
    config : FragConfig
        Simulation configuration.
    control_u : float
        Control input u ≥ 0  (additive modulation factor).

    Returns
    -------
    float
        κ(x, u) ≥ 0.

    Notes
    -----
    For α=0: κ(x) = κ₀  (constant rate — reference case for Malthus test).
    For α>0: larger sub-swarms fragment faster (power-law).
    """
    base = config.kappa_0 * (x**config.alpha)
    return base * (1.0 + control_u)


def frag_rates_array(sizes: np.ndarray, config: FragConfig, control_u: float = 0.0) -> np.ndarray:
    """Vectorized fragmentation rates for all particles.

    Parameters
    ----------
    sizes : np.ndarray
        Capability scores, shape (N,).
    config : FragConfig
        Simulation configuration.
    control_u : float
        Uniform control input applied to all particles.

    Returns
    -------
    np.ndarray
        Rates κ(xᵢ, u), shape (N,).
    """
    base = config.kappa_0 * (sizes**config.alpha)
    return base * (1.0 + control_u)


# ---------------------------------------------------------------------------
# Daughter kernel p(x, y)
# ---------------------------------------------------------------------------


def sample_daughters(
    x_parent: float,
    config: FragConfig,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Sample two daughter particles from the uniform binary kernel with loss.

    Kernel: p(x, y) = 2 / (β·y)  for 0 ≤ x ≤ β·y

    Sample mechanism:
        U ~ Uniform(0, 1)
        x₁ = U · β · x_parent
        x₂ = (1 − U) · β · x_parent
        x₁ + x₂ = β · x_parent < x_parent   [A2]

    Parameters
    ----------
    x_parent : float
        Size of the fragmenting particle (x_parent > 0).
    config : FragConfig
        Simulation configuration (uses loss_efficiency β).
    rng : np.random.Generator
        Seeded random number generator.

    Returns
    -------
    tuple[float, float]
        (x₁, x₂) with x₁ + x₂ = β · x_parent.
    """
    u = rng.uniform(0.0, 1.0)
    effective_mass = config.loss_efficiency * x_parent
    x1 = u * effective_mass
    x2 = (1.0 - u) * effective_mass
    return x1, x2


def kernel_normalization(y: float, config: FragConfig, n_quad: int = 1000) -> float:
    """Numerically verify ∫₀^y p(x, y) dx = 2  (daughter count per event).

    Parameters
    ----------
    y : float
        Parent size.
    config : FragConfig
        Simulation configuration (uses loss_efficiency β).
    n_quad : int
        Number of quadrature points.

    Returns
    -------
    float
        Integral value, should be ≈ 2.0 for uniform binary kernel.
    """
    xs = np.linspace(0.0, y, n_quad + 1)
    beta_y = config.loss_efficiency * y
    dx = xs[1] - xs[0]
    # p(x, y) = 2 / (β·y) for 0 ≤ x ≤ β·y, else 0
    # Use rectangle rule (np.trapz removed in NumPy 2.0)
    p_vals = np.where(xs <= beta_y, 2.0 / beta_y, 0.0)
    return float(np.sum(p_vals) * dx)


# ---------------------------------------------------------------------------
# Initial particle sampler
# ---------------------------------------------------------------------------


def sample_initial_particles(
    config: FragConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample N initial capability scores from the configured distribution.

    Supported distributions
    -----------------------
    ``"truncated_t"``
        Truncated Student-t(ν=df) on (x_min, x_max).
        Heavy tails with finite 2nd moments (df ≥ 3 enforced by schema).
        [A5]

    ``"gamma"``
        Gamma(shape=k, scale=θ) fitted to (init_loc, init_scale) as
        mean=k·θ and std=√k·θ  →  k=(loc/scale)², θ=scale²/loc.

    ``"uniform_box"``
        Uniform on (x_min, x_max).  Used for sensitivity checks.

    Parameters
    ----------
    config : FragConfig
        Simulation configuration.
    rng : np.random.Generator
        Seeded random number generator.

    Returns
    -------
    np.ndarray
        Shape (n_particles,), all values in (x_min, x_max).
    """
    n = config.n_particles

    if config.init_dist == "truncated_t":
        return _sample_truncated_t(n, config, rng)
    if config.init_dist == "gamma":
        return _sample_gamma(n, config, rng)
    # uniform_box
    return rng.uniform(config.x_min, config.x_max, size=n)


def _sample_truncated_t(n: int, config: FragConfig, rng: np.random.Generator) -> np.ndarray:
    """Rejection sampler for Truncated Student-t(ν, μ, σ) on (x_min, x_max).

    Uses scipy.stats.t for the untruncated CDF/PPF, then rejection-samples
    to enforce x > x_min.  x_max truncation is applied as a secondary rejection.
    """
    df = config.init_df
    loc = config.init_loc
    scale = config.init_scale

    samples: list[float] = []
    batch = max(n * 4, 500)  # over-sample to reduce rejection rounds
    while len(samples) < n:
        raw = rng.standard_t(df=df, size=batch) * scale + loc
        valid = raw[(raw > config.x_min) & (raw < config.x_max)]
        samples.extend(valid.tolist())

    arr = np.array(samples[:n], dtype=float)
    # Normalize to (x_min, x_max) — should already be satisfied, sanity clamp
    arr = np.clip(arr, config.x_min + _MIN_PARTICLE_SIZE, config.x_max)
    return arr


def _sample_gamma(n: int, config: FragConfig, rng: np.random.Generator) -> np.ndarray:
    """Sample from Gamma fitted to (init_loc, init_scale) as (mean, std).

    k = (mean/std)²,  θ = std²/mean
    """
    mean = config.init_loc
    std = config.init_scale
    k = (mean / std) ** 2
    theta = (std**2) / mean
    raw = rng.gamma(shape=k, scale=theta, size=n * 2)
    valid = raw[(raw > config.x_min) & (raw < config.x_max)]
    if len(valid) < n:
        # fallback: pad with uniform samples
        extra = rng.uniform(config.x_min, config.x_max, size=n)
        valid = np.concatenate([valid, extra])
    return valid[:n]
