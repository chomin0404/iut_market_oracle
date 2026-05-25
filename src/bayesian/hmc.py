"""Hamiltonian Monte Carlo (HMC) sampler.

Algorithm
---------
HMC augments the position x with a momentum variable p ~ N(0, M) and
integrates Hamilton's equations using a leapfrog scheme.  A final
Metropolis–Hastings correction keeps the chain exact despite numerical
integration error.

Hamiltonian:
    H(x, p) = U(x) + K(p)
    U(x)    = -log p̃(x)           potential energy
    K(p)    = pᵀ M⁻¹ p / 2        kinetic energy (diagonal M)

Leapfrog integrator (L steps, step size ε):
    p_{1/2}  = p_0 + (ε/2) ∇log p̃(x_0)
    for l in 1 … L-1:
        x_l      = x_{l-1} + ε M⁻¹ p_{l-1/2}
        p_{l+1/2} = p_{l-1/2} + ε ∇log p̃(x_l)
    x*       = x_{L-1} + ε M⁻¹ p_{L-1/2}
    p*       = p_{L-1/2} + (ε/2) ∇log p̃(x*)

Accept (x*, −p*) with probability min(1, exp(H(x₀,p₀) − H(x*,p*))).
Only x is collected; the momentum is discarded and re-drawn each step.

References
----------
Neal, R. M. (2011). MCMC using Hamiltonian dynamics.
    *Handbook of Markov Chain Monte Carlo*, Chapter 5.

Functions
---------
run_hmc
    Run an HMC chain and return samples + diagnostics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from bayesian.sampler import TargetDistribution


@dataclass(frozen=True)
class HMCResult:
    """Output of a single HMC run.

    Attributes
    ----------
    samples:
        Collected chain, shape ``(n_samples, dim)``.
    acceptance_rate:
        Fraction of leapfrog proposals accepted over the full run
        (burn-in + thinned collection phase).
    n_accepted:
        Raw count of accepted proposals over the full run.
    n_total:
        Total number of leapfrog proposals
        (``burn_in + n_samples * thin``).
    """

    samples: np.ndarray
    acceptance_rate: float
    n_accepted: int
    n_total: int


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _leapfrog(
    x: np.ndarray,
    p: np.ndarray,
    grad_fn: object,
    step_size: float,
    n_leapfrog: int,
    inv_mass: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance (x, p) by L leapfrog steps of size ε.

    Parameters
    ----------
    x:
        Position, shape ``(d,)``.
    p:
        Momentum, shape ``(d,)``.
    grad_fn:
        Callable ``x -> ∇ log p̃(x)``, shape ``(d,)``.
    step_size:
        Leapfrog step size ε.
    n_leapfrog:
        Number of leapfrog steps L.
    inv_mass:
        Diagonal of M⁻¹, shape ``(d,)``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(x*, p*)`` after L steps.
    """
    assert callable(grad_fn)
    x = x.copy()
    p = p.copy()

    # Half-step for momentum
    p += 0.5 * step_size * grad_fn(x)  # type: ignore[operator]

    for _ in range(n_leapfrog - 1):
        x += step_size * inv_mass * p
        p += step_size * grad_fn(x)  # type: ignore[operator]

    # Final full position step + half momentum step
    x += step_size * inv_mass * p
    p += 0.5 * step_size * grad_fn(x)  # type: ignore[operator]

    return x, p


def _kinetic(p: np.ndarray, inv_mass: np.ndarray) -> float:
    """Compute kinetic energy K(p) = pᵀ M⁻¹ p / 2."""
    return 0.5 * float(np.dot(p, inv_mass * p))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_hmc(
    target: TargetDistribution,
    step_size: float,
    n_leapfrog: int,
    initial: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
    *,
    burn_in: int = 0,
    thin: int = 1,
    mass: np.ndarray | None = None,
) -> HMCResult:
    """Run a Hamiltonian Monte Carlo chain.

    Parameters
    ----------
    target:
        Unnormalized target distribution.  ``grad_log_prob`` **must** be
        implemented; it raises ``NotImplementedError`` otherwise.
    step_size:
        Leapfrog step size ε > 0.
    n_leapfrog:
        Number of leapfrog steps L ≥ 1 per proposal.
    initial:
        Starting position, shape ``(dim,)``.
    n_samples:
        Number of samples to collect (after burn-in and thinning).
    rng:
        NumPy random generator for reproducibility.
    burn_in:
        Steps to discard before collection.  Defaults to 0.
    thin:
        Collect one sample every ``thin`` steps after burn-in.
        Defaults to 1 (no thinning).
    mass:
        Diagonal mass matrix M as a 1-D array of shape ``(dim,)``.
        Larger values damp momentum more along the corresponding axis.
        Defaults to the identity (all ones).

    Returns
    -------
    HMCResult
        Collected samples and acceptance diagnostics.

    Raises
    ------
    ValueError
        On invalid hyperparameters or mismatched shapes.
    NotImplementedError
        If ``target.grad_log_prob`` is not implemented.
    """
    # --- Validate inputs ---
    if step_size <= 0.0:
        raise ValueError(f"step_size must be > 0, got {step_size!r}")
    if n_leapfrog < 1:
        raise ValueError(f"n_leapfrog must be >= 1, got {n_leapfrog}")
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    if burn_in < 0:
        raise ValueError(f"burn_in must be >= 0, got {burn_in}")
    if thin < 1:
        raise ValueError(f"thin must be >= 1, got {thin}")

    initial = np.asarray(initial, dtype=float)
    d = target.dim
    if initial.shape != (d,):
        raise ValueError(
            f"initial shape {initial.shape} does not match target.dim={d}"
        )

    if mass is None:
        mass_diag = np.ones(d)
    else:
        mass_diag = np.asarray(mass, dtype=float)
        if mass_diag.shape != (d,):
            raise ValueError(
                f"mass shape {mass_diag.shape} does not match target.dim={d}"
            )
        if np.any(mass_diag <= 0.0):
            raise ValueError("All mass entries must be > 0")

    inv_mass = 1.0 / mass_diag          # M⁻¹ diagonal
    sqrt_mass = np.sqrt(mass_diag)      # for sampling p ~ N(0, M)

    # Verify grad_log_prob is available (raises NotImplementedError if not)
    _ = target.grad_log_prob(initial)

    # --- Initialise chain ---
    current = initial.copy()
    log_p_current = target.log_prob(current)

    n_total = burn_in + n_samples * thin
    samples = np.empty((n_samples, d), dtype=float)
    n_accepted = 0
    collect_idx = 0

    for step in range(n_total):
        # Refresh momentum: p ~ N(0, M)
        p0 = rng.standard_normal(d) * sqrt_mass

        # Leapfrog trajectory
        x_new, p_new = _leapfrog(
            current, p0, target.grad_log_prob, step_size, n_leapfrog, inv_mass
        )
        log_p_new = target.log_prob(x_new)

        # Hamiltonian difference (current − proposed  →  log acceptance ratio)
        # ΔH = H(x*,p*) − H(x₀,p₀) = [U(x*) − U(x₀)] + [K(p*) − K(p₀)]
        current_H = -log_p_current + _kinetic(p0, inv_mass)
        proposed_H = -log_p_new + _kinetic(p_new, inv_mass)
        log_alpha = current_H - proposed_H

        # Accept / reject
        if math.log(rng.uniform()) < log_alpha:
            current = x_new
            log_p_current = log_p_new
            n_accepted += 1

        # Collect after burn-in with thinning
        if step >= burn_in:
            offset = step - burn_in
            if offset % thin == 0:
                samples[collect_idx] = current
                collect_idx += 1

    return HMCResult(
        samples=samples,
        acceptance_rate=n_accepted / n_total,
        n_accepted=n_accepted,
        n_total=n_total,
    )
