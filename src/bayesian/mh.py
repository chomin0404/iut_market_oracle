"""Metropolis–Hastings sampler.

The acceptance step:

    Symmetric kernel (q(x'|x) == q(x|x')):
        log α = log p̃(x') − log p̃(x)

    Asymmetric kernel:
        log α = log p̃(x') − log p̃(x)
              + log q(x | x') − log q(x' | x)

    Accept x' if log U < log α  where U ~ Uniform(0, 1).

Functions
---------
run_mh
    Run a Metropolis–Hastings chain and return samples + diagnostics.

Data classes
------------
MCMCResult
    Immutable container for chain output.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from bayesian.sampler import ProposalKernel, TargetDistribution


@dataclass(frozen=True)
class MCMCResult:
    """Output of a single Metropolis–Hastings run.

    Attributes
    ----------
    samples:
        Collected chain, shape ``(n_samples, dim)``.
    acceptance_rate:
        Fraction of proposals accepted over the *full* run
        (burn-in + thinned collection phase).
    n_accepted:
        Raw count of accepted proposals over the full run.
    n_total:
        Total number of proposals (``burn_in + n_samples * thin``).
    """

    samples: np.ndarray
    acceptance_rate: float
    n_accepted: int
    n_total: int


def run_mh(
    target: TargetDistribution,
    kernel: ProposalKernel,
    initial: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
    *,
    burn_in: int = 0,
    thin: int = 1,
) -> MCMCResult:
    """Run a Metropolis–Hastings chain.

    Parameters
    ----------
    target:
        Unnormalized target distribution.
    kernel:
        Proposal kernel.
    initial:
        Starting state, shape ``(dim,)``.
    n_samples:
        Number of samples to *collect* (after burn-in and thinning).
    rng:
        NumPy random generator for reproducibility.
    burn_in:
        Number of initial steps to discard.  Defaults to 0.
    thin:
        Collect one sample every ``thin`` steps after burn-in.
        Defaults to 1 (no thinning).

    Returns
    -------
    MCMCResult
        Collected samples and acceptance diagnostics.

    Raises
    ------
    ValueError
        If ``n_samples < 1``, ``burn_in < 0``, or ``thin < 1``.
    ValueError
        If ``initial`` shape does not match ``target.dim``.
    """
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    if burn_in < 0:
        raise ValueError(f"burn_in must be >= 0, got {burn_in}")
    if thin < 1:
        raise ValueError(f"thin must be >= 1, got {thin}")

    initial = np.asarray(initial, dtype=float)
    if initial.shape != (target.dim,):
        raise ValueError(
            f"initial shape {initial.shape} does not match target.dim={target.dim}"
        )

    current = initial.copy()
    log_p_current = target.log_prob(current)

    n_total = burn_in + n_samples * thin
    samples = np.empty((n_samples, target.dim), dtype=float)
    n_accepted = 0
    collect_idx = 0  # index into samples

    for step in range(n_total):
        # --- Propose ---
        proposed = kernel.propose(current, rng)
        log_p_proposed = target.log_prob(proposed)

        # --- Log acceptance ratio ---
        log_alpha = log_p_proposed - log_p_current
        if not kernel.is_symmetric:
            log_alpha += kernel.log_transition_prob(current, proposed)
            log_alpha -= kernel.log_transition_prob(proposed, current)

        # --- Accept / reject ---
        if math.log(rng.uniform()) < log_alpha:
            current = proposed
            log_p_current = log_p_proposed
            n_accepted += 1

        # --- Collect after burn-in, respecting thin ---
        if step >= burn_in:
            offset = step - burn_in
            if offset % thin == 0:
                samples[collect_idx] = current
                collect_idx += 1

    return MCMCResult(
        samples=samples,
        acceptance_rate=n_accepted / n_total,
        n_accepted=n_accepted,
        n_total=n_total,
    )
