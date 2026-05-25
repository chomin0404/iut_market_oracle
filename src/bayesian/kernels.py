"""Concrete ProposalKernel implementations for MCMC.

Classes
-------
GaussianRWKernel
    Isotropic or full-covariance Gaussian random-walk proposal.
    x' = x + ε,  ε ~ N(0, Σ)
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as la

from bayesian.sampler import ProposalKernel

_TWO_PI = 2.0 * np.pi


class GaussianRWKernel(ProposalKernel):
    """Gaussian random-walk proposal kernel.

    Draws candidates as x' = x + ε where ε ~ N(0, Σ).

    Parameters
    ----------
    step_size:
        Isotropic standard deviation σ so that Σ = σ²I.
        Mutually exclusive with ``cov``.
    cov:
        Full covariance matrix Σ, shape ``(d, d)``.  Must be symmetric
        positive definite.  Mutually exclusive with ``step_size``.

    Notes
    -----
    Because q(x'|x) = N(x'; x, Σ) = N(x; x', Σ) = q(x|x'),
    ``is_symmetric`` is ``True`` and the Metropolis–Hastings acceptance
    ratio simplifies to min(1, p̃(x') / p̃(x)).

    For the full-covariance case the Cholesky factor L (Σ = LLᵀ) is
    computed once at construction and reused for both sampling and
    log-probability evaluation.
    """

    def __init__(
        self,
        step_size: float | None = None,
        cov: np.ndarray | None = None,
    ) -> None:
        if (step_size is None) == (cov is None):
            raise ValueError("Specify exactly one of step_size or cov.")

        if step_size is not None:
            if step_size <= 0.0:
                raise ValueError(f"step_size must be > 0, got {step_size!r}")
            self._step_size: float | None = float(step_size)
            self._chol: np.ndarray | None = None
        else:
            arr = np.asarray(cov, dtype=float)
            if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
                raise ValueError(f"cov must be a square 2-D matrix, got shape {arr.shape}")
            # la.cholesky raises LinAlgError if cov is not positive definite
            self._chol = la.cholesky(arr, lower=True)
            self._step_size = None

    # ------------------------------------------------------------------
    # ProposalKernel interface
    # ------------------------------------------------------------------

    @property
    def is_symmetric(self) -> bool:
        return True

    def propose(self, current: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Draw x' ~ N(current, Σ).

        Parameters
        ----------
        current:
            Current state, shape ``(d,)``.
        rng:
            NumPy random generator (passed in; not stored).

        Returns
        -------
        np.ndarray
            Proposed state, shape ``(d,)``.
        """
        z = rng.standard_normal(current.shape)
        if self._step_size is not None:
            return current + self._step_size * z
        assert self._chol is not None
        return current + self._chol @ z

    def log_transition_prob(self, x_new: np.ndarray, x_from: np.ndarray) -> float:
        """Compute log q(x_new | x_from) = log N(x_new; x_from, Σ).

        Parameters
        ----------
        x_new:
            Proposed state, shape ``(d,)``.
        x_from:
            Current state, shape ``(d,)``.

        Returns
        -------
        float
            log q(x_new | x_from).
        """
        delta = x_new - x_from

        if self._step_size is not None:
            d = float(delta.size)
            # log N(δ; 0, σ²I) = -d/2 log(2π) - d log(σ) - ||δ||² / (2σ²)
            return float(
                -0.5 * d * np.log(_TWO_PI)
                - d * np.log(self._step_size)
                - 0.5 * np.dot(delta, delta) / self._step_size**2
            )

        assert self._chol is not None
        d = float(self._chol.shape[0])
        # Solve L y = δ  →  yᵀy = δᵀ Σ⁻¹ δ
        y = la.solve_triangular(self._chol, delta, lower=True)
        # log|Σ| = 2 Σ log diag(L)
        log_det = 2.0 * float(np.sum(np.log(np.diag(self._chol))))
        return float(
            -0.5 * (d * np.log(_TWO_PI) + log_det)
            - 0.5 * float(np.dot(y, y))
        )
