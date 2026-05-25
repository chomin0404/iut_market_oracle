"""Abstract interfaces for MCMC sampling components.

Classes
-------
TargetDistribution
    Defines the unnormalized log density to sample from.
ProposalKernel
    Defines how candidate states are generated and transition probabilities
    are evaluated.

Design notes
------------
- ``rng`` is passed into ``ProposalKernel.propose`` rather than stored, so
  each call is a pure function and results are reproducible given the same
  generator state.
- ``grad_log_prob`` is optional (raises ``NotImplementedError`` by default);
  subclasses override it only when gradient-based samplers (HMC, MALA) are
  used.
- ``ProposalKernel.is_symmetric`` signals that
  ``log q(x'|x) == log q(x|x')``, allowing the Metropolis–Hastings
  acceptance ratio to skip the transition-probability computation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class TargetDistribution(ABC):
    """Unnormalized target density p̃(x) ∝ p(x).

    Subclasses must implement ``dim`` and ``log_prob``.
    ``grad_log_prob`` is optional and required only for gradient-based
    samplers (HMC, MALA).
    """

    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimension of the sample space."""

    @abstractmethod
    def log_prob(self, x: np.ndarray) -> float:
        """Log unnormalized density: log p̃(x).

        Parameters
        ----------
        x:
            Sample point, shape ``(dim,)``.

        Returns
        -------
        float
            log p̃(x).  Need not be normalized.
        """

    def grad_log_prob(self, x: np.ndarray) -> np.ndarray:
        """Gradient of the log unnormalized density: ∇ log p̃(x).

        Parameters
        ----------
        x:
            Sample point, shape ``(dim,)``.

        Returns
        -------
        np.ndarray
            Gradient vector, shape ``(dim,)``.

        Raises
        ------
        NotImplementedError
            Default implementation; override for HMC / MALA.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement grad_log_prob. "
            "Override this method to use gradient-based samplers (HMC, MALA)."
        )


class ProposalKernel(ABC):
    """Markov transition kernel q(x' | x).

    Subclasses must implement ``propose`` and ``log_transition_prob``.
    Override ``is_symmetric`` to return ``True`` for kernels where
    q(x'|x) == q(x|x') so that the MH acceptance ratio can skip the
    log-transition-probability computation.
    """

    @abstractmethod
    def propose(self, current: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Draw a candidate state x' ~ q(· | current).

        Parameters
        ----------
        current:
            Current state, shape ``(dim,)``.
        rng:
            NumPy random generator.  Passed in (not stored) to keep the
            method stateless and reproducible.

        Returns
        -------
        np.ndarray
            Proposed state x', shape ``(dim,)``.
        """

    @abstractmethod
    def log_transition_prob(self, x_new: np.ndarray, x_from: np.ndarray) -> float:
        """Log transition probability: log q(x_new | x_from).

        Parameters
        ----------
        x_new:
            Proposed state, shape ``(dim,)``.
        x_from:
            Current state, shape ``(dim,)``.

        Returns
        -------
        float
            log q(x_new | x_from).
        """

    @property
    def is_symmetric(self) -> bool:
        """Return ``True`` if q(x'|x) == q(x|x') for all x, x'.

        When ``True``, the Metropolis–Hastings acceptance ratio reduces to
        the simple form min(1, p̃(x') / p̃(x)) and
        ``log_transition_prob`` need not be evaluated.

        Defaults to ``False``; override in symmetric kernels such as
        Gaussian random walk.
        """
        return False
