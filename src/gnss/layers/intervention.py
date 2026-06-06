"""GNSS Resilience Twin — Layer 4: Fault Entropy Monitor.

Shannon entropy + KL divergence monitor on the 4-class fault posterior.
Used as Pillar 4 (Intervention) in the ResilienceTwin stack.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EPS: float = 1e-300

# 4-class prior: [nominal, multipath, hw_fault, spoofing]
_FEL_PRIOR: tuple[float, float, float, float] = (0.97, 0.01, 0.01, 0.01)
_FEL_H_THRESH: float = 0.8 * math.log(4.0)  # entropy alert threshold [nats]
_FEL_KL_THRESH: float = 1.0  # KL divergence alert threshold [nats]
_FEL_GRAD_THRESH: float = 0.3  # |ΔH| alert threshold [nats/epoch]

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FaultEntropyResult:
    """Output of fault entropy monitor per epoch."""

    entropy: float  # H(π) [nats]
    kl: float  # KL(π ‖ π₀) [nats]
    alert: bool  # True if any threshold exceeded


# ---------------------------------------------------------------------------
# Layer 4 — Fault Entropy Monitor
# ---------------------------------------------------------------------------


class FaultEntropyMonitor:
    """Shannon entropy + KL divergence monitor on the 4-class fault posterior.

    Alerts when:
        H(π) > H_thresh        — high classification uncertainty
        KL(π ‖ π₀) > kl_thresh — large deviation from nominal prior
        |ΔH| > grad_thresh     — rapid entropy change between epochs
    """

    def __init__(
        self,
        prior: tuple[float, float, float, float] = _FEL_PRIOR,
        h_thresh: float = _FEL_H_THRESH,
        kl_thresh: float = _FEL_KL_THRESH,
        grad_thresh: float = _FEL_GRAD_THRESH,
    ) -> None:
        pi0 = np.array(prior, dtype=float)
        self._pi0 = pi0 / pi0.sum()
        self._h_thresh = h_thresh
        self._kl_thresh = kl_thresh
        self._grad_thresh = grad_thresh
        self._prev_h: float | None = None

    def update(self, fault_probs: np.ndarray) -> FaultEntropyResult:
        """Update monitor with current 4-class fault posterior.

        Args:
            fault_probs: (4,) probability vector [P_nom, P_mp, P_hw, P_spoof]
        """
        pi = np.clip(fault_probs, _EPS, 1.0)
        pi = pi / pi.sum()

        h = float(-np.sum(pi * np.log(pi)))
        kl = float(np.sum(pi * np.log(pi / self._pi0)))

        delta_h = abs(h - self._prev_h) if self._prev_h is not None else 0.0
        self._prev_h = h

        alert = h > self._h_thresh or kl > self._kl_thresh or delta_h > self._grad_thresh
        return FaultEntropyResult(entropy=h, kl=kl, alert=alert)
