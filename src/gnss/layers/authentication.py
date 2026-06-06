"""GNSS Resilience Twin — Layer 7: OSNMA Galileo Authentication.

Computes the fraction of satellites with verified OSNMA authentication tags.
Defaults to fully authenticated (fraction = 1.0, contribution = 0.0) when
no OSNMA data is supplied (GPS-only or non-Galileo receiver).
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_OSNMA_AUTH_FRAC_THRESH: float = 0.50  # alert if fewer than 50% authenticated

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OSNMALayerResult:
    """Output of OSNMA Galileo authentication layer (Layer 7).

    auth_fraction = n_auth / n_total   (defaults to 1.0 when no OSNMA data)
    p_spoof_contribution = 1 − auth_fraction  (used as fusion signal)
    """

    auth_fraction: float  # fraction of authenticated satellites ∈ [0, 1]
    p_spoof_contribution: float  # 1 − auth_fraction
    n_auth: int  # number of authenticated satellites
    n_total: int  # total satellites checked (0 if no data)
    alert: bool  # True if auth_fraction < _OSNMA_AUTH_FRAC_THRESH


# ---------------------------------------------------------------------------
# Layer 7 — OSNMA Authentication Layer
# ---------------------------------------------------------------------------


class OSNMALayer:
    """Galileo OSNMA authentication coverage monitor (Layer 7).

    Alert threshold: < 50 % authenticated satellites.
    """

    def __init__(self, alert_thresh: float = _OSNMA_AUTH_FRAC_THRESH) -> None:
        self._thresh = alert_thresh

    def assess(self, osnma_auth: list[bool] | None) -> OSNMALayerResult:
        """Evaluate OSNMA authentication coverage for the current epoch.

        Args:
            osnma_auth: Per-satellite boolean authentication flags, or None.
        """
        if osnma_auth is None or len(osnma_auth) == 0:
            return OSNMALayerResult(
                auth_fraction=1.0,
                p_spoof_contribution=0.0,
                n_auth=0,
                n_total=0,
                alert=False,
            )
        n_total = len(osnma_auth)
        n_auth = sum(osnma_auth)
        auth_fraction = n_auth / n_total
        return OSNMALayerResult(
            auth_fraction=auth_fraction,
            p_spoof_contribution=1.0 - auth_fraction,
            n_auth=n_auth,
            n_total=n_total,
            alert=auth_fraction < self._thresh,
        )
