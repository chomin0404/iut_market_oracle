"""C/N0 Anomaly Detector for GNSS spoofing detection.

Three independent tests are fused per epoch:

  Test 1 — CUSUM step detector (per satellite)
      Detects sudden mean shifts in per-satellite C/N0 series.
      Statistic: S_t = max(0, S_{t-1} + (x_t - mu_0 - k))
      Alert when any S_t > h  (one-sided upper CUSUM).
      Ref: Page (1954); Montgomery (2009) §9.1

  Test 2 — Ensemble spread collapse
      Genuine constellation: ~5–10 dB-Hz inter-satellite spread from geometry.
      Under meaconing/spoofing, all channels share the same signal → spread collapses.
      Statistic: sigma_t = std(cn0_t) across all active satellites
      Alert when sigma_t < spread_min.

  Test 3 — Pairwise correlation burst  (requires window >= 2 epochs)
      Genuine C/N0 series are weakly correlated (sky geometry drift, independent noise).
      After spoofing takeover, common injection synchronises all channels.
      Statistic: mean |rho_ij| over a rolling W-epoch window.
      Alert when mean_corr > corr_thresh.

Score fusion:
    p_spoof_cn0 = clip( max(cusum_score, spread_score, corr_score), 0, 1 )
    alarm       = cusum_alert | spread_alert | corr_alert
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

_CN0_NOMINAL_DBHz: float = 40.0  # assumed nominal C/N0 [dB-Hz] (fallback pre-warmup)
_CN0_CUSUM_K: float = 0.50  # CUSUM slack (allowance) parameter k [dB-Hz]
_CN0_CUSUM_H: float = 5.00  # CUSUM decision threshold h [dB-Hz]
_CN0_SPREAD_MIN_DBHz: float = 2.50  # minimum genuine inter-satellite spread [dB-Hz]
_CN0_CORR_THRESH: float = 0.85  # mean |rho_ij| threshold for correlation burst
_CN0_WINDOW: int = 20  # rolling window length W [epochs]
_CN0_CORR_MIN_EPOCHS: int = 4  # minimum epochs required before correlation test fires
_CN0_WARMUP_EPOCHS: int = 10  # warm-up epochs used to estimate per-satellite mu0


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CN0AnomalyResult:
    """Output of CN0AnomalyDetector for one epoch.

    Fields
    ------
    alarm           True when any test fires.
    p_spoof_cn0     Fused spoofing probability contribution ∈ [0, 1].
    cusum_alert     CUSUM step-change alert.
    spread_alert    Ensemble spread-collapse alert.
    corr_alert      Pairwise correlation burst alert.
    cusum_score     Normalised CUSUM score (max over satellites) ∈ [0, 1].
    spread_score    1 - sigma_t / spread_min  clamped to [0, 1].
    corr_score      mean |rho_ij| clamped to [0, 1]; NaN before min epochs.
    sigma_t         Inter-satellite C/N0 std [dB-Hz] for this epoch.
    mean_corr       Mean absolute pairwise correlation; NaN before min epochs.
    reasons         Human-readable list of active alert reasons.
    """

    alarm: bool
    p_spoof_cn0: float

    cusum_alert: bool
    spread_alert: bool
    corr_alert: bool

    cusum_score: float
    spread_score: float
    corr_score: float  # NaN until _CN0_CORR_MIN_EPOCHS epochs collected

    sigma_t: float  # [dB-Hz]
    mean_corr: float  # NaN until min epochs

    reasons: tuple[str, ...]


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class CN0AnomalyDetector:
    """Stateful C/N0 anomaly detector.  Call assess() once per epoch.

    Parameters
    ----------
    n_sats : int
        Expected number of satellites.  CUSUM state is reset if the
        received array length changes.
    mu0 : float
        Fallback nominal C/N0 [dB-Hz], used before warmup completes.
        After warmup, per-satellite means estimated from data replace this.
    cusum_k : float
        CUSUM slack parameter k [dB-Hz].
    cusum_h : float
        CUSUM decision threshold h [dB-Hz].
    spread_min : float
        Minimum acceptable inter-satellite C/N0 spread [dB-Hz].
    corr_thresh : float
        Mean |rho_ij| threshold for correlation burst alert.
    window : int
        Rolling window length W for correlation test.
    corr_min_epochs : int
        Minimum epochs before the correlation test is activated.
    warmup_epochs : int
        Number of initial epochs used to estimate per-satellite mu0.
        CUSUM does not fire during warmup.

    Invariants
    ----------
    - CUSUM state S_t >= 0 always.
    - Window deque length <= window always.
    - Per-satellite mu0 is set once after warmup_epochs and held fixed.
    - assess() is pure with respect to external state (only self._* mutated).
    """

    def __init__(
        self,
        n_sats: int,
        mu0: float = _CN0_NOMINAL_DBHz,
        cusum_k: float = _CN0_CUSUM_K,
        cusum_h: float = _CN0_CUSUM_H,
        spread_min: float = _CN0_SPREAD_MIN_DBHz,
        corr_thresh: float = _CN0_CORR_THRESH,
        window: int = _CN0_WINDOW,
        corr_min_epochs: int = _CN0_CORR_MIN_EPOCHS,
        warmup_epochs: int = _CN0_WARMUP_EPOCHS,
    ) -> None:
        self._n_sats = n_sats
        self._mu0_fallback = mu0
        self._k = cusum_k
        self._h = cusum_h
        self._spread_min = spread_min
        self._corr_thresh = corr_thresh
        self._window = window
        self._corr_min = corr_min_epochs
        self._warmup_epochs = warmup_epochs

        # Per-satellite mu0 (None until warmup is complete)
        self._mu0_per_sat: np.ndarray | None = None
        # Warmup buffer accumulates (n_sats,) arrays
        self._warmup_buf: list[np.ndarray] = []

        # CUSUM upper-side accumulators S_t per satellite
        self._cusum_s: np.ndarray = np.zeros(n_sats, dtype=float)

        # Rolling window of C/N0 vectors: deque of (n_sats,) arrays
        self._history: deque[np.ndarray] = deque(maxlen=window)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset CUSUM state, warmup buffer, and history.

        Call after receiver reacquisition or signal loss to restart calibration.
        """
        self._cusum_s = np.zeros(self._n_sats, dtype=float)
        self._history.clear()
        self._mu0_per_sat = None
        self._warmup_buf = []

    def assess(self, cn0_dbhz: np.ndarray) -> CN0AnomalyResult:
        """Evaluate C/N0 anomaly for one epoch.

        Parameters
        ----------
        cn0_dbhz : np.ndarray, shape (n_sats,)
            Carrier-to-noise ratios [dB-Hz].  NaN values are ignored.

        Returns
        -------
        CN0AnomalyResult
        """
        cn0 = np.asarray(cn0_dbhz, dtype=float)

        # Adapt CUSUM state if satellite count changed
        if cn0.shape[0] != self._n_sats:
            self._n_sats = cn0.shape[0]
            self._cusum_s = np.zeros(self._n_sats, dtype=float)
            self._mu0_per_sat = None
            self._warmup_buf = []

        # Valid (non-NaN) mask
        valid = ~np.isnan(cn0)
        cn0_valid = cn0[valid]

        # -- Warmup phase: accumulate per-satellite baseline ----------------
        in_warmup = self._mu0_per_sat is None
        if in_warmup:
            self._warmup_buf.append(cn0.copy())
            if len(self._warmup_buf) >= self._warmup_epochs:
                warmup_mat = np.stack(self._warmup_buf, axis=0)  # (W, n_sats)
                self._mu0_per_sat = np.where(
                    np.any(~np.isnan(warmup_mat), axis=0),
                    np.nanmean(warmup_mat, axis=0),
                    self._mu0_fallback,
                )
                # Reset CUSUM: discard any accumulation from the warmup phase
                # (warmup used fallback mu0; post-warmup uses per-satellite mu0)
                self._cusum_s = np.zeros(self._n_sats, dtype=float)

        # -- Test 1: CUSUM --------------------------------------------------
        cusum_alert, cusum_score = self._update_cusum(cn0, valid, in_warmup)

        # -- Test 2: Spread collapse ----------------------------------------
        spread_alert, spread_score, sigma_t = self._check_spread(cn0_valid)

        # -- Test 3: Pairwise correlation burst -----------------------------
        self._history.append(cn0.copy())
        corr_alert, corr_score, mean_corr = self._check_correlation()

        # -- Fusion ---------------------------------------------------------
        scores = [cusum_score, spread_score]
        if not np.isnan(corr_score):
            scores.append(corr_score)
        p_spoof_cn0 = float(np.clip(max(scores), 0.0, 1.0))
        alarm = cusum_alert or spread_alert or corr_alert

        reasons: list[str] = []
        if cusum_alert:
            reasons.append(f"CUSUM step detected (score={cusum_score:.3f})")
        if spread_alert:
            reasons.append(
                f"C/N0 spread collapse (sigma={sigma_t:.2f} dB-Hz < {self._spread_min:.2f})"
            )
        if corr_alert:
            reasons.append(
                f"Pairwise correlation burst (mean_rho={mean_corr:.3f} > {self._corr_thresh:.3f})"
            )

        return CN0AnomalyResult(
            alarm=alarm,
            p_spoof_cn0=p_spoof_cn0,
            cusum_alert=cusum_alert,
            spread_alert=spread_alert,
            corr_alert=corr_alert,
            cusum_score=cusum_score,
            spread_score=spread_score,
            corr_score=corr_score,
            sigma_t=sigma_t,
            mean_corr=mean_corr,
            reasons=tuple(reasons),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _update_cusum(
        self, cn0: np.ndarray, valid: np.ndarray, in_warmup: bool
    ) -> tuple[bool, float]:
        """One-sided upper CUSUM update with per-satellite baseline.

        S_t^i = max(0, S_{t-1}^i + (x_t^i - mu0_i - k))

        mu0_i = per-satellite mean estimated from warmup data.
               Falls back to global _mu0_fallback before warmup completes.

        During warmup (in_warmup=True): state updates but alert is suppressed.

        Returns (alert, normalised_score).
        normalised_score = max(S_t) / h  clamped to [0, 1].
        """
        mu0_vec: np.ndarray = (
            self._mu0_per_sat
            if self._mu0_per_sat is not None
            else np.full(self._n_sats, self._mu0_fallback)
        )
        for i in range(self._n_sats):
            if valid[i]:
                increment = cn0[i] - mu0_vec[i] - self._k
                self._cusum_s[i] = max(0.0, self._cusum_s[i] + increment)
            # NaN satellite: hold state (no update)

        max_s = float(np.max(self._cusum_s))
        cusum_score = float(np.clip(max_s / self._h, 0.0, 1.0))
        if in_warmup:
            return False, 0.0
        return max_s > self._h, cusum_score

    def _check_spread(self, cn0_valid: np.ndarray) -> tuple[bool, float, float]:
        """Ensemble spread-collapse test.

        sigma_t = std(cn0_valid)
        spread_score = clip(1 - sigma_t / spread_min, 0, 1)
        """
        if cn0_valid.size < 2:
            return False, 0.0, float("nan")

        sigma_t = float(np.std(cn0_valid))
        if self._spread_min <= 0.0:
            return False, 0.0, sigma_t

        spread_score = float(np.clip(1.0 - sigma_t / self._spread_min, 0.0, 1.0))
        alert = sigma_t < self._spread_min
        return alert, spread_score, sigma_t

    def _check_correlation(self) -> tuple[bool, float, float]:
        """Mean absolute pairwise Pearson correlation over the rolling window.

        Returns (alert, corr_score, mean_corr).
        Returns (False, nan, nan) when fewer than _corr_min epochs are available.
        """
        n_ep = len(self._history)
        if n_ep < self._corr_min:
            return False, float("nan"), float("nan")

        # Stack window: shape (n_ep, n_sats)
        mat = np.stack(list(self._history), axis=0)  # (W, n_sats)

        # Remove satellites with zero variance (constant series → skip)
        col_std = np.std(mat, axis=0)
        active_cols = np.where(col_std > 0)[0]
        if len(active_cols) < 2:
            return False, float("nan"), float("nan")

        mat_active = mat[:, active_cols]

        # Pearson correlation matrix (n_active_sats × n_active_sats)
        # np.corrcoef works on rows → transpose
        corr_mat = np.corrcoef(mat_active.T)  # (n_active, n_active)

        # Extract upper triangle (excluding diagonal)
        n_a = corr_mat.shape[0]
        idx_i, idx_j = np.triu_indices(n_a, k=1)
        if len(idx_i) == 0:
            return False, float("nan"), float("nan")

        mean_corr = float(np.mean(np.abs(corr_mat[idx_i, idx_j])))
        corr_score = float(np.clip(mean_corr, 0.0, 1.0))
        alert = mean_corr > self._corr_thresh
        return alert, corr_score, mean_corr
