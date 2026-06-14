"""Correlation peak monitor — Early-Late ratio + GLRT change-point detector.

Two complementary L2 signal-processing tests that detect non-genuine signals
from correlation-domain evidence:

Sub-monitor 1 — Early-Late (E-L) Power Ratio
---------------------------------------------
The GNSS receiver correlator computes the correlation between the incoming
signal and time-shifted local replicas (Early, Prompt, Late).

Under a genuine signal the correlation function is approximately symmetric
(triangular for BPSK codes), so P_early / P_late ≈ 1.

Under spoofing or severe multipath the peak becomes asymmetric:
    - Meaconing/spoofing: artificial signal superimposed with an offset
      → systematic early/late imbalance depending on code-phase offset.
    - Severe multipath: secondary reflection shifts correlation centroid.

Metric (per satellite i):
    el_ratio_i = P_early_i / (P_late_i + ε)
    el_dev_i   = |el_ratio_i − 1|
Ensemble RMS alert:
    EL_RMS = RMS(el_dev)  >  el_rms_thresh

Ref: Borre et al. (2007) A Software-Defined GPS and Galileo Receiver §7.3;
     Psiaki & Humphreys (2016) Proc. IEEE §V-C.

Sub-monitor 2 — GLRT Change-Point Detector
-------------------------------------------
The Generalised Likelihood Ratio Test (GLRT) detects an abrupt shift in the
mean of C/N₀ observations within a sliding window.

Hypothesis:
    H₀: C/N₀_t ~ N(μ₀, σ²)   for all t in window
    H₁: C/N₀_t ~ N(μ₁, σ²) before change-point τ,
                 N(μ₂, σ²) after τ    (μ₁ ≠ μ₂)

Test statistic (Wald form):
    T_GLRT(τ) = n₁·n₂/(n₁+n₂) · (ȳ₁ − ȳ₂)² / σ̂²

    where n₁, n₂ are sub-window sizes, ȳ₁, ȳ₂ are sub-window means,
    and σ̂² is the pooled variance (estimated from the full window).

Decision:
    T_max = max_τ T_GLRT(τ)  >  glrt_thresh  →  alarm

Under H₀, T_max is approximately χ²(1) for large windows.

Ref: Page (1954) Biometrika; Kay (1998) Fundamentals of Statistical Signal
     Processing §II, §15.7.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EPS: float = 1e-12

# Early-Late ratio sub-monitor
EL_RMS_THRESH: float = 0.15  # EL deviation RMS alarm threshold
EL_SAT_THRESH: float = 0.30  # per-satellite EL deviation alarm threshold

# GLRT sub-monitor
GLRT_WINDOW: int = 30  # sliding window length W for GLRT [epochs]
GLRT_THRESH: float = 6.635  # χ²(1, 0.99) ≈ 6.635

# Minimum window split size to avoid degenerate single-sample sub-windows
_GLRT_MIN_SPLIT: int = 2

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorrelationMonitorResult:
    """Per-epoch output of the correlation peak monitor.

    Fields
    ------
    epoch           Epoch index passed to assess().
    el_rms          RMS of (|P_early/P_late − 1|) across all satellites.
    el_alarm        True when el_rms > EL_RMS_THRESH.
    el_sat_flags    Per-satellite bool: True = E-L ratio deviation exceeded.
    glrt_stat       Max GLRT statistic T_max over the C/N₀ window.
    glrt_alarm      True when glrt_stat > GLRT_THRESH.
    alarm           el_alarm OR glrt_alarm.
    quality_score   max(el_score, glrt_score) ∈ [0, 1].
    reasons         List of triggered alert labels.
    """

    epoch: int
    el_rms: float
    el_alarm: bool
    el_sat_flags: list[bool]
    glrt_stat: float
    glrt_alarm: bool
    alarm: bool
    quality_score: float
    reasons: list[str]


# ---------------------------------------------------------------------------
# Correlation peak monitor
# ---------------------------------------------------------------------------


class CorrelationMonitor:
    """Real-time Early-Late ratio + GLRT C/N₀ change-point monitor.

    Usage::

        monitor = CorrelationMonitor()
        result  = monitor.assess(
            epoch=t,
            p_early=[...],   # per-satellite Early power [linear]
            p_late=[...],    # per-satellite Late  power [linear]
            cn0_db=[...],    # ensemble mean C/N₀ for GLRT window update [dB-Hz]
        )

    The GLRT window is updated with one scalar (mean ensemble C/N₀) per epoch.
    Call reset() to clear the window between experiments.
    """

    def __init__(
        self,
        el_rms_thresh: float = EL_RMS_THRESH,
        el_sat_thresh: float = EL_SAT_THRESH,
        glrt_window: int = GLRT_WINDOW,
        glrt_thresh: float = GLRT_THRESH,
    ) -> None:
        self._el_rms_thresh = el_rms_thresh
        self._el_sat_thresh = el_sat_thresh
        self._glrt_window = glrt_window
        self._glrt_thresh = glrt_thresh
        self._cn0_history: list[float] = []

    def reset(self) -> None:
        """Clear the C/N₀ sliding window (call between independent simulations)."""
        self._cn0_history = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assess(
        self,
        epoch: int,
        p_early: list[float] | np.ndarray,
        p_late: list[float] | np.ndarray,
        cn0_db: float,
    ) -> CorrelationMonitorResult:
        """Evaluate E-L ratio and GLRT for the current epoch.

        Args:
            epoch:   Epoch index.
            p_early: Per-satellite Early correlator power [linear, >0].
            p_late:  Per-satellite Late  correlator power [linear, >0].
            cn0_db:  Ensemble mean C/N₀ for this epoch [dB-Hz].

        Returns:
            CorrelationMonitorResult.
        """
        p_e = np.asarray(p_early, dtype=float)
        p_l = np.asarray(p_late, dtype=float)

        # --- Sub-monitor 1: Early-Late ratio ---
        el_ratio = p_e / (p_l + _EPS)
        el_dev = np.abs(el_ratio - 1.0)
        el_rms = float(np.sqrt(np.mean(el_dev**2)))
        el_sat_flags = [bool(d > self._el_sat_thresh) for d in el_dev]
        el_alarm = el_rms > self._el_rms_thresh
        el_score = min(1.0, el_rms / (self._el_rms_thresh + _EPS))

        # --- Sub-monitor 2: GLRT change-point ---
        self._cn0_history.append(float(cn0_db))
        if len(self._cn0_history) > self._glrt_window:
            self._cn0_history = self._cn0_history[-self._glrt_window :]

        glrt_stat, glrt_alarm = self._compute_glrt()
        glrt_score = min(1.0, glrt_stat / (self._glrt_thresh + _EPS))

        # --- Fusion ---
        alarm = el_alarm or glrt_alarm
        quality_score = max(el_score, glrt_score)
        reasons: list[str] = []
        if el_alarm:
            reasons.append(f"el_rms={el_rms:.3f}>{self._el_rms_thresh}")
        if glrt_alarm:
            reasons.append(f"glrt={glrt_stat:.2f}>{self._glrt_thresh}")

        return CorrelationMonitorResult(
            epoch=epoch,
            el_rms=el_rms,
            el_alarm=el_alarm,
            el_sat_flags=el_sat_flags,
            glrt_stat=glrt_stat,
            glrt_alarm=glrt_alarm,
            alarm=alarm,
            quality_score=quality_score,
            reasons=reasons,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute_glrt(self) -> tuple[float, bool]:
        """Compute the max GLRT statistic over the current C/N₀ window.

        Returns:
            (T_max, alarm): GLRT statistic and threshold decision.
        """
        w = np.asarray(self._cn0_history, dtype=float)
        n = len(w)
        if n < 2 * _GLRT_MIN_SPLIT + 1:
            return 0.0, False

        sigma_sq = float(np.var(w, ddof=1)) + _EPS

        t_max = 0.0
        for tau in range(_GLRT_MIN_SPLIT, n - _GLRT_MIN_SPLIT + 1):
            w1 = w[:tau]
            w2 = w[tau:]
            n1, n2 = len(w1), len(w2)
            mean_diff = float(np.mean(w1) - np.mean(w2))
            t_tau = (n1 * n2) / (n1 + n2) * mean_diff**2 / sigma_sq
            if t_tau > t_max:
                t_max = t_tau

        return t_max, t_max > self._glrt_thresh
