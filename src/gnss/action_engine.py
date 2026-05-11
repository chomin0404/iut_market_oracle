"""GNSS Action Engine — Satellite Scoring, Failsafe State Machine,
and Alert Hierarchisation (T1500 Action Layer).

Provides the building blocks consumed by ActionPlanner in mvp.py:

    SatelliteScorer   — per-satellite soft-exclusion score ∈ [0, 1]
    FailsafeManager   — 4-level state machine with chatter-guard recovery
    AlertBuilder      — classifies per-epoch diagnostics into AlertEvent

Satellite exclusion tiers
--------------------------
    s_i ≥ HARD_EXCLUDE_THRESH (0.75) → weight = 0  (hard exclude)
    DOWNWEIGHT_THRESH ≤ s_i < 0.75  → weight = 1 − s_i
    s_i < DOWNWEIGHT_THRESH (0.40)  → weight = 1  (accept)

Failsafe levels (ordered by severity)
--------------------------------------
    NOMINAL        — normal operation; ins_weight unclamped
    DEGRADED       — reduced accuracy; ins_weight ∈ [0.45, 0.70]
    INS_ONLY       — GNSS unreliable; ins_weight fixed at 0.90
    DEAD_RECKONING — GNSS unavailable; ins_weight fixed at 1.00

Descent (worsening) is immediate; ascent (recovery) requires
_FAILSAFE_RECOVERY_EPOCHS consecutive eligible epochs (chatter guard).

Alert levels
------------
    INFO     — no alerts fired
    CAUTION  — one alert source fired
    WARNING  — ≥ 2 sources OR P(spoof) > 0.50
    CRITICAL — P(spoof) > 0.80 OR failsafe ≥ INS_ONLY
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# SatelliteScorer fusion weights (must sum to 1.0)
_SCORE_W_GMM: float = 0.60
_SCORE_W_SQM: float = 0.25
_SCORE_W_OSNMA: float = 0.15
_SCORE_SQM_THRESH: float = 0.70  # SQM > this contributes the full w_sqm term

# Soft-exclusion tier boundaries
HARD_EXCLUDE_THRESH: float = 0.75  # s_i ≥ this → hard exclude
DOWNWEIGHT_THRESH: float = 0.40  # 0.40 ≤ s_i < 0.75 → downweight

# Failsafe transition thresholds
_SPOOFING_DEGRADED_THRESH: float = 0.50  # spoof_prob > this → DEGRADED
_SPOOFING_INS_ONLY_THRESH: float = 0.80  # spoof_prob > this → INS_ONLY / CRITICAL
_FAILSAFE_RECOVERY_EPOCHS: int = 3  # consecutive eligible epochs to ascend

# INS weight clamping per failsafe level: (floor, ceil)
_FAILSAFE_INS_BOUNDS: dict[str, tuple[float, float]] = {
    "nominal": (0.0, 1.0),
    "degraded": (0.45, 0.70),
    "ins_only": (0.90, 0.90),
    "dead_reckoning": (1.0, 1.0),
}

# Alert level thresholds
_ALERT_SOURCES_WARNING: int = 2  # ≥ 2 sources → WARNING

# Severity order (higher index = worse)
_LEVEL_ORDER: tuple[str, ...] = ("nominal", "degraded", "ins_only", "dead_reckoning")


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class FailsafeLevel(str, Enum):
    """Operational safety state of the GNSS position solution."""

    NOMINAL = "nominal"
    DEGRADED = "degraded"
    INS_ONLY = "ins_only"
    DEAD_RECKONING = "dead_reckoning"


class AlertLevel(str, Enum):
    """Severity level for structured alert events."""

    INFO = "info"
    CAUTION = "caution"
    WARNING = "warning"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FailsafeState:
    """Snapshot of the failsafe state machine at one epoch.

    level             : current operational state
    epochs_in_level   : consecutive epochs spent at current level
    recovery_streak   : consecutive epochs meeting recovery criteria
    transitioned      : True if level changed this epoch
    ins_weight_floor  : minimum ins_weight enforced by this level
    ins_weight_ceil   : maximum ins_weight enforced by this level
    """

    level: FailsafeLevel
    epochs_in_level: int
    recovery_streak: int
    transitioned: bool
    ins_weight_floor: float
    ins_weight_ceil: float


@dataclass(frozen=True)
class AlertEvent:
    """Structured per-epoch alert emitted by the action layer.

    level         : INFO / CAUTION / WARNING / CRITICAL
    epoch         : observation epoch index
    sources       : alert source names that fired ("entropy","osnma","phase","structure")
    spoofing_prob : P(spoofing) from fault_posterior[3]
    n_active      : active satellites after exclusion
    failsafe_level: current failsafe level
    mc_auc        : MC replay AUC, None if not triggered
    """

    level: AlertLevel
    epoch: int
    sources: tuple[str, ...]
    spoofing_prob: float
    n_active: int
    failsafe_level: FailsafeLevel
    mc_auc: float | None


# ---------------------------------------------------------------------------
# SatelliteScorer
# ---------------------------------------------------------------------------


class SatelliteScorer:
    """Compute per-satellite soft-exclusion scores ∈ [0, 1].

    Score formula
    -------------
        s_i = w_gmm · γ_i  +  w_sqm · 𝟙[SQM_i > θ]  +  w_osnma · (1 − auth_i)

    Parameters are fusion weights (must sum to 1.0) and the SQM threshold.
    When a sensor is unavailable (None), its term is omitted and the remaining
    weights are preserved as-is (the score is a lower bound in that case).
    """

    def __init__(
        self,
        w_gmm: float = _SCORE_W_GMM,
        w_sqm: float = _SCORE_W_SQM,
        w_osnma: float = _SCORE_W_OSNMA,
        sqm_thresh: float = _SCORE_SQM_THRESH,
    ) -> None:
        total = w_gmm + w_sqm + w_osnma
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"SatelliteScorer weights must sum to 1.0, got {total:.6f}")
        self._w_gmm = w_gmm
        self._w_sqm = w_sqm
        self._w_osnma = w_osnma
        self._sqm_thresh = sqm_thresh

    def score(
        self,
        gmm_gamma: tuple[float, ...],
        n_sats: int,
        sqm: np.ndarray | None = None,
        osnma_auth: list[bool] | None = None,
    ) -> np.ndarray:
        """Return (n_sats,) float64 exclusion-score array ∈ [0, 1]."""
        s = np.zeros(n_sats, dtype=np.float64)

        gamma = np.array(gmm_gamma[:n_sats], dtype=np.float64)
        s += self._w_gmm * np.clip(gamma, 0.0, 1.0)

        if sqm is not None:
            s += self._w_sqm * (np.asarray(sqm[:n_sats], dtype=np.float64) > self._sqm_thresh)

        if osnma_auth is not None:
            auth = np.array(osnma_auth[:n_sats], dtype=np.float64)
            s += self._w_osnma * (1.0 - auth)

        return np.clip(s, 0.0, 1.0)


# ---------------------------------------------------------------------------
# FailsafeManager
# ---------------------------------------------------------------------------


def _level_index(level: FailsafeLevel) -> int:
    """Return numeric severity index (higher = worse)."""
    return _LEVEL_ORDER.index(level.value)


class FailsafeManager:
    """4-level failsafe state machine with chatter-guard recovery.

    Descent (worsening) is immediate.
    Ascent (recovery) requires ``recovery_thresh`` consecutive eligible epochs.

    Transition criteria
    --------------------
    DEAD_RECKONING : n_active == 0  OR  (osnma available AND all failed)
    INS_ONLY       : n_active < min_sats  OR  spoof_prob > 0.80
    DEGRADED       : n_active < min_sats+1  OR  entropy_alert  OR  spoof_prob > 0.50
    NOMINAL        : otherwise
    """

    def __init__(
        self,
        min_sats: int,
        recovery_thresh: int = _FAILSAFE_RECOVERY_EPOCHS,
    ) -> None:
        self._min_sats = min_sats
        self._recovery_thresh = recovery_thresh
        self._level = FailsafeLevel.NOMINAL
        self._epochs_in_level: int = 0
        self._recovery_streak: int = 0

    @property
    def current_level(self) -> FailsafeLevel:
        return self._level

    def update(
        self,
        n_active: int,
        spoofing_prob: float,
        entropy_alert: bool,
        osnma_all_failed: bool,
    ) -> FailsafeState:
        """Advance state machine by one epoch and return updated FailsafeState."""
        prev = self._level
        target = self._target_level(n_active, spoofing_prob, entropy_alert, osnma_all_failed)
        t_idx = _level_index(target)
        c_idx = _level_index(self._level)

        if t_idx > c_idx:
            # Descend immediately
            self._level = target
            self._epochs_in_level = 1
            self._recovery_streak = 0
        elif t_idx < c_idx:
            # Recovery candidate — ascend only when streak threshold is met
            self._recovery_streak += 1
            self._epochs_in_level += 1
            if self._recovery_streak >= self._recovery_thresh:
                self._level = target
                self._recovery_streak = 0
                self._epochs_in_level = 1
        else:
            # Same level
            self._epochs_in_level += 1
            self._recovery_streak = 0

        floor, ceil_ = _FAILSAFE_INS_BOUNDS[self._level.value]
        return FailsafeState(
            level=self._level,
            epochs_in_level=self._epochs_in_level,
            recovery_streak=self._recovery_streak,
            transitioned=self._level != prev,
            ins_weight_floor=floor,
            ins_weight_ceil=ceil_,
        )

    def _target_level(
        self,
        n_active: int,
        spoofing_prob: float,
        entropy_alert: bool,
        osnma_all_failed: bool,
    ) -> FailsafeLevel:
        if n_active == 0 or osnma_all_failed:
            return FailsafeLevel.DEAD_RECKONING
        if n_active < self._min_sats or spoofing_prob > _SPOOFING_INS_ONLY_THRESH:
            return FailsafeLevel.INS_ONLY
        if (
            n_active < self._min_sats + 1
            or entropy_alert
            or spoofing_prob > _SPOOFING_DEGRADED_THRESH
        ):
            return FailsafeLevel.DEGRADED
        return FailsafeLevel.NOMINAL

    def reset(self) -> None:
        """Reset to initial NOMINAL state."""
        self._level = FailsafeLevel.NOMINAL
        self._epochs_in_level = 0
        self._recovery_streak = 0


# ---------------------------------------------------------------------------
# AlertBuilder
# ---------------------------------------------------------------------------


class AlertBuilder:
    """Map per-epoch diagnostic signals to a structured AlertEvent.

    Level assignment (evaluated top-down)
    ----------------------------------------
    CRITICAL : P(spoof) > 0.80  OR  failsafe ∈ {INS_ONLY, DEAD_RECKONING}
    WARNING  : P(spoof) > 0.50  OR  len(sources) ≥ 2
    CAUTION  : any source fired
    INFO     : no sources
    """

    def build(
        self,
        epoch: int,
        fault_posterior: tuple[float, float, float, float],
        entropy_alert: bool,
        osnma_alert: bool,
        phase_alert: bool,
        structure_alert: bool,
        failsafe: FailsafeState,
        n_active: int,
        mc_auc: float | None,
    ) -> AlertEvent:
        spoofing_prob = float(fault_posterior[3])

        sources: list[str] = []
        if entropy_alert:
            sources.append("entropy")
        if osnma_alert:
            sources.append("osnma")
        if phase_alert:
            sources.append("phase")
        if structure_alert:
            sources.append("structure")

        if spoofing_prob > _SPOOFING_INS_ONLY_THRESH or failsafe.level in (
            FailsafeLevel.INS_ONLY,
            FailsafeLevel.DEAD_RECKONING,
        ):
            level = AlertLevel.CRITICAL
        elif spoofing_prob > _SPOOFING_DEGRADED_THRESH or len(sources) >= _ALERT_SOURCES_WARNING:
            level = AlertLevel.WARNING
        elif sources:
            level = AlertLevel.CAUTION
        else:
            level = AlertLevel.INFO

        return AlertEvent(
            level=level,
            epoch=epoch,
            sources=tuple(sources),
            spoofing_prob=spoofing_prob,
            n_active=n_active,
            failsafe_level=failsafe.level,
            mc_auc=mc_auc,
        )
