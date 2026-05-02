"""GNSS Resilience MVP Pipeline (T1500 — 4-module architecture).

Wires four modules into a single processing chain per epoch:

  ReceiverAgent  →  TwinCore  →  ActionPlanner
       ↑                              ↓
  RawEpochData            ControlAction (exclusion mask + INS weight)

MVPPipeline orchestrates the chain and maintains multi-epoch history.

Signal domains:
  C/N0            [dB-Hz]   — carrier-to-noise ratio, 1 per satellite
  Doppler         [Hz]      — residuals after predicted removal
  Pseudorange     [m]       — residuals after model subtraction
  SQM             [0,1]     — signal-quality metric (0=OK, 1=degraded)
  IMU velocity    [m/s]     — 3-D body-frame velocity deviation from INS
  OSNMA           bool list — per-satellite Galileo authentication flags
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from gnss.resilience_twin import (
    EpochDiagnosis,
    ResilienceTwin,
    _DOPPLER_NOISE_STD,
    _GRAPH_SIGMA,
    _INS_VEL_STD,
    run_resilience_simulation,
    ResilienceTwinConfig,
)
from gnss.spoof_sim import _init_constellation
from schemas import FaultClass

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

_CN0_MIN_DBHz: float = 20.0       # minimum acceptable C/N0 [dB-Hz]
_CN0_MAX_DBHz: float = 60.0       # maximum plausible C/N0 [dB-Hz]
_SQM_EXCLUDE_THRESH: float = 0.70 # SQM threshold — exclude satellite if SQM > this
_MIN_SATS_REQUIRED: int = 4        # hard floor for satellite count

# INS trust weights per fault class (index = FaultClass enum value order)
# NOMINAL  MULTIPATH  HARDWARE_FAULT  SPOOFING
_INS_WEIGHT_BY_CLASS: dict[FaultClass, float] = {
    FaultClass.NOMINAL: 0.05,
    FaultClass.MULTIPATH: 0.25,
    FaultClass.HARDWARE_FAULT: 0.60,
    FaultClass.SPOOFING: 0.90,
}

_CONFIDENCE_MC_THRESH: float = 0.60   # trigger MC replay when confidence < this
_MC_REPLAY_N: int = 16                 # MC runs for confidence-triggered replay

# ---------------------------------------------------------------------------
# Shared data classes
# ---------------------------------------------------------------------------


@dataclass
class RawEpochData:
    """Raw multi-sensor observations for one epoch.

    All arrays are length n_sats unless otherwise noted.
    Optional fields default to None when the sensor is unavailable.
    """

    epoch: int
    doppler_residuals: np.ndarray          # (n_sats,) [Hz]
    cn0_dbhz: np.ndarray | None = None     # (n_sats,) C/N0 [dB-Hz]
    pseudorange_residuals: np.ndarray | None = None  # (n_sats,) [m]
    sqm: np.ndarray | None = None          # (n_sats,) signal quality metric ∈ [0,1]
    imu_velocity: np.ndarray | None = None # (3,) IMU velocity deviation [m/s]
    osnma_auth: list[bool] | None = None   # per-satellite OSNMA flags


@dataclass(frozen=True)
class ReceiverObservation:
    """Validated and normalised observation ready for TwinCore.

    Satellites that failed C/N0 or SQM checks are captured in
    ``pre_excluded`` so downstream modules can skip them.
    """

    epoch: int
    doppler_residuals: np.ndarray          # (n_sats,) validated [Hz]
    ins_velocity: np.ndarray | None        # (3,) from IMU, or None
    osnma_auth: list[bool] | None          # forwarded as-is
    sqm: np.ndarray | None                 # (n_sats,), may be None
    pre_excluded: tuple[int, ...]          # satellites failed at RX stage
    n_sats: int                            # total satellite count


@dataclass(frozen=True)
class TwinDiagnosis:
    """Output of TwinCore per epoch."""

    epoch: int
    epoch_diag: EpochDiagnosis             # full ResilienceTwin output
    mc_auc: float | None                   # AUC from MC replay (None if not triggered)


@dataclass(frozen=True)
class ControlAction:
    """Actionable output of ActionPlanner.

    ``excluded_satellites``: indices to drop from position solution.
    ``ins_weight``: recommended INS blending weight ∈ [0, 1].
    ``reason``: human-readable rationale string.
    """

    epoch: int
    excluded_satellites: tuple[int, ...]   # satellite indices to exclude
    n_active: int                          # satellites remaining after exclusion
    ins_weight: float                      # blending weight for INS ∈ [0, 1]
    diagnosis: FaultClass                  # dominant fault class
    confidence: float                      # max(fault_posterior)
    reason: str                            # plain-language justification


# ---------------------------------------------------------------------------
# Module 1 — ReceiverAgent
# ---------------------------------------------------------------------------


class ReceiverAgent:
    """Validate and normalise raw multi-sensor epoch data.

    Checks per-satellite C/N0 bounds and SQM quality gate.
    Converts IMU velocity vector to INS coupling format.
    Forwards OSNMA flags and SQM array without modification.
    """

    def __init__(
        self,
        n_sats: int,
        cn0_min: float = _CN0_MIN_DBHz,
        cn0_max: float = _CN0_MAX_DBHz,
        sqm_thresh: float = _SQM_EXCLUDE_THRESH,
    ) -> None:
        self._n_sats = n_sats
        self._cn0_min = cn0_min
        self._cn0_max = cn0_max
        self._sqm_thresh = sqm_thresh

    def process(self, raw: RawEpochData) -> ReceiverObservation:
        """Validate raw data and return a normalised ReceiverObservation.

        Raises:
            ValueError: if doppler_residuals has unexpected length.
        """
        n = len(raw.doppler_residuals)
        if n != self._n_sats:
            raise ValueError(
                f"ReceiverAgent: expected {self._n_sats} Doppler values, got {n}"
            )

        pre_excluded: list[int] = []

        # C/N0 gate
        if raw.cn0_dbhz is not None:
            cn0 = np.asarray(raw.cn0_dbhz, dtype=float)
            for i, c in enumerate(cn0):
                if not (self._cn0_min <= c <= self._cn0_max):
                    pre_excluded.append(i)

        # SQM gate (exclude high-SQM satellites)
        sqm_arr: np.ndarray | None = None
        if raw.sqm is not None:
            sqm_arr = np.asarray(raw.sqm, dtype=float)
            for i, q in enumerate(sqm_arr):
                if q > self._sqm_thresh and i not in pre_excluded:
                    pre_excluded.append(i)

        doppler = np.asarray(raw.doppler_residuals, dtype=float).copy()
        # Zero out pre-excluded channels so they don't corrupt statistics
        for idx in pre_excluded:
            doppler[idx] = 0.0

        ins_velocity: np.ndarray | None = None
        if raw.imu_velocity is not None:
            ins_velocity = np.asarray(raw.imu_velocity, dtype=float)

        return ReceiverObservation(
            epoch=raw.epoch,
            doppler_residuals=doppler,
            ins_velocity=ins_velocity,
            osnma_auth=raw.osnma_auth,
            sqm=sqm_arr,
            pre_excluded=tuple(sorted(set(pre_excluded))),
            n_sats=n,
        )


# ---------------------------------------------------------------------------
# Module 2 — TwinCore
# ---------------------------------------------------------------------------


class TwinCore:
    """Posterior estimation via ResilienceTwin + optional MC replay.

    The underlying ResilienceTwin is stateful across epochs (IMM-KF, FaultEntropyMonitor,
    StructuralDependencyMonitor all carry state).  Instantiate one TwinCore per flight.

    MC replay is triggered when confidence < _CONFIDENCE_MC_THRESH to give the
    ActionPlanner a more stable AUC estimate for logging / alerting.
    """

    def __init__(
        self,
        los: np.ndarray,
        noise_std: float = _DOPPLER_NOISE_STD,
        graph_sigma: float = _GRAPH_SIGMA,
        ins_noise_std: float = _INS_VEL_STD,
        mc_replay_n: int = _MC_REPLAY_N,
    ) -> None:
        self._twin = ResilienceTwin(
            los=los,
            noise_std=noise_std,
            graph_sigma=graph_sigma,
            ins_noise_std=ins_noise_std,
        )
        self._n_sats = len(los)
        self._mc_replay_n = mc_replay_n
        self._noise_std = noise_std
        self._graph_sigma = graph_sigma

    def process(self, obs: ReceiverObservation) -> TwinDiagnosis:
        """Run one epoch through the 4-pillar stack.

        If confidence < threshold, also runs a short MC simulation and stores AUC.
        """
        diag = self._twin.step(
            doppler_dev=obs.doppler_residuals,
            t=obs.epoch,
            ins_velocity=obs.ins_velocity,
            osnma_auth=obs.osnma_auth,
        )

        mc_auc: float | None = None
        if diag.confidence < _CONFIDENCE_MC_THRESH and self._mc_replay_n > 0:
            cfg = ResilienceTwinConfig(
                n_mc=self._mc_replay_n,
                n_epochs=10,
                n_sats=self._n_sats,
                doppler_noise_std=self._noise_std,
                graph_sigma=self._graph_sigma,
                random_seed=obs.epoch,
            )
            report = run_resilience_simulation(cfg)
            mc_auc = report.auc

        return TwinDiagnosis(epoch=obs.epoch, epoch_diag=diag, mc_auc=mc_auc)


# ---------------------------------------------------------------------------
# Module 3 — ActionPlanner
# ---------------------------------------------------------------------------


class ActionPlanner:
    """Map TwinDiagnosis + ReceiverObservation to a ControlAction.

    Satellite exclusion policy (priority order):
      1. Pre-excluded by ReceiverAgent (C/N0 / SQM)
      2. Excluded by Huh D-optimal subset selector (GM-RAIM fault flags)
      Fallback: if fewer than _MIN_SATS_REQUIRED remain, revert to ReceiverAgent
      exclusions only (preserves geometric redundancy floor).

    INS weight:
      w_ins = Σᵢ P(class_i) · w_i
      where w_i ∈ {0.05, 0.25, 0.60, 0.90} indexed by FaultClass.
    """

    def __init__(self, min_sats: int = _MIN_SATS_REQUIRED) -> None:
        self._min_sats = min_sats

    def plan(
        self,
        twin_diag: TwinDiagnosis,
        obs: ReceiverObservation,
    ) -> ControlAction:
        diag = twin_diag.epoch_diag
        fp = diag.fault_posterior          # (P_nom, P_mp, P_hw, P_spoof)
        huh = diag.integrity.huh

        n = obs.n_sats
        all_indices = set(range(n))

        # --- Satellite exclusion ---
        # Huh excluded set is complement of selected_subset
        huh_selected = set(huh.selected_subset)
        huh_excluded = all_indices - huh_selected
        # Union with ReceiverAgent pre-exclusions
        combined_excluded = huh_excluded | set(obs.pre_excluded)
        remaining = all_indices - combined_excluded

        # Fallback: if too few satellites remain after Huh+RX exclusion,
        # only apply RX-level exclusions (honour geometry floor)
        if len(remaining) < self._min_sats:
            combined_excluded = set(obs.pre_excluded)
            remaining = all_indices - combined_excluded

        # Final floor: if still below minimum, keep everything
        if len(remaining) < self._min_sats:
            combined_excluded = set()
            remaining = all_indices

        # --- INS weight ---
        # Posterior-weighted sum over fault classes
        weights = [
            _INS_WEIGHT_BY_CLASS[FaultClass.NOMINAL],
            _INS_WEIGHT_BY_CLASS[FaultClass.MULTIPATH],
            _INS_WEIGHT_BY_CLASS[FaultClass.HARDWARE_FAULT],
            _INS_WEIGHT_BY_CLASS[FaultClass.SPOOFING],
        ]
        ins_weight = float(sum(p * w for p, w in zip(fp, weights)))
        ins_weight = float(np.clip(ins_weight, 0.0, 1.0))

        # --- Reason string ---
        diagnosis = diag.diagnosis
        reason_parts: list[str] = [f"diagnosis={diagnosis.value}(conf={diag.confidence:.2f})"]
        if combined_excluded:
            reason_parts.append(f"excluded={sorted(combined_excluded)}")
        if diag.entropy.alert:
            reason_parts.append("entropy_alert")
        if diag.auth.alert:
            reason_parts.append("osnma_alert")
        if diag.structure.phase.phase_alert:
            reason_parts.append("phase_alert")
        if twin_diag.mc_auc is not None:
            reason_parts.append(f"mc_auc={twin_diag.mc_auc:.3f}")
        reason = "; ".join(reason_parts)

        return ControlAction(
            epoch=twin_diag.epoch,
            excluded_satellites=tuple(sorted(combined_excluded)),
            n_active=len(remaining),
            ins_weight=ins_weight,
            diagnosis=diagnosis,
            confidence=diag.confidence,
            reason=reason,
        )


# ---------------------------------------------------------------------------
# Module 4 — MVPPipeline
# ---------------------------------------------------------------------------


@dataclass
class _EpochRecord:
    """Internal history entry."""

    obs: ReceiverObservation
    twin_diag: TwinDiagnosis
    action: ControlAction


class MVPPipeline:
    """Orchestrates ReceiverAgent → TwinCore → ActionPlanner per epoch.

    Maintains a per-flight history of observations, diagnoses, and actions
    for post-flight analysis or adaptive tuning.

    Usage::

        los = _init_constellation(n_sats)
        pipeline = MVPPipeline(n_sats=6, los=los)
        for raw in epoch_stream:
            action = pipeline.step(raw)
            apply_action(action)
    """

    def __init__(
        self,
        n_sats: int,
        los: np.ndarray | None = None,
        noise_std: float = _DOPPLER_NOISE_STD,
        graph_sigma: float = _GRAPH_SIGMA,
        ins_noise_std: float = _INS_VEL_STD,
        sqm_thresh: float = _SQM_EXCLUDE_THRESH,
        mc_replay_n: int = _MC_REPLAY_N,
        min_sats: int = _MIN_SATS_REQUIRED,
    ) -> None:
        if los is None:
            los = _init_constellation(n_sats)
        self._receiver = ReceiverAgent(
            n_sats=n_sats, sqm_thresh=sqm_thresh
        )
        self._core = TwinCore(
            los=los,
            noise_std=noise_std,
            graph_sigma=graph_sigma,
            ins_noise_std=ins_noise_std,
            mc_replay_n=mc_replay_n,
        )
        self._planner = ActionPlanner(min_sats=min_sats)
        self._history: list[_EpochRecord] = []

    def step(self, raw: RawEpochData) -> ControlAction:
        """Process one raw epoch and return a ControlAction.

        Side-effect: appends to ``self.history``.
        """
        obs = self._receiver.process(raw)
        twin_diag = self._core.process(obs)
        action = self._planner.plan(twin_diag, obs)
        self._history.append(_EpochRecord(obs=obs, twin_diag=twin_diag, action=action))
        return action

    @property
    def history(self) -> list[_EpochRecord]:
        return self._history

    def dominant_diagnosis(self) -> FaultClass:
        """Return the most frequent diagnosis across all processed epochs."""
        if not self._history:
            return FaultClass.NOMINAL
        counts: dict[FaultClass, int] = {}
        for rec in self._history:
            fc = rec.action.diagnosis
            counts[fc] = counts.get(fc, 0) + 1
        return max(counts, key=lambda k: counts[k])

    def mean_ins_weight(self) -> float:
        """Mean INS blending weight across processed epochs."""
        if not self._history:
            return 0.0
        return float(np.mean([rec.action.ins_weight for rec in self._history]))
