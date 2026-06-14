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

from dataclasses import dataclass

import numpy as np

from gnss.action_engine import (
    DOWNWEIGHT_THRESH,
    HARD_EXCLUDE_THRESH,
    AlertBuilder,
    AlertEvent,
    FailsafeLevel,
    FailsafeManager,
    FailsafeState,
    SatelliteScorer,
)
from gnss.cn0_detector import CN0AnomalyDetector, CN0AnomalyResult
from gnss.constants import _DOPPLER_NOISE_STD, _GRAPH_SIGMA, _INS_VEL_STD
from gnss.math_utils import init_constellation
from gnss.resilience_twin import (
    EpochDiagnosis,
    ResilienceTwin,
    ResilienceTwinConfig,
    run_resilience_simulation,
)
from schemas import FaultClass

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

_CN0_MIN_DBHz: float = 20.0  # minimum acceptable C/N0 [dB-Hz]
_CN0_MAX_DBHz: float = 60.0  # maximum plausible C/N0 [dB-Hz]
_SQM_EXCLUDE_THRESH: float = 0.70  # SQM threshold — exclude satellite if SQM > this
_MIN_SATS_REQUIRED: int = 4  # hard floor for satellite count

# INS trust weights per fault class (index = FaultClass enum value order)
# NOMINAL  MULTIPATH  HARDWARE_FAULT  SPOOFING
_INS_WEIGHT_BY_CLASS: dict[FaultClass, float] = {
    FaultClass.NOMINAL: 0.05,
    FaultClass.MULTIPATH: 0.25,
    FaultClass.HARDWARE_FAULT: 0.60,
    FaultClass.SPOOFING: 0.90,
}

_CONFIDENCE_MC_THRESH: float = 0.60  # trigger MC replay when confidence < this
_MC_REPLAY_N: int = 16  # MC runs for confidence-triggered replay

# ActionPlanner — INS EMA parameters
_EMA_ALPHA: float = 0.30  # smoothing factor (≈ 3-epoch 95% window)
_CONFIDENCE_GATE_THRESH: float = 0.70  # below this, bypass EMA toward raw value

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
    doppler_residuals: np.ndarray  # (n_sats,) [Hz]
    cn0_dbhz: np.ndarray | None = None  # (n_sats,) C/N0 [dB-Hz]
    pseudorange_residuals: np.ndarray | None = None  # (n_sats,) [m]
    sqm: np.ndarray | None = None  # (n_sats,) signal quality metric ∈ [0,1]
    imu_velocity: np.ndarray | None = None  # (3,) IMU velocity deviation [m/s]
    osnma_auth: list[bool] | None = None  # per-satellite OSNMA flags


@dataclass(frozen=True)
class ReceiverObservation:
    """Validated and normalised observation ready for TwinCore.

    Satellites that failed C/N0 or SQM checks are captured in
    ``pre_excluded`` so downstream modules can skip them.
    ``cn0_anomaly`` carries the statistical C/N0 anomaly result when
    cn0_dbhz data was provided; None otherwise.
    """

    epoch: int
    doppler_residuals: np.ndarray  # (n_sats,) validated [Hz]
    ins_velocity: np.ndarray | None  # (3,) from IMU, or None
    osnma_auth: list[bool] | None  # forwarded as-is
    sqm: np.ndarray | None  # (n_sats,), may be None
    pre_excluded: tuple[int, ...]  # satellites failed at RX stage
    n_sats: int  # total satellite count
    cn0_anomaly: CN0AnomalyResult | None = None  # C/N0 anomaly result; None if no C/N0 data


@dataclass(frozen=True)
class TwinDiagnosis:
    """Output of TwinCore per epoch."""

    epoch: int
    epoch_diag: EpochDiagnosis  # full ResilienceTwin output
    mc_auc: float | None  # AUC from MC replay (None if not triggered)


@dataclass(frozen=True)
class ControlAction:
    """Actionable output of ActionPlanner.

    ``excluded_satellites``: indices hard-excluded from the position solution.
    ``satellite_weights``  : per-satellite blend weight ∈ [0, 1]; 0 for excluded,
                             (1−s_i) for downweighted, 1 for accepted.
    ``ins_weight``         : recommended INS blending weight ∈ [0, 1] (EMA-smoothed).
    ``failsafe``           : current failsafe state machine snapshot.
    ``alert``              : structured alert event for this epoch.
    ``reason``             : human-readable rationale string (backward compat).
    """

    epoch: int
    excluded_satellites: tuple[int, ...]  # hard-excluded satellite indices
    n_active: int  # satellites remaining after hard exclusion
    ins_weight: float  # EMA-smoothed INS blending weight ∈ [0, 1]
    diagnosis: FaultClass  # dominant fault class
    confidence: float  # max(fault_posterior)
    reason: str  # plain-language justification
    satellite_weights: tuple[float, ...]  # (n_sats,) per-satellite weights ∈ [0, 1]
    failsafe: FailsafeState  # failsafe state machine snapshot
    alert: AlertEvent  # structured severity-levelled alert


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
        # Stateful C/N0 anomaly detector (warmup adapts per-satellite baseline)
        self._cn0_detector = CN0AnomalyDetector(n_sats=n_sats)

    def process(self, raw: RawEpochData) -> ReceiverObservation:
        """Validate raw data and return a normalised ReceiverObservation.

        Raises:
            ValueError: if doppler_residuals has unexpected length.
        """
        n = len(raw.doppler_residuals)
        if n != self._n_sats:
            raise ValueError(f"ReceiverAgent: expected {self._n_sats} Doppler values, got {n}")

        pre_excluded: list[int] = []
        cn0_anomaly: CN0AnomalyResult | None = None

        # C/N0 gate + anomaly detection
        if raw.cn0_dbhz is not None:
            cn0 = np.asarray(raw.cn0_dbhz, dtype=float)
            for i, c in enumerate(cn0):
                if not (self._cn0_min <= c <= self._cn0_max):
                    pre_excluded.append(i)
            cn0_anomaly = self._cn0_detector.assess(cn0)

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
            cn0_anomaly=cn0_anomaly,
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
            cn0_anomaly=obs.cn0_anomaly,
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

    Satellite exclusion (3-tier soft scoring):
      s_i = 0.60·γ_i + 0.25·𝟙[SQM_i>0.70] + 0.15·(1−auth_i)
      s_i ≥ 0.75  → hard exclude (weight = 0)
      0.40 ≤ s_i  → downweight  (weight = 1−s_i)
      s_i < 0.40  → accept      (weight = 1.0)
      ReceiverAgent pre-excluded satellites are always hard-excluded.
      Fallback: if hard exclusion leaves < min_sats, apply only pre-exclusions;
      if still < min_sats, keep all satellites.

    INS weight (EMA-smoothed, confidence-gated):
      w_raw = Σᵢ P(class_i) · w_i          (posterior-weighted fixed table)
      w_ema ← α·w_raw + (1−α)·w_ema        (cold-start: w_ema = w_raw)
      if confidence ≥ 0.70 → w_ins = w_ema
      else → w_ins = blend·w_ema + (1−blend)·w_raw  (blend = conf/0.70)
      w_ins is then clamped to the failsafe level's [floor, ceil] interval.

    Failsafe state machine:
      Descends immediately on worsening; recovers after
      _FAILSAFE_RECOVERY_EPOCHS consecutive eligible epochs.

    Alert hierarchisation:
      CRITICAL, WARNING, CAUTION, INFO based on spoofing probability,
      number of active alert sources, and current failsafe level.
    """

    def __init__(self, min_sats: int = _MIN_SATS_REQUIRED) -> None:
        self._min_sats = min_sats
        self._scorer = SatelliteScorer()
        self._failsafe_mgr = FailsafeManager(min_sats=min_sats)
        self._alert_builder = AlertBuilder()
        self._ema_ins_weight: float | None = None  # None = uninitialized (cold start)

    def plan(
        self,
        twin_diag: TwinDiagnosis,
        obs: ReceiverObservation,
    ) -> ControlAction:
        diag = twin_diag.epoch_diag
        fp = diag.fault_posterior  # (P_nom, P_mp, P_hw, P_spoof)

        n = obs.n_sats
        all_indices = set(range(n))

        # ----------------------------------------------------------------
        # 1. Satellite soft-exclusion scoring
        # ----------------------------------------------------------------
        scores = self._scorer.score(
            gmm_gamma=diag.integrity.gmm.gamma,
            n_sats=n,
            sqm=obs.sqm,
            osnma_auth=obs.osnma_auth,
        )
        # ReceiverAgent pre-excluded → force hard-exclude
        for idx in obs.pre_excluded:
            scores[idx] = 1.0

        hard_excluded = {i for i in range(n) if scores[i] >= HARD_EXCLUDE_THRESH}
        remaining = all_indices - hard_excluded

        # Fallback 1: too few after soft scoring → only use pre-exclusions
        if len(remaining) < self._min_sats:
            hard_excluded = set(obs.pre_excluded)
            remaining = all_indices - hard_excluded

        # Fallback 2: still too few → keep all satellites
        if len(remaining) < self._min_sats:
            hard_excluded = set()
            remaining = all_indices

        # Per-satellite weights: 0 for hard-excluded, (1−s) downweighted, 1 accepted
        sat_weights = np.zeros(n, dtype=np.float64)
        for i in range(n):
            if i not in hard_excluded:
                if scores[i] >= DOWNWEIGHT_THRESH:
                    sat_weights[i] = 1.0 - float(scores[i])
                else:
                    sat_weights[i] = 1.0

        n_active = len(remaining)

        # ----------------------------------------------------------------
        # 2. INS weight — EMA with confidence gate
        # ----------------------------------------------------------------
        ins_class_weights = [
            _INS_WEIGHT_BY_CLASS[FaultClass.NOMINAL],
            _INS_WEIGHT_BY_CLASS[FaultClass.MULTIPATH],
            _INS_WEIGHT_BY_CLASS[FaultClass.HARDWARE_FAULT],
            _INS_WEIGHT_BY_CLASS[FaultClass.SPOOFING],
        ]
        w_raw = float(np.clip(sum(p * w for p, w in zip(fp, ins_class_weights)), 0.0, 1.0))

        # Cold-start: first call initialises EMA to raw value (preserves old formula)
        if self._ema_ins_weight is None:
            self._ema_ins_weight = w_raw
        else:
            self._ema_ins_weight = _EMA_ALPHA * w_raw + (1.0 - _EMA_ALPHA) * self._ema_ins_weight

        # Confidence gate: low-confidence epochs bypass EMA to stay responsive
        confidence = diag.confidence
        if confidence >= _CONFIDENCE_GATE_THRESH:
            ins_weight = self._ema_ins_weight
        else:
            blend = confidence / _CONFIDENCE_GATE_THRESH
            ins_weight = blend * self._ema_ins_weight + (1.0 - blend) * w_raw
        ins_weight = float(np.clip(ins_weight, 0.0, 1.0))

        # ----------------------------------------------------------------
        # 3. Failsafe state machine
        # ----------------------------------------------------------------
        spoofing_prob = float(fp[3])
        osnma = diag.auth.osnma
        osnma_all_failed = osnma.n_total > 0 and osnma.n_auth == 0

        failsafe = self._failsafe_mgr.update(
            n_active=n_active,
            spoofing_prob=spoofing_prob,
            entropy_alert=diag.entropy.alert,
            osnma_all_failed=osnma_all_failed,
        )

        # Apply failsafe clamping to ins_weight
        ins_weight = float(np.clip(ins_weight, failsafe.ins_weight_floor, failsafe.ins_weight_ceil))

        # ----------------------------------------------------------------
        # 4. Alert event
        # ----------------------------------------------------------------
        alert = self._alert_builder.build(
            epoch=twin_diag.epoch,
            fault_posterior=diag.fault_posterior,
            entropy_alert=diag.entropy.alert,
            osnma_alert=diag.auth.alert,
            phase_alert=diag.structure.phase.phase_alert,
            structure_alert=diag.structure.structural.alert,
            failsafe=failsafe,
            n_active=n_active,
            mc_auc=twin_diag.mc_auc,
        )

        # ----------------------------------------------------------------
        # 5. Reason string (backward compatible)
        # ----------------------------------------------------------------
        diagnosis = diag.diagnosis
        reason_parts: list[str] = [f"diagnosis={diagnosis.value}(conf={confidence:.2f})"]
        if hard_excluded:
            reason_parts.append(f"excluded={sorted(hard_excluded)}")
        if alert.sources:
            reason_parts.append(f"alerts={','.join(alert.sources)}")
        if failsafe.level != FailsafeLevel.NOMINAL:
            reason_parts.append(f"failsafe={failsafe.level.value}")
        if twin_diag.mc_auc is not None:
            reason_parts.append(f"mc_auc={twin_diag.mc_auc:.3f}")

        return ControlAction(
            epoch=twin_diag.epoch,
            excluded_satellites=tuple(sorted(hard_excluded)),
            n_active=n_active,
            ins_weight=ins_weight,
            diagnosis=diagnosis,
            confidence=diag.confidence,
            reason="; ".join(reason_parts),
            satellite_weights=tuple(sat_weights.tolist()),
            failsafe=failsafe,
            alert=alert,
        )


# ---------------------------------------------------------------------------
# Module 4 — MVPPipeline
# ---------------------------------------------------------------------------


@dataclass
class EpochRecord:
    """Per-epoch pipeline record (observation + diagnosis + action)."""

    obs: ReceiverObservation
    twin_diag: TwinDiagnosis
    action: ControlAction


class MVPPipeline:
    """Orchestrates ReceiverAgent → TwinCore → ActionPlanner per epoch.

    Maintains a per-flight history of observations, diagnoses, and actions
    for post-flight analysis or adaptive tuning.

    Usage::

        los = init_constellation(n_sats)
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
            los = init_constellation(n_sats)
        self._receiver = ReceiverAgent(n_sats=n_sats, sqm_thresh=sqm_thresh)
        self._core = TwinCore(
            los=los,
            noise_std=noise_std,
            graph_sigma=graph_sigma,
            ins_noise_std=ins_noise_std,
            mc_replay_n=mc_replay_n,
        )
        self._planner = ActionPlanner(min_sats=min_sats)
        self._history: list[EpochRecord] = []

    def step(self, raw: RawEpochData) -> ControlAction:
        """Process one raw epoch and return a ControlAction.

        Side-effect: appends to ``self.history``.
        """
        obs = self._receiver.process(raw)
        twin_diag = self._core.process(obs)
        action = self._planner.plan(twin_diag, obs)
        self._history.append(EpochRecord(obs=obs, twin_diag=twin_diag, action=action))
        return action

    @property
    def history(self) -> list[EpochRecord]:
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
