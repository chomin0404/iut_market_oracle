"""Edge Collector for GNSS Resilience Pipeline (T1500).

Accumulates per-epoch observables, residuals, authentication results,
and event time series from MVPPipeline history into structured snapshots
and stacked numpy arrays for downstream analysis or export.

Signal domains collected
------------------------
Observables    doppler_residuals (n_sats,) [Hz]
               sqm              (n_sats,) ∈ [0,1]   — None when unavailable
               ins_velocity     (3,)      [m/s]      — None when unavailable
               osnma_auth       bool list            — None when unavailable

Residuals      gmm_gamma        (n_sats,)  per-satellite GM-RAIM fault posterior γᵢ
               imm_innovation_norms  (3,)  ‖νₘ‖₂ per IMM mode [nom, mp, spoof]
               imm_mode_weights      (3,)  μ weights  [μ_nom, μ_mp, μ_spoof]

Authentication auth_fraction    scalar    ∈ [0,1]
               n_auth / n_total ints
               osnma_alert      bool

Events         fault_posterior  (4,)  [P_nom, P_mp, P_hw, P_spoof]
               diagnosis        FaultClass
               confidence       scalar  = max(fault_posterior)
               entropy_alert    bool
               structure_alert  bool
               phase_alert      bool
               ins_weight       scalar  ∈ [0,1]
               n_excluded       int
               n_active         int
               mc_auc           float | None  (NaN in arrays when absent)

Note: C/N0 values are consumed by ReceiverAgent but not forwarded in
ReceiverObservation. Use ``pre_excluded`` (captured via n_excluded) as proxy.

Typical usage::

    pipeline = MVPPipeline(n_sats=6, los=los)
    collector = EdgeCollector()
    for raw in epoch_stream:
        pipeline.step(raw)
        collector.collect(pipeline.history[-1])

    arrays = collector.to_arrays()
    # arrays.doppler_residuals  → (n_epochs, n_sats)
    # arrays.fault_posterior    → (n_epochs, 4)
    # arrays.auth_fraction      → (n_epochs,)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gnss.mvp import _EpochRecord
from schemas import FaultClass

# ---------------------------------------------------------------------------
# EdgeSnapshot — single-epoch frozen record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EdgeSnapshot:
    """All collected signals for one epoch.

    All array fields carry a copy of the underlying data (immutable).
    Optional sensor fields are None when the sensor was unavailable for that epoch.
    """

    epoch: int

    # --- Observables ---
    doppler_residuals: np.ndarray           # (n_sats,) [Hz]
    sqm: np.ndarray | None                  # (n_sats,) ∈ [0,1], None if unavailable
    ins_velocity: np.ndarray | None         # (3,) [m/s], None if unavailable
    osnma_auth: tuple[bool, ...] | None     # per-satellite auth flags, None if unavailable
    pre_excluded: tuple[int, ...]           # indices excluded by ReceiverAgent (C/N0 + SQM)

    # --- Residuals ---
    gmm_gamma: tuple[float, ...]            # (n_sats,) per-satellite fault posterior γᵢ
    imm_innovation_norms: tuple[float, float, float]  # ‖νₘ‖₂ per IMM mode
    imm_mode_weights: tuple[float, float, float]      # μ = [μ_nom, μ_mp, μ_spoof]

    # --- Authentication ---
    auth_fraction: float                    # fraction of authenticated satellites ∈ [0,1]
    n_auth: int                             # count of authenticated satellites
    n_total: int                            # total satellites checked (0 if no OSNMA data)
    osnma_alert: bool                       # True if auth_fraction < threshold

    # --- Events ---
    fault_posterior: tuple[float, float, float, float]  # [P_nom, P_mp, P_hw, P_spoof]
    diagnosis: FaultClass
    confidence: float                       # max(fault_posterior)
    entropy_alert: bool
    structure_alert: bool
    phase_alert: bool
    ins_weight: float                       # INS blending weight ∈ [0,1]
    n_excluded: int                         # satellite count excluded from solution
    n_active: int                           # satellite count active in solution
    mc_auc: float | None                    # MC replay AUC; None if not triggered


# ---------------------------------------------------------------------------
# EdgeArrays — stacked numpy arrays over all collected epochs
# ---------------------------------------------------------------------------


@dataclass
class EdgeArrays:
    """All edge signals stacked along axis-0 (epoch axis).

    Shape conventions
    -----------------
    (n_epochs,)        — scalar-per-epoch signals
    (n_epochs, n_sats) — per-satellite signals
    (n_epochs, 3)      — 3-component IMM signals
    (n_epochs, 4)      — 4-class fault posterior

    mc_auc contains NaN for epochs where MC replay was not triggered.
    diagnosis is a 1-D object array of FaultClass enum values.
    """

    epochs: np.ndarray              # (n_epochs,) int

    # Observables
    doppler_residuals: np.ndarray   # (n_epochs, n_sats) float64 [Hz]

    # Residuals
    gmm_gamma: np.ndarray           # (n_epochs, n_sats) float64
    imm_innovation_norms: np.ndarray  # (n_epochs, 3) float64
    imm_mode_weights: np.ndarray    # (n_epochs, 3) float64

    # Authentication
    auth_fraction: np.ndarray       # (n_epochs,) float64
    n_auth: np.ndarray              # (n_epochs,) int64
    n_total: np.ndarray             # (n_epochs,) int64
    osnma_alert: np.ndarray         # (n_epochs,) bool

    # Events
    fault_posterior: np.ndarray     # (n_epochs, 4) float64
    diagnosis: np.ndarray           # (n_epochs,) object — FaultClass
    confidence: np.ndarray          # (n_epochs,) float64
    entropy_alert: np.ndarray       # (n_epochs,) bool
    structure_alert: np.ndarray     # (n_epochs,) bool
    phase_alert: np.ndarray         # (n_epochs,) bool
    ins_weight: np.ndarray          # (n_epochs,) float64
    n_excluded: np.ndarray          # (n_epochs,) int64
    n_active: np.ndarray            # (n_epochs,) int64
    mc_auc: np.ndarray              # (n_epochs,) float64 — NaN when absent

    @property
    def n_epochs(self) -> int:
        return len(self.epochs)

    @property
    def n_sats(self) -> int:
        return self.doppler_residuals.shape[1]

    def event_mask(
        self,
        entropy: bool = True,
        structure: bool = True,
        phase: bool = True,
        osnma: bool = True,
    ) -> np.ndarray:
        """Return boolean mask of epochs where any selected alert fired.

        Parameters
        ----------
        entropy, structure, phase, osnma:
            Include the corresponding alert type in the OR mask.

        Returns
        -------
        np.ndarray of shape (n_epochs,) bool
        """
        mask = np.zeros(self.n_epochs, dtype=bool)
        if entropy:
            mask |= self.entropy_alert
        if structure:
            mask |= self.structure_alert
        if phase:
            mask |= self.phase_alert
        if osnma:
            mask |= self.osnma_alert
        return mask


# ---------------------------------------------------------------------------
# EdgeCollector
# ---------------------------------------------------------------------------


class EdgeCollector:
    """Accumulate per-epoch edge signals from MVPPipeline records.

    Feed records one at a time via ``collect()``, or in bulk via
    ``collect_all(pipeline.history)``.  The collector is stateless with
    respect to the pipeline — it only reads, never writes.

    Parameters
    ----------
    capacity : int, optional
        Pre-allocate the internal snapshot list to this length.
        Has no effect on correctness; only avoids list reallocations for
        large known-length runs.
    """

    def __init__(self, capacity: int = 0) -> None:
        self._snapshots: list[EdgeSnapshot] = []
        if capacity > 0:
            self._snapshots = []  # list growth is fine; capacity is a hint

    # ------------------------------------------------------------------
    # Collection
    # ------------------------------------------------------------------

    def collect(self, record: _EpochRecord) -> None:
        """Append one epoch snapshot from an MVPPipeline _EpochRecord.

        Parameters
        ----------
        record:
            A single entry from ``MVPPipeline.history``.
        """
        obs = record.obs
        td = record.twin_diag
        action = record.action
        diag = td.epoch_diag

        snap = EdgeSnapshot(
            epoch=obs.epoch,
            # Observables
            doppler_residuals=obs.doppler_residuals.copy(),
            sqm=obs.sqm.copy() if obs.sqm is not None else None,
            ins_velocity=obs.ins_velocity.copy() if obs.ins_velocity is not None else None,
            osnma_auth=tuple(obs.osnma_auth) if obs.osnma_auth is not None else None,
            pre_excluded=obs.pre_excluded,
            # Residuals
            gmm_gamma=diag.integrity.gmm.gamma,
            imm_innovation_norms=diag.integrity.imm.innovation_norms,
            imm_mode_weights=diag.integrity.imm.mode_weights,
            # Authentication
            auth_fraction=diag.auth.auth_fraction,
            n_auth=diag.auth.osnma.n_auth,
            n_total=diag.auth.osnma.n_total,
            osnma_alert=diag.auth.alert,
            # Events
            fault_posterior=diag.fault_posterior,
            diagnosis=diag.diagnosis,
            confidence=diag.confidence,
            entropy_alert=diag.entropy.alert,
            structure_alert=diag.structure.structural.alert,
            phase_alert=diag.structure.phase.phase_alert,
            ins_weight=action.ins_weight,
            n_excluded=len(action.excluded_satellites),
            n_active=action.n_active,
            mc_auc=td.mc_auc,
        )
        self._snapshots.append(snap)

    def collect_all(self, history: list[_EpochRecord]) -> None:
        """Collect all records from a pipeline history list.

        Equivalent to calling ``collect(r)`` for each ``r`` in ``history``.
        """
        for record in history:
            self.collect(record)

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    @property
    def snapshots(self) -> list[EdgeSnapshot]:
        """Read-only view of collected snapshots."""
        return self._snapshots

    def __len__(self) -> int:
        return len(self._snapshots)

    def to_arrays(self) -> EdgeArrays:
        """Stack all collected snapshots into numpy arrays.

        Returns
        -------
        EdgeArrays
            All arrays share axis-0 = epoch index.

        Raises
        ------
        ValueError
            If no snapshots have been collected yet.
        """
        if not self._snapshots:
            raise ValueError("EdgeCollector.to_arrays(): no snapshots collected")

        snaps = self._snapshots
        n = len(snaps)
        n_sats = len(snaps[0].doppler_residuals)

        # Pre-allocate
        epochs = np.empty(n, dtype=np.int64)
        doppler = np.zeros((n, n_sats), dtype=np.float64)
        gmm_gamma = np.zeros((n, n_sats), dtype=np.float64)
        imm_inno = np.zeros((n, 3), dtype=np.float64)
        imm_weights = np.zeros((n, 3), dtype=np.float64)
        auth_fraction = np.empty(n, dtype=np.float64)
        n_auth = np.empty(n, dtype=np.int64)
        n_total = np.empty(n, dtype=np.int64)
        osnma_alert = np.empty(n, dtype=bool)
        fault_posterior = np.zeros((n, 4), dtype=np.float64)
        diagnosis = np.empty(n, dtype=object)
        confidence = np.empty(n, dtype=np.float64)
        entropy_alert = np.empty(n, dtype=bool)
        structure_alert = np.empty(n, dtype=bool)
        phase_alert = np.empty(n, dtype=bool)
        ins_weight = np.empty(n, dtype=np.float64)
        n_excluded = np.empty(n, dtype=np.int64)
        n_active_arr = np.empty(n, dtype=np.int64)
        mc_auc = np.full(n, np.nan, dtype=np.float64)

        for i, s in enumerate(snaps):
            epochs[i] = s.epoch
            doppler[i] = s.doppler_residuals
            gmm_gamma[i] = np.fromiter(s.gmm_gamma, dtype=np.float64, count=n_sats)
            imm_inno[i] = s.imm_innovation_norms
            imm_weights[i] = s.imm_mode_weights
            auth_fraction[i] = s.auth_fraction
            n_auth[i] = s.n_auth
            n_total[i] = s.n_total
            osnma_alert[i] = s.osnma_alert
            fault_posterior[i] = s.fault_posterior
            diagnosis[i] = s.diagnosis
            confidence[i] = s.confidence
            entropy_alert[i] = s.entropy_alert
            structure_alert[i] = s.structure_alert
            phase_alert[i] = s.phase_alert
            ins_weight[i] = s.ins_weight
            n_excluded[i] = s.n_excluded
            n_active_arr[i] = s.n_active
            if s.mc_auc is not None:
                mc_auc[i] = s.mc_auc

        return EdgeArrays(
            epochs=epochs,
            doppler_residuals=doppler,
            gmm_gamma=gmm_gamma,
            imm_innovation_norms=imm_inno,
            imm_mode_weights=imm_weights,
            auth_fraction=auth_fraction,
            n_auth=n_auth,
            n_total=n_total,
            osnma_alert=osnma_alert,
            fault_posterior=fault_posterior,
            diagnosis=diagnosis,
            confidence=confidence,
            entropy_alert=entropy_alert,
            structure_alert=structure_alert,
            phase_alert=phase_alert,
            ins_weight=ins_weight,
            n_excluded=n_excluded,
            n_active=n_active_arr,
            mc_auc=mc_auc,
        )
