"""Syndrome Graph — append-only consistency-check graph for GNSS fault discrimination.

Implements the Syndrome Layer of the 5-layer GNSS defense architecture.
Parity-like consistency checks generate edges; only violated constraints
are recorded, forming a sparse adjacency structure over (satellite, epoch) nodes.

Constraint types
----------------
CROSS_SAT_DOPPLER   coherent Doppler shift: n·mean(Δf)² / var(Δf) > threshold
                    Detects common-mode meaconing (all sats biased by same spoofer).
GEOMETRY_RAIM       RAIM chi² > chi²_alpha(n−4): constellation geometry integrity failure.
CN0_COHERENCE       std(C/N₀) / mean(C/N₀) < threshold: all sats equally strong
                    (single transmitter spoofing signature).
TEMPORAL_PHASE      |φ(t) − φ(t−1)| > threshold: carrier-phase integer-cycle jump.
AUTH_MISMATCH       physical=SPOOFED AND crypto=AUTHENTICATED: cryptographic incoherence.
IONO_RESIDUAL       ionosphere-free combination outlier (reserved; not yet triggered).

Graph structure
---------------
    Node  v ∈ V : SyndromeNode = (satellite_id, epoch)
    Edge  e ∈ E : SyndromeEdge = violated constraint between two nodes

Special satellite_id values used for multi-satellite constraints:
    "ALL"           — constraint spans all satellites (e.g. coherent Doppler, CN0)
    "CONSTELLATION" — geometry-level check (RAIM chi²)

Invariants (append-only)
------------------------
    * Nodes and edges are never deleted or modified after creation.
    * add_epoch() raises ValueError if called twice with the same epoch.
    * Each SyndromeEdge carries a SHA-256 digest of its canonical representation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple

import numpy as np

from gnss.correspondence_layer import CorrespondenceState

# ---------------------------------------------------------------------------
# Constraint types and graph primitives
# ---------------------------------------------------------------------------


class ConstraintType(str, Enum):
    """Parity-like consistency constraint kinds."""

    CROSS_SAT_DOPPLER = "cross_sat_doppler"
    GEOMETRY_RAIM = "geometry_raim"
    CN0_COHERENCE = "cn0_coherence"
    TEMPORAL_PHASE = "temporal_phase"
    AUTH_MISMATCH = "auth_mismatch"
    IONO_RESIDUAL = "iono_residual"


class SyndromeNode(NamedTuple):
    """Vertex in the syndrome graph: one (satellite, epoch) observation point."""

    satellite_id: str
    epoch: int


@dataclass(frozen=True)
class SyndromeEdge:
    """A violated consistency constraint — one directed edge in the syndrome graph.

    Attributes:
        node_a, node_b:   Connected (satellite_id, epoch) nodes.
                          For global constraints, satellite_id is "ALL" or "CONSTELLATION".
        constraint_type:  Which parity check was violated.
        value:            Observed test statistic at violation time.
        threshold:        Decision boundary (constraint fires when value > threshold,
                          except CN0_COHERENCE which fires when value < threshold).
        epoch:            Epoch at which the constraint was evaluated.
        digest:           SHA-256 of canonical JSON representation (tamper-evident).
    """

    node_a: SyndromeNode
    node_b: SyndromeNode
    constraint_type: ConstraintType
    value: float
    threshold: float
    epoch: int
    digest: str

    def is_violated(self) -> bool:
        """True when this constraint is violated.

        CN0_COHERENCE fires when spread is TOO LOW (value < threshold).
        All other constraints fire when test statistic is TOO HIGH (value > threshold).
        """
        if self.constraint_type == ConstraintType.CN0_COHERENCE:
            return self.value < self.threshold
        return self.value > self.threshold


# ---------------------------------------------------------------------------
# Constraint thresholds
# ---------------------------------------------------------------------------

# Cross-satellite Doppler coherence SNR threshold.
# SNR = n · mean(Δf)² / var(Δf).  Threshold=5 corresponds to mean|Δf| > √(5·var/n).
_CROSS_SAT_SNR_THRESHOLD: float = 5.0

# CN0 spread collapse: std(C/N₀) / mean(C/N₀) < threshold indicates single transmitter.
# Typical nominal spread ≈ 0.08–0.12; spoofing spread ≈ 0.01–0.03.
_CN0_SPREAD_MIN: float = 0.05

# Carrier-phase jump threshold [cycles].  Half-cycle jumps are measurable; use 0.4.
_PHASE_JUMP_CYCLES: float = 0.4

# Sentinel satellite IDs for multi-satellite constraints.
_SAT_ALL: str = "ALL"
_SAT_CONSTELLATION: str = "CONSTELLATION"


# ---------------------------------------------------------------------------
# Digest helper
# ---------------------------------------------------------------------------


def _compute_digest(
    node_a: SyndromeNode,
    node_b: SyndromeNode,
    constraint_type: ConstraintType,
    value: float,
    threshold: float,
    epoch: int,
) -> str:
    """SHA-256 of the canonical JSON representation of one syndrome edge."""
    payload = json.dumps(
        {
            "node_a": list(node_a),
            "node_b": list(node_b),
            "constraint_type": constraint_type.value,
            "value": round(value, 8),
            "threshold": round(threshold, 8),
            "epoch": epoch,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _make_edge(
    node_a: SyndromeNode,
    node_b: SyndromeNode,
    constraint_type: ConstraintType,
    value: float,
    threshold: float,
    epoch: int,
) -> SyndromeEdge:
    digest = _compute_digest(node_a, node_b, constraint_type, value, threshold, epoch)
    return SyndromeEdge(
        node_a=node_a,
        node_b=node_b,
        constraint_type=constraint_type,
        value=value,
        threshold=threshold,
        epoch=epoch,
        digest=digest,
    )


# ---------------------------------------------------------------------------
# Syndrome Graph
# ---------------------------------------------------------------------------


class SyndromeGraph:
    """Append-only syndrome graph over (satellite, epoch) observation nodes.

    Usage::

        graph = SyndromeGraph()
        new_edges = graph.add_epoch(
            epoch=0,
            satellite_ids=["G01", "G02", ...],
            correspondences=...,
            doppler_deviations=np.zeros(n),
            raim_chi2=0.0,
            raim_threshold=16.92,
        )
    """

    def __init__(self) -> None:
        # Flat list of all edges, ordered by insertion time.
        self._edges: list[SyndromeEdge] = []
        # epoch → list of indices into self._edges (for O(1) epoch lookup).
        self._epoch_index: dict[int, list[int]] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def add_epoch(
        self,
        epoch: int,
        satellite_ids: list[str],
        correspondences: list[CorrespondenceState],
        doppler_deviations: np.ndarray,
        raim_chi2: float,
        raim_threshold: float,
        cn0_values: np.ndarray | None = None,
        carrier_phases: np.ndarray | None = None,
        prev_carrier_phases: np.ndarray | None = None,
        phase_jump_threshold: float = _PHASE_JUMP_CYCLES,
    ) -> list[SyndromeEdge]:
        """Evaluate all constraints for one epoch and append violated ones.

        Returns the list of new SyndromeEdge objects added (may be empty).
        The edges are also stored internally for later retrieval.

        Args:
            epoch:                Epoch index (must be strictly new; raises on duplicate).
            satellite_ids:        Ordered list of n satellite identifiers.
            correspondences:      Per-satellite CorrespondenceState from CorrespondenceLayer.
            doppler_deviations:   (n,) Doppler residuals [Hz].
            raim_chi2:            RAIM chi² test statistic.
            raim_threshold:       RAIM chi² decision boundary.
            cn0_values:           (n,) C/N₀ values [dB-Hz]; skip check if None.
            carrier_phases:       (n,) carrier phase [cycles]; skip temporal check if None.
            prev_carrier_phases:  (n,) previous-epoch carrier phases; skip if None.
            phase_jump_threshold: Threshold for carrier-phase jump [cycles].

        Raises:
            ValueError: If epoch has already been processed (append-only invariant).
        """
        if epoch in self._epoch_index:
            raise ValueError(
                f"Epoch {epoch} already processed (append-only invariant). "
                "SyndromeGraph does not allow retroactive modification."
            )

        new_edges: list[SyndromeEdge] = []
        n = len(satellite_ids)

        # ---- 1. Cross-satellite Doppler coherence ----------------------------
        # coherent_snr = n · mean² / var; large SNR → common-mode meaconing.
        mean_dop = float(doppler_deviations.mean())
        var_dop = max(float(doppler_deviations.var()), 1e-9)
        coherent_snr = n * mean_dop**2 / var_dop
        if coherent_snr > _CROSS_SAT_SNR_THRESHOLD:
            node = SyndromeNode(_SAT_ALL, epoch)
            new_edges.append(
                _make_edge(
                    node,
                    node,
                    ConstraintType.CROSS_SAT_DOPPLER,
                    coherent_snr,
                    _CROSS_SAT_SNR_THRESHOLD,
                    epoch,
                )
            )

        # ---- 2. RAIM geometry check -----------------------------------------
        if raim_chi2 > raim_threshold:
            node = SyndromeNode(_SAT_CONSTELLATION, epoch)
            new_edges.append(
                _make_edge(
                    node, node, ConstraintType.GEOMETRY_RAIM, raim_chi2, raim_threshold, epoch
                )
            )

        # ---- 3. C/N₀ coherence (spread collapse) ----------------------------
        if cn0_values is not None and len(cn0_values) >= 2:
            cn0_mean = max(float(cn0_values.mean()), 1e-9)
            cn0_std = float(cn0_values.std())
            spread_ratio = cn0_std / cn0_mean
            if spread_ratio < _CN0_SPREAD_MIN:
                node = SyndromeNode(_SAT_ALL, epoch)
                new_edges.append(
                    _make_edge(
                        node,
                        node,
                        ConstraintType.CN0_COHERENCE,
                        spread_ratio,
                        _CN0_SPREAD_MIN,
                        epoch,
                    )
                )

        # ---- 4. Temporal carrier-phase jump (per satellite) -----------------
        if (
            carrier_phases is not None
            and prev_carrier_phases is not None
            and len(carrier_phases) == n
            and len(prev_carrier_phases) == n
        ):
            for i, sat_id in enumerate(satellite_ids):
                jump = abs(float(carrier_phases[i] - prev_carrier_phases[i]))
                if jump > phase_jump_threshold:
                    node_prev = SyndromeNode(sat_id, epoch - 1)
                    node_curr = SyndromeNode(sat_id, epoch)
                    new_edges.append(
                        _make_edge(
                            node_prev,
                            node_curr,
                            ConstraintType.TEMPORAL_PHASE,
                            jump,
                            phase_jump_threshold,
                            epoch,
                        )
                    )

        # ---- 5. Auth mismatch (physical ≠ crypto) ---------------------------
        for state in correspondences:
            if not state.is_coherent():
                node = SyndromeNode(state.satellite_id, epoch)
                new_edges.append(
                    _make_edge(node, node, ConstraintType.AUTH_MISMATCH, 1.0, 0.5, epoch)
                )

        # Atomically append and index.
        start_idx = len(self._edges)
        self._edges.extend(new_edges)
        self._epoch_index[epoch] = list(range(start_idx, start_idx + len(new_edges)))

        return new_edges

    def edges_at_epoch(self, epoch: int) -> list[SyndromeEdge]:
        """Return all edges added at the given epoch (empty list if epoch not processed)."""
        return [self._edges[i] for i in self._epoch_index.get(epoch, [])]

    def all_edges(self) -> list[SyndromeEdge]:
        """Return a shallow copy of all edges (order: insertion time)."""
        return list(self._edges)

    def violated_constraints_at(self, epoch: int) -> list[ConstraintType]:
        """Return constraint types that fired at the given epoch."""
        return [e.constraint_type for e in self.edges_at_epoch(epoch)]

    def syndrome_score(self, epoch: int) -> float:
        """Fraction of constraint types violated at this epoch, normalized to [0, 1]."""
        n_violated = len(self.edges_at_epoch(epoch))
        n_types = len(ConstraintType)
        return min(n_violated / n_types, 1.0)

    def node_fault_count(self, satellite_id: str, last_n_epochs: int = 5) -> int:
        """Number of constraint violations involving satellite_id in the last n epochs.

        Args:
            satellite_id:   Satellite PRN or sentinel ("ALL", "CONSTELLATION").
            last_n_epochs:  Sliding window size in epochs.
        """
        if not self._epoch_index:
            return 0
        last_epoch = max(self._epoch_index)
        first_epoch = max(0, last_epoch - last_n_epochs + 1)
        count = 0
        for ep in range(first_epoch, last_epoch + 1):
            for edge in self.edges_at_epoch(ep):
                if (
                    edge.node_a.satellite_id == satellite_id
                    or edge.node_b.satellite_id == satellite_id
                ):
                    count += 1
        return count

    def total_edges(self) -> int:
        """Total number of syndrome edges in the graph."""
        return len(self._edges)
