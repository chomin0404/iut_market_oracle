"""Decoder Layer — belief propagation on a bipartite factor graph.

Mathematical formulation
------------------------
Variable nodes:  v_i ∈ {0: genuine, 1: anomalous}, one per satellite.
Factor nodes:    f_e, one per violated syndrome edge.

Factor potential:
    ψ_e(x_{N(e)}) = exp(β · Σ_{i ∈ N(e)} x_i)
    Higher energy when more satellites in the factor are anomalous.

Sum-product message (factor → variable, for this specific potential):
    For ψ = exp(β · Σ x_i), the factor-to-variable message is independent
    of all other variable messages. The log-odds ratio reduces to:

        log-odds(f_e → v_i, x=1) − log-odds(f_e → v_i, x=0) = β

    This means BP converges in one pass for this class of potentials,
    and the exact marginal posterior is given by the log-linear model:

        log_odds(v_i = anomalous) = log_prior + β · Σ_{e: i ∈ N(e)} strength(e)

    where:
        strength(e) = clip((value_e − threshold_e) / max(threshold_e, ε), 0, 5)
        β           = BP_BETA (factor potential strength)
        log_prior   = log(P_prior / (1 − P_prior))

Reference: Kschischang, Frey, Loeliger (2001) "Factor graphs and the
sum-product algorithm", IEEE Trans. Inf. Theory 47(2):498–519.

4-class global decision
-----------------------
    n_anomalous == 0:
        GEOMETRY_RAIM in active constraints (no per-sat anomaly) → HARDWARE_FAULT
        otherwise                                                  → NOMINAL
    n_anomalous == 1:
        HARDWARE_FAULT
    n_anomalous >= 2, elevation-correlated anomaly beliefs:
        MULTIPATH
    n_anomalous >= 2, CROSS_SAT_DOPPLER in active:
        SPOOFING
    n_anomalous >= 2, otherwise:
        SPOOFING  (default for multi-satellite coordinated anomaly)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gnss.syndrome_graph import ConstraintType, SyndromeEdge
from schemas.gnss import FaultClass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Log-linear model parameters.
# log_prior = log(P_anomalous / (1 - P_anomalous)) for prior P_anomalous.
_P_PRIOR_ANOMALOUS: float = 0.10  # 10 % prior probability of anomaly
_LOG_PRIOR_ANOMALOUS: float = np.log(_P_PRIOR_ANOMALOUS / (1.0 - _P_PRIOR_ANOMALOUS))

# Factor potential strength β.  Each violated constraint adds β·strength to log-odds.
_BP_BETA: float = 2.0

# Maximum violation strength (caps extreme ratios to prevent numerical saturation).
_VIOLATION_STRENGTH_MAX: float = 5.0

# Pearson |corr(anomaly_belief, 1/elevation)| threshold for multipath classification.
_ELEVATION_CORR_THRESHOLD: float = 0.60

# Sentinel satellite IDs for global (multi-satellite) constraints.
_GLOBAL_SATS: frozenset[str] = frozenset({"ALL", "CONSTELLATION"})


# ---------------------------------------------------------------------------
# Decision class
# ---------------------------------------------------------------------------


# Re-export the canonical fault taxonomy from the schemas layer.
# gnss.decoder imports are unchanged for downstream code; the single source
# of truth is schemas.gnss.FaultClass.
DecisionClass = FaultClass


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DecoderResult:
    """MAP decoding result from the factor graph.

    Attributes:
        satellite_ids:       Ordered tuple of n satellite identifiers.
        anomaly_beliefs:     P(anomalous | syndrome) per satellite ∈ [0, 1].
        decision:            Global 4-class MAP decision.
        decision_score:      Evidence strength for the decision ∈ [0, 1].
        n_anomalous:         Number of satellites with P(anomalous) > 0.5.
        bp_converged:        Always True for the log-linear model (1-pass exact).
        active_constraints:  Set of constraint types present in violated edges.
    """

    satellite_ids: tuple[str, ...]
    anomaly_beliefs: tuple[float, ...]
    decision: DecisionClass
    decision_score: float
    n_anomalous: int
    bp_converged: bool
    active_constraints: frozenset[ConstraintType]


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------


class FactorGraphDecoder:
    """Log-linear belief propagation decoder.

    Converts a list of violated SyndromeEdge objects into per-satellite
    anomaly beliefs and a global 4-class fault decision.
    """

    def decode(
        self,
        satellite_ids: list[str],
        syndrome_edges: list[SyndromeEdge],
        elevations_rad: np.ndarray | None = None,
    ) -> DecoderResult:
        """Decode satellite fault states from violated syndrome edges.

        Args:
            satellite_ids:   Ordered list of n satellite identifiers.
            syndrome_edges:  Edges from SyndromeGraph.edges_at_epoch() for this epoch.
            elevations_rad:  (n,) elevation angles [rad]; used for MULTIPATH vs SPOOFING
                             discrimination.  None disables elevation-correlation check.

        Returns:
            DecoderResult with per-satellite beliefs and global decision.
        """
        n = len(satellite_ids)
        sat_index: dict[str, int] = {sid: i for i, sid in enumerate(satellite_ids)}
        violated = [e for e in syndrome_edges if e.is_violated()]

        # No violated constraints → NOMINAL with very low anomaly beliefs.
        if not violated:
            return DecoderResult(
                satellite_ids=tuple(satellite_ids),
                anomaly_beliefs=tuple(_sigmoid(_LOG_PRIOR_ANOMALOUS) for _ in range(n)),
                decision=DecisionClass.NOMINAL,
                decision_score=1.0 - _sigmoid(_LOG_PRIOR_ANOMALOUS),
                n_anomalous=0,
                bp_converged=True,
                active_constraints=frozenset(),
            )

        # Build factor-to-variable adjacency.
        # factor_vars[f] = list of satellite indices connected to factor f.
        factor_vars: list[list[int]] = []
        for edge in violated:
            if edge.node_a.satellite_id in _GLOBAL_SATS:
                # Global constraint: connects to all n satellites.
                factor_vars.append(list(range(n)))
            else:
                involved: list[int] = []
                for node in (edge.node_a, edge.node_b):
                    idx = sat_index.get(node.satellite_id)
                    if idx is not None and idx not in involved:
                        involved.append(idx)
                factor_vars.append(involved)

        # Log-linear model: one-pass exact sum-product.
        # log_odds(v_i = anomalous) = log_prior + Σ_e β · strength(e) · I[i ∈ N(e)]
        log_odds = np.full(n, _LOG_PRIOR_ANOMALOUS, dtype=float)
        for edge, fvars in zip(violated, factor_vars):
            strength = _violation_strength(edge)
            contribution = _BP_BETA * strength
            for v in fvars:
                log_odds[v] += contribution

        anomaly_beliefs = _sigmoid(log_odds)

        # MAP decision.
        is_anomalous = anomaly_beliefs > 0.5
        n_anomalous = int(is_anomalous.sum())
        active = frozenset(e.constraint_type for e in violated)

        decision = self._classify(
            n_anomalous=n_anomalous,
            is_anomalous=is_anomalous,
            anomaly_beliefs=anomaly_beliefs,
            elevations_rad=elevations_rad,
            active=active,
        )

        # Decision score: max anomaly belief if any anomaly; confidence in nominal otherwise.
        if n_anomalous > 0:
            decision_score = float(np.max(anomaly_beliefs))
        else:
            decision_score = float(1.0 - np.max(anomaly_beliefs))

        return DecoderResult(
            satellite_ids=tuple(satellite_ids),
            anomaly_beliefs=tuple(float(b) for b in anomaly_beliefs),
            decision=decision,
            decision_score=min(decision_score, 1.0),
            n_anomalous=n_anomalous,
            bp_converged=True,  # log-linear model is exact in 1 pass
            active_constraints=active,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _classify(
        self,
        n_anomalous: int,
        is_anomalous: np.ndarray,
        anomaly_beliefs: np.ndarray,
        elevations_rad: np.ndarray | None,
        active: frozenset[ConstraintType],
    ) -> DecisionClass:
        """Map decoder outputs to a 4-class fault decision."""
        # RAIM failure without coherent Doppler shift → hardware fault.
        # Checked first because the global RAIM factor pushes all satellites
        # to anomalous (n_anomalous == n) even for a single-satellite fault,
        # so the n_anomalous == 0 branch would never fire.
        if (
            ConstraintType.GEOMETRY_RAIM in active
            and ConstraintType.CROSS_SAT_DOPPLER not in active
        ):
            return DecisionClass.HARDWARE_FAULT

        if n_anomalous == 0:
            return DecisionClass.NOMINAL

        if n_anomalous == 1:
            return DecisionClass.HARDWARE_FAULT

        # n_anomalous >= 2 — distinguish multipath from spoofing.

        # Elevation correlation test: multipath preferentially affects low-elevation sats.
        # corr(anomaly_belief, −elevation) > threshold → elevation-driven → MULTIPATH.
        if elevations_rad is not None and len(elevations_rad) == len(anomaly_beliefs):
            fault_vec = anomaly_beliefs.astype(float)
            el_vec = elevations_rad.astype(float)
            if np.std(fault_vec) > 1e-6 and np.std(el_vec) > 1e-6:
                corr = float(np.corrcoef(fault_vec, -el_vec)[0, 1])
                if abs(corr) > _ELEVATION_CORR_THRESHOLD:
                    return DecisionClass.MULTIPATH

        # Coherent common-mode Doppler shift is the primary spoofing signature.
        if ConstraintType.CROSS_SAT_DOPPLER in active:
            return DecisionClass.SPOOFING

        # Multiple anomalous sats without elevation explanation → spoofing.
        return DecisionClass.SPOOFING


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    """Element-wise sigmoid: σ(x) = 1 / (1 + exp(−x))."""
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def _violation_strength(edge: SyndromeEdge) -> float:
    """Normalised violation magnitude, clipped to [0, MAX].

    For constraints that fire when value > threshold (most cases):
        strength = (value − threshold) / threshold

    For CN0_COHERENCE (fires when value < threshold):
        strength = (threshold − value) / threshold
    """
    t = max(edge.threshold, 1e-9)
    if edge.constraint_type == ConstraintType.CN0_COHERENCE:
        raw = (edge.threshold - edge.value) / t
    else:
        raw = (edge.value - edge.threshold) / t
    return float(min(max(raw, 0.0), _VIOLATION_STRENGTH_MAX))
