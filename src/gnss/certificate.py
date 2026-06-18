"""Certificate Layer — machine-readable invariant proof for GNSS alarm decisions.

Each AlarmCertificate records:
    decision:              MAP fault class from the Decoder Layer.
    invariants_satisfied:  Names of checks that passed (positive evidence).
    checks_failed:         Names of violated constraints (basis for the alarm).
    evidence_digest:       SHA-256 of canonical (epoch, run_id, decision, checks_failed).
    run_id, epoch, gps_tow: Traceability fields.

Invariant
---------
    For any non-NOMINAL decision, checks_failed MUST be non-empty.
    A certificate with decision=SPOOFING and checks_failed=[] is structurally
    invalid — AlarmCertificate.is_valid() will return False.

This mirrors the verifiable-computation principle: the certificate is
self-contained evidence that can be verified without re-running the pipeline.

Usage::

    certificate = build_certificate(
        epoch=0,
        gps_tow=518400.0,
        run_id="a1b2c3d4",
        decoder_result=decoder_result,
        syndrome_edges=syndrome_edges,
    )
    assert certificate.is_valid()
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from gnss.decoder import DecisionClass, DecoderResult
from gnss.syndrome_graph import SyndromeEdge

# ---------------------------------------------------------------------------
# All invariant names that the pipeline evaluates per epoch.
# Used to compute the invariants_satisfied complement set.
# ---------------------------------------------------------------------------

_ALL_INVARIANTS: tuple[str, ...] = (
    "cross_sat_doppler",
    "temporal_phase",
    "geometry_raim",
    "cn0_coherence",
    "auth_mismatch",
    "iono_residual",
    "bp_convergence",
    "correspondence_coherence",
)


# ---------------------------------------------------------------------------
# Certificate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AlarmCertificate:
    """Machine-readable proof of which invariants were checked for one epoch decision.

    Immutable: all fields are set at construction and cannot be modified.

    Attributes:
        epoch:                 Epoch index.
        gps_tow:               GPS Time of Week [s].
        run_id:                Unique run identifier (8-char hex).
        decision:              MAP fault class.
        invariants_satisfied:  Constraint names that passed.
        checks_failed:         Constraint names that were violated.
        n_anomalous_sats:      Number of satellites flagged as anomalous by the decoder.
        evidence_digest:       SHA-256 of the canonical certificate representation.
    """

    epoch: int
    gps_tow: float
    run_id: str
    decision: DecisionClass
    invariants_satisfied: tuple[str, ...]
    checks_failed: tuple[str, ...]
    n_anomalous_sats: int
    evidence_digest: str

    def is_valid(self) -> bool:
        """True when the certificate satisfies structural invariants.

        Non-NOMINAL decisions must have at least one failed check.
        NOMINAL decisions may have empty checks_failed (no violations detected).
        """
        if self.decision != DecisionClass.NOMINAL:
            return len(self.checks_failed) > 0
        return True


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_certificate(
    epoch: int,
    gps_tow: float,
    run_id: str,
    decoder_result: DecoderResult,
    syndrome_edges: list[SyndromeEdge],
    correspondence_incoherence_count: int = 0,
) -> AlarmCertificate:
    """Build an AlarmCertificate from decoder and syndrome outputs.

    The checks_failed list is derived from:
        1. Violated syndrome edges (constraint_type.value for each violated edge).
        2. "bp_convergence" if decoder BP did not converge (always False for log-linear).
        3. "correspondence_coherence" if any satellite had an incoherent dual state.

    invariants_satisfied = _ALL_INVARIANTS − checks_failed.
    evidence_digest = SHA-256(epoch, run_id, decision, sorted(checks_failed)).

    Args:
        epoch:                           Epoch index.
        gps_tow:                         GPS Time of Week [s].
        run_id:                          Run identifier.
        decoder_result:                  DecoderResult from FactorGraphDecoder.
        syndrome_edges:                  Edges from SyndromeGraph.edges_at_epoch().
        correspondence_incoherence_count: Number of incoherent (physical, crypto) pairs.
    """
    # Collect violated constraint names.
    raw_failed: list[str] = [e.constraint_type.value for e in syndrome_edges if e.is_violated()]

    # Additional checks.
    if not decoder_result.bp_converged:
        raw_failed.append("bp_convergence")
    if correspondence_incoherence_count > 0:
        raw_failed.append("correspondence_coherence")

    # Deduplicate preserving insertion order.
    seen: set[str] = set()
    checks_failed: list[str] = []
    for name in raw_failed:
        if name not in seen:
            seen.add(name)
            checks_failed.append(name)

    invariants_satisfied = tuple(inv for inv in _ALL_INVARIANTS if inv not in seen)

    # Compute evidence digest over the canonical certificate representation.
    canonical = json.dumps(
        {
            "epoch": epoch,
            "run_id": run_id,
            "decision": decoder_result.decision.value,
            "checks_failed": sorted(checks_failed),
            "n_anomalous": decoder_result.n_anomalous,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    evidence_digest = hashlib.sha256(canonical.encode()).hexdigest()

    return AlarmCertificate(
        epoch=epoch,
        gps_tow=gps_tow,
        run_id=run_id,
        decision=decoder_result.decision,
        invariants_satisfied=invariants_satisfied,
        checks_failed=tuple(checks_failed),
        n_anomalous_sats=decoder_result.n_anomalous,
        evidence_digest=evidence_digest,
    )
