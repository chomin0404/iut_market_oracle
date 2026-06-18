"""GNSS spoofing detection core — shared types, constants, and re-exports (T1500).

Canonical location for:
    · simulation constants  (NUM_SVIDS, KEY_SIZE_BITS, …)
    · shared data structures (NavMessage, VerificationResult, SimReport)

Implementation classes live in:
    gnss.osnma_simulation  — TESLAKeyChain, OSNMAAuthority, OSNMATransmitter,
                             OSNMAReceiver, SpoofingAttacker, make_eph
    gnss.sim_runner        — run_simulation, verify_tesla_key

All names are re-exported here so existing ``from gnss.core import ...`` call
sites continue to work without modification.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_SVIDS: int = 4
KEY_SIZE_BITS: int = 128  # TESLA key size [bits]
MAC_SIZE_BITS: int = 40  # MAC tag size [bits]
DISCLOSURE_DELAY: int = 2  # key disclosure delay [subframes] — simulation parameter only.
# NOTE: The Galileo OSNMA SIS ICD v1.1 §5.3 specifies TESLA_DELAY = 1 (K_{i-1} disclosed in
#       subframe i).  This value is deliberately set to 2 as a conservative simulation margin.
#       ICD-compliant code uses osnma_inav.TESLA_DELAY = 1.
SUBFRAME_DURATION: int = 30  # subframe length [seconds]
EPH_SIZE: int = 32  # dummy ephemeris size [bytes]
DEFAULT_SEED: int = 42


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class NavMessage:
    """Simplified Galileo I/NAV message (one subframe = 30 s)."""

    svid: int
    epoch: int
    gst: int  # Galileo System Time [s]
    eph_data: bytes  # ephemeris (EPH_SIZE bytes)
    tesla_key: bytes | None  # disclosed TESLA key K_{epoch-delay}
    mac_tag: bytes = field(default_factory=lambda: bytes(MAC_SIZE_BITS // 8))
    is_spoofed: bool = False

    def auth_payload(self) -> bytes:
        """MAC input: SVID(1) || GST(4) || EPH_DATA."""
        return struct.pack("B", self.svid) + struct.pack(">I", self.gst) + self.eph_data


@dataclass
class VerificationResult:
    """Per-message OSNMA verification outcome."""

    epoch: int  # epoch of the buffered message being verified
    disclosure_epoch: int  # epoch at which the key was disclosed
    svid: int
    key_valid: bool  # TESLA key lies on authenticated chain
    mac_valid: bool  # MAC tag matches recomputed value
    receipt_safe: bool  # message received before key was disclosed
    is_spoofed: bool  # ground-truth label
    detected: bool  # any check failed (TESLA or quantum fidelity)
    quantum_anomaly: bool = False  # quantum fidelity below threshold (eph mismatch)


@dataclass
class SimReport:
    """Aggregated detection metrics from run_simulation()."""

    total: int
    spoofed: int
    normal: int
    tp: int
    fp: int
    fn: int
    tn: int
    p_fa: float
    p_md: float
    precision: float
    recall: float
    f1: float
    by_attack_type: dict[str, dict[str, int | float]]
    quantum_detections: int = 0  # key_compromise attacks caught only by quantum fidelity layer


# ---------------------------------------------------------------------------
# Re-exports — implementation classes and functions live in sub-modules
# ---------------------------------------------------------------------------

from gnss.osnma_simulation import (  # noqa: E402
    OSNMAAuthority,
    OSNMAReceiver,
    OSNMATransmitter,
    SpoofingAttacker,
    TESLAKeyChain,
    _AuthorityProtocol,
    make_eph,
)
from gnss.sim_runner import (  # noqa: E402
    _dedup,
    _emit_rows,
    _metrics,
    run_simulation,
    verify_tesla_key,
)

__all__ = [
    # Constants
    "NUM_SVIDS",
    "KEY_SIZE_BITS",
    "MAC_SIZE_BITS",
    "DISCLOSURE_DELAY",
    "SUBFRAME_DURATION",
    "EPH_SIZE",
    "DEFAULT_SEED",
    # Data structures
    "NavMessage",
    "VerificationResult",
    "SimReport",
    # OSNMA simulation
    "TESLAKeyChain",
    "OSNMAAuthority",
    "OSNMATransmitter",
    "_AuthorityProtocol",
    "OSNMAReceiver",
    "SpoofingAttacker",
    "make_eph",
    # Simulation runner
    "run_simulation",
    "verify_tesla_key",
    "_dedup",
    "_metrics",
    "_emit_rows",
]
