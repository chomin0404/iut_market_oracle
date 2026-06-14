"""Shared data structures for GNSS OSNMA and related modules.

Provides hash/MAC function enumerations, ADKD / ECDSAType codes, and the
DSMKROOTMessage / DSMPKRMessage dataclasses used by the TESLA chain and
ECDSA verification engines.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class HashFunction(Enum):
    """Hash function selector — HF field in DSM-KROOT (ICD §5.3 Table).

    Values:
        SHA_256  (0) — SHA-256 (default, ICD-compliant)
        SHA3_256 (1) — SHA3-256
    """

    SHA_256 = 0
    SHA3_256 = 1


class MACFunction(Enum):
    """MAC function selector — MF field in DSM-KROOT (ICD §5.3 Table).

    Values:
        HMAC_SHA_256 (0) — HMAC-SHA-256 (default, ICD-compliant)
        CMAC_AES     (1) — CMAC-AES-128
    """

    HMAC_SHA_256 = 0
    CMAC_AES = 1


class ADKD(Enum):
    """Authentication Data and Key Distribution type codes (ICD §5.4.1 Table 11).

    Identifies which navigation data is authenticated by a given MACK tag,
    and the disclosure timing (TESLA delay slots) for that data type.

    Values:
        INAV_CED    (0)  — I/NAV Clock & Ephemeris Data (default)
        INAV_TIMING (4)  — I/NAV Timing Parameters (GST-UTC, GST-GPS)
        SLOW_MAC    (12) — Slow MAC / Galileo cross-authentication
    """

    INAV_CED = 0
    INAV_TIMING = 4
    SLOW_MAC = 12


class ECDSAType(Enum):
    """ECDSA curve/hash selector — NPKT field in DSM-PKR (ICD §5.5 Table).

    Values:
        P256 (0) — P-256 + SHA-256 (default, ICD-compliant)
        P521 (1) — P-521 + SHA-512
    """

    P256 = 0
    P521 = 1


@dataclass(frozen=True)
class DSMKROOTMessage:
    """Decoded DSM-KROOT message (Galileo OSNMA SIS ICD §5.3).

    Carries chain identity, cryptographic parameters, and the root TESLA key K_ROOT.
    Field names follow ICD notation where possible.

    Attributes:
        cidkr:          TESLA chain ID from K_ROOT header (2-bit CHAIN_ID field).
        hash_func:      Hash function used in TESLA key derivation (HF field).
        mac_func:       MAC function used for authentication tags (MF field).
        key_size_bytes: TESLA key size in bytes (derived from KS selector field).
        tag_size_bits:  MAC tag truncation size in bits (derived from TS selector field).
        gst0:           GST [s] at subframe index 0 — chain epoch anchor.
        alpha:          6-byte nonce (ALPHA field).  Prevents cross-chain tag reuse
                        by binding key derivation to this chain's identity.
        kroot:          K_ROOT bytes (``key_size_bytes`` long).  Chain anchor key;
                        index 0 in the TESLAChain key-index scheme.
        ds:             Digital signature over DSM-KROOT fields — raw (r || s) format.
                        64 bytes for ECDSA-P256, 132 bytes for ECDSA-P521.
        m_kroot_body:   Pre-serialised signed body (everything between the NMA header
                        byte and the DS field, per ICD §5.4.4).  Empty bytes for
                        simulation instances that do not exercise ECDSA verification.
    """

    cidkr: int
    hash_func: HashFunction
    mac_func: MACFunction
    key_size_bytes: int
    tag_size_bits: int
    gst0: int
    alpha: bytes  # 6 bytes
    kroot: bytes  # key_size_bytes long
    ds: bytes
    m_kroot_body: bytes = b""  # optional — required only for ECDSA DS verification

    def build_m_kroot(self, nma_hdr_byte: int) -> bytes:
        """Assemble M_KROOT for ECDSA signature verification (ICD §5.4.4).

        M_KROOT = NMA_Header(1B) || DSM-KROOT body (all fields before DS).

        Args:
            nma_hdr_byte: NMA status / header byte (bits 7-0 of NMA header word).

        Returns:
            bytes to be passed to ECDSA verify.
        """
        return bytes([nma_hdr_byte & 0xFF]) + self.m_kroot_body


@dataclass(frozen=True)
class DSMPKRMessage:
    """Decoded DSM-PKR message (Galileo OSNMA SIS ICD §5.5).

    Carries a public key and its Merkle Tree proof path for verification
    against a trusted Merkle root.

    Attributes:
        pkid:         Public key ID (leaf index in Merkle tree, 0–15).
        pktype:       ECDSA curve/hash selector (NPKT field; ECDSAType enum).
        public_key:   Uncompressed or compressed EC public key bytes.
        merkle_nodes: Ordered sibling nodes from leaf to root (exclusive).
                      Length equals ⌈log2(tree_leaves)⌉.
    """

    pkid: int
    pktype: ECDSAType
    public_key: bytes
    merkle_nodes: tuple[bytes, ...]
