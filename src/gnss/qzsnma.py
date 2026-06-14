"""QZSS Navigation Message Authentication (QZSNMA) — L1 complement to OSNMA.

QZSS NMA authenticates QZSS L1C/A and L1S navigation messages using a
TESLA-based key chain with ECDSA root-of-trust, mirroring the Galileo
OSNMA architecture described in IS-QZSS-NMA-001.

Architecture
-----------
QZSNMAChain   — models one TESLA key chain for a QZSS satellite.
QZSNMAVerifier— validates the chain and per-satellite MAC tags.
QZSNMALayer   — epoch-level coverage monitor; output mirrors OSNMALayerResult.

Key chain parameters (IS-QZSS-NMA-001 §5)
------------------------------------------
SUBFRAME_DURATION_S  :  1.0 s   (QZSS L1C sub-frame epoch)
KEY_SIZE_BYTES       :  16      (128-bit TESLA keys)
HASH_TRUNCATION_BITS :  40      (NMA tag truncation)
KDF_ROUNDS           :  1       (one SHA-256 round per step)

Note
----
Bit-exact ICD verification against a licensed copy of IS-QZSS-NMA-001 has
**not** been performed.  Treat field offsets and chain parameters as
best-effort until cross-checked with the official CAB document.
"""

from __future__ import annotations

import hashlib
import hmac
import math
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUBFRAME_DURATION_S: float = 1.0  # QZSS L1C sub-frame period [s]
KEY_SIZE_BYTES: int = 16  # 128-bit TESLA key
HASH_TRUNCATION_BITS: int = 40  # NMA MAC tag length [bits]
HASH_TRUNCATION_BYTES: int = math.ceil(HASH_TRUNCATION_BITS / 8)  # 5

_AUTH_FRAC_THRESH: float = 0.50  # alert if < 50 % of QZSS sats authenticated
_MAX_CHAIN_LEN: int = 4096  # safety upper bound on chain length

# ---------------------------------------------------------------------------
# TESLA key derivation (single SHA-256 round, left-truncated)
# ---------------------------------------------------------------------------


def _derive_parent_key(child_key: bytes) -> bytes:
    """Derive parent key: K_{i-1} = left_128(SHA-256(K_i)).

    In a TESLA chain the root key K_0 is broadcast via ECDSA-signed DSM-KROOT.
    Each subsequent key K_i is derived from K_{i-1} by one-way hashing so that
    past keys cannot be forged.

    Args:
        child_key: 16-byte TESLA key for step i.

    Returns:
        16-byte TESLA key for step i-1 (left-truncated SHA-256).
    """
    digest = hashlib.sha256(child_key).digest()
    return digest[:KEY_SIZE_BYTES]


def _compute_mac_tag(key: bytes, message: bytes) -> bytes:
    """Compute HMAC-SHA256 truncated to HASH_TRUNCATION_BYTES.

    Args:
        key:     16-byte TESLA key.
        message: authenticated navigation data bytes.

    Returns:
        Truncated MAC tag (HASH_TRUNCATION_BYTES bytes).
    """
    mac = hmac.new(key, message, hashlib.sha256).digest()
    return mac[:HASH_TRUNCATION_BYTES]


# ---------------------------------------------------------------------------
# TESLA key chain
# ---------------------------------------------------------------------------


@dataclass
class QZSNMAChain:
    """One TESLA key chain for a single QZSS satellite.

    Stores keys in forward order: keys[0] = K_0 (root), keys[i] = K_i.

    Attributes
    ----------
    prn          QZSS PRN number (193–202 for QZSS).
    gst0         GST epoch of the chain root [integer second].
    keys         Forward key sequence (list of 16-byte objects).
    """

    prn: int
    gst0: int
    keys: list[bytes] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_root_and_length(
        cls,
        prn: int,
        gst0: int,
        root_key: bytes,
        chain_length: int,
    ) -> QZSNMAChain:
        """Build a full forward chain by repeatedly hashing root_key.

        Generates keys[0..chain_length-1] where keys[0] = root_key and
        keys[i] = SHA-256(keys[i-1]) (left 16 bytes).

        Args:
            prn:          QZSS PRN number.
            gst0:         Chain epoch origin [s].
            root_key:     16-byte root TESLA key (K_0).
            chain_length: Number of keys to generate (including root).

        Raises:
            ValueError: if root_key length ≠ KEY_SIZE_BYTES or chain_length out of range.
        """
        if len(root_key) != KEY_SIZE_BYTES:
            raise ValueError(f"root_key must be {KEY_SIZE_BYTES} bytes, got {len(root_key)}")
        if not (1 <= chain_length <= _MAX_CHAIN_LEN):
            raise ValueError(f"chain_length must be in [1, {_MAX_CHAIN_LEN}], got {chain_length}")
        keys: list[bytes] = [root_key]
        for _ in range(chain_length - 1):
            keys.append(_derive_parent_key(keys[-1]))
        return cls(prn=prn, gst0=gst0, keys=keys)

    # ------------------------------------------------------------------
    # Key access
    # ------------------------------------------------------------------

    def key_at(self, gst_sf: int) -> bytes | None:
        """Return the TESLA key for sub-frame at GPS time gst_sf.

        Key index: idx = (gst_sf − gst0) / SUBFRAME_DURATION_S
        The key is disclosed one sub-frame after use (TESLA delay = 1).

        Args:
            gst_sf: GPS/QZSS time of the sub-frame [integer second].

        Returns:
            16-byte key, or None if the index is out of range.
        """
        delta = gst_sf - self.gst0
        if delta < 0:
            return None
        idx = int(round(delta / SUBFRAME_DURATION_S))
        if idx >= len(self.keys):
            return None
        return self.keys[idx]

    def __len__(self) -> int:
        return len(self.keys)


# ---------------------------------------------------------------------------
# Per-satellite verification result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QZSNMAVerifyResult:
    """Per-satellite QZSNMA tag verification result.

    Attributes
    ----------
    prn           QZSS PRN number.
    authenticated True when the MAC tag matched the disclosed key.
    reason        Short description of success or failure.
    """

    prn: int
    authenticated: bool
    reason: str


# ---------------------------------------------------------------------------
# Chain verifier
# ---------------------------------------------------------------------------


class QZSNMAVerifier:
    """Validates NMA tags against a registered TESLA key chain.

    Usage::

        chain   = QZSNMAChain.from_root_and_length(prn=193, gst0=1000, ...)
        verifier = QZSNMAVerifier()
        verifier.register_chain(chain)
        result = verifier.verify_tag(
            prn=193, gst_auth=1030, nav_data=b"...", tag_received=b"...",
        )
    """

    def __init__(self) -> None:
        self._chains: dict[int, QZSNMAChain] = {}

    def register_chain(self, chain: QZSNMAChain) -> None:
        """Register a TESLA chain for a QZSS PRN."""
        self._chains[chain.prn] = chain

    def verify_tag(
        self,
        prn: int,
        gst_auth: int,
        nav_data: bytes,
        tag_received: bytes,
    ) -> QZSNMAVerifyResult:
        """Verify an NMA MAC tag.

        TESLA authentication is delayed by one sub-frame: the key disclosed at
        sub-frame t authenticates the navigation data from sub-frame t-1.

        Args:
            prn:          QZSS PRN number.
            gst_auth:     GST of the sub-frame that discloses the key [s].
            nav_data:     Navigation message bytes being authenticated.
            tag_received: Received truncated MAC tag (HASH_TRUNCATION_BYTES).

        Returns:
            QZSNMAVerifyResult with authenticated=True on match.
        """
        chain = self._chains.get(prn)
        if chain is None:
            return QZSNMAVerifyResult(prn=prn, authenticated=False, reason="no_chain_registered")

        # The authenticating key is the one disclosed at gst_auth
        auth_key = chain.key_at(gst_auth)
        if auth_key is None:
            return QZSNMAVerifyResult(prn=prn, authenticated=False, reason="key_index_out_of_range")

        # TESLA delay: key at gst_auth authenticates data from gst_auth - 1 sf
        # To verify data from gst_auth - 1 sf, we use the key at gst_auth.
        # This is the TESLA disclosure model (1-sf delay).
        expected_tag = _compute_mac_tag(auth_key, nav_data)
        if not hmac.compare_digest(tag_received, expected_tag):
            return QZSNMAVerifyResult(prn=prn, authenticated=False, reason="tag_mismatch")
        return QZSNMAVerifyResult(prn=prn, authenticated=True, reason="ok")


# ---------------------------------------------------------------------------
# Epoch-level coverage layer (mirrors OSNMALayerResult interface)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QZSNMALayerResult:
    """Per-epoch QZSS NMA authentication coverage result.

    Mirrors OSNMALayerResult so callers can treat Galileo and QZSS
    authentication uniformly.

    auth_fraction         n_auth / n_total  (1.0 when no QZSS data)
    p_spoof_contribution  1 − auth_fraction  (fusion signal)
    """

    auth_fraction: float
    p_spoof_contribution: float
    n_auth: int
    n_total: int
    alert: bool


class QZSNMALayer:
    """QZSS NMA coverage monitor — L1 counterpart of OSNMALayer.

    Accepts per-satellite boolean authentication flags (or None for
    GPS-only receivers without QZSS) and returns a coverage fraction
    consistent with the OSNMA API.

    Alert threshold: < 50 % of QZSS satellites authenticated.
    """

    def __init__(self, alert_thresh: float = _AUTH_FRAC_THRESH) -> None:
        self._thresh = alert_thresh

    def assess(self, qzsnma_auth: list[bool] | None) -> QZSNMALayerResult:
        """Evaluate QZSS NMA authentication coverage for the current epoch.

        Args:
            qzsnma_auth: Per-satellite boolean authentication flags, or None.

        Returns:
            QZSNMALayerResult with auth_fraction, p_spoof_contribution, and alert.
        """
        if qzsnma_auth is None or len(qzsnma_auth) == 0:
            return QZSNMALayerResult(
                auth_fraction=1.0,
                p_spoof_contribution=0.0,
                n_auth=0,
                n_total=0,
                alert=False,
            )
        n_total = len(qzsnma_auth)
        n_auth = sum(qzsnma_auth)
        auth_fraction = n_auth / n_total
        return QZSNMALayerResult(
            auth_fraction=auth_fraction,
            p_spoof_contribution=1.0 - auth_fraction,
            n_auth=n_auth,
            n_total=n_total,
            alert=auth_fraction < self._thresh,
        )
