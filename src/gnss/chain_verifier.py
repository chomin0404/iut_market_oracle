"""TESLA Chain Manager and OSNMA Chain Verification Engine.

TESLA Chain Manager — ICD Section 5.3

数学的構造:
  K_0 = K_ROOT  (ECDSA署名済みチェーンアンカー)
  K_{i-1} = F(K_i) = trunc(key_size, Hash(K_i || alpha || CID))
  受信 K_i 検証: F^{i - last_idx}(K_i) == K_{last_idx}

鍵インデックス規則:
  subframe at GST G → key index idx = (G − gst0) / 30 + 1
  K_0 = K_ROOT はインデックス 0 に固定 (DSM-KROOT 署名で事前検証済み)
  K_1, K_2, ... が運用鍵 (インデックス昇順に開示)

鍵開示 (TESLA_DELAY = 1 の場合):
  gst_sf での subframe が tesla_key = K_{idx-1} を開示
  → gst_sf - 30 の subframe の MAC を K_{idx-1} で事後検証

既存実装との差異:
  gnss/core.py TESLAKeyChain  — K_i = SHA-256(K_{i+1} || LE32(i))  [インデックスのみ]
  gnss/osnma_inav.py GSTTESLAChain — K_i = SHA-256(K_{i+1} || GST[BE] || alpha) [GST束縛]
  本モジュール TESLAChain     — K_i = trunc(Hash(K_{i+1} || alpha || CID))   [alpha+CID束縛]

alpha 束縛 → 異なるチェーン間でのタグ再利用攻撃を防止
CID 束縛   → チェーン識別子にスコープを限定
GST 非束縛 → GST を使わないため時刻リプレイ耐性は alpha+CID 依存
"""

from __future__ import annotations

import dataclasses
import hashlib
import hmac
import logging
import struct

import numpy as np

from core.data_structures import DSMKROOTMessage, HashFunction

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Protocol constants
# ---------------------------------------------------------------------------

#: Galileo I/NAV subframe duration [s].
SUBFRAME_DURATION_S: int = 30

#: Default TESLA disclosure delay [subframes] (ICD §2.7, TESLA_DELAY = 1).
#: NOTE: gnss/core.py uses DISCLOSURE_DELAY = 2 as a conservative *simulation* margin
#:       and documents the divergence from ICD.  This value (1) is ICD-compliant.
TESLA_DELAY: int = 1

# ADKD codes (ICD §5.4.1)
ADKD_INAV_CED: int = 0  # I/NAV clock & ephemeris data
ADKD_INAV_TIMING: int = 4  # I/NAV timing parameters

# NMA status values (ICD §5.3.1)
NMA_STATUS_TEST: int = 0
NMA_STATUS_OPERATIONAL: int = 1
NMA_STATUS_DONT_USE: int = 2


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class NavAuthMessage:
    """One Galileo I/NAV subframe carrying OSNMA authentication fields.

    Attributes:
        svid:        Galileo SVID (1–36).
        gst_sf:      GST at subframe start [s].
        nav_data:    Navigation payload authenticated by ``mac_tag``.
        mac_tag:     Self-authentication tag = trunc(HMAC(K_idx, mac_input)).
                     K_idx is the key for THIS subframe (not yet public).
        tesla_key:   Disclosed K_{idx − TESLA_DELAY}, or ``None`` for the
                     first TESLA_DELAY subframes (no prior key to disclose).
        adkd:        ADKD type for ``mac_tag`` (default 0 = INAV_CED).
        cop:         Continuity-of-protection nibble (0–15).
        nma_status:  NMA status from HKROOT (0=TEST, 1=OPERATIONAL, 2=DONT_USE).
    """

    __slots__ = ("svid", "gst_sf", "nav_data", "mac_tag", "tesla_key", "adkd", "cop", "nma_status")

    def __init__(
        self,
        svid: int,
        gst_sf: int,
        nav_data: bytes,
        mac_tag: bytes,
        tesla_key: bytes | None,
        adkd: int = ADKD_INAV_CED,
        cop: int = 0,
        nma_status: int = NMA_STATUS_OPERATIONAL,
    ) -> None:
        self.svid = svid
        self.gst_sf = gst_sf
        self.nav_data = nav_data
        self.mac_tag = mac_tag
        self.tesla_key = tesla_key
        self.adkd = adkd
        self.cop = cop
        self.nma_status = nma_status

    def __repr__(self) -> str:
        return (
            f"NavAuthMessage(svid={self.svid}, gst_sf={self.gst_sf}, "
            f"has_key={self.tesla_key is not None})"
        )


@dataclasses.dataclass(frozen=True)
class ChainAuthResult:
    """Per-(SVID, subframe) TESLA chain authentication result.

    Produced by :meth:`OSNMAChainVerifier.receive` when a key is disclosed.

    Attributes:
        svid:             Galileo SVID.
        gst_sf_auth:      GST of the *authenticated* (buffered) subframe [s].
        gst_sf_disclose:  GST of the *disclosing* subframe [s].
        key_idx:          TESLA chain index of the verified/attempted key.
        key_valid:        Disclosed key lies on the hash chain anchored at K_ROOT.
        receipt_safe:     Authenticated subframe was received before disclosing GST.
        mac_valid:        MAC tag matches HMAC recomputation with disclosed key.
        authenticated:    ``key_valid AND receipt_safe AND mac_valid``.
    """

    svid: int
    gst_sf_auth: int
    gst_sf_disclose: int
    key_idx: int
    key_valid: bool
    receipt_safe: bool
    mac_valid: bool
    authenticated: bool


# ---------------------------------------------------------------------------
# TESLA Chain Manager
# ---------------------------------------------------------------------------


class TESLAChain:
    """TESLA key chain manager — forward-hash verification of disclosed keys.

    数学的構造:
      K_0 = K_ROOT  (ECDSA署名済み)
      K_{i-1} = F(K_i) = trunc(lK, Hash(K_i || alpha || CID))
      受信K_i検証: F^{i - last_idx}(K_i) == K_{last_idx}

    Usage::

        chain = TESLAChain(kroot_msg)
        # When K_i is disclosed in a received subframe:
        if chain.verify_key(disclosed_key, gst_sf_of_key):
            authenticated_key = chain.get_key(chain.key_index(gst_sf_of_key))

    Args:
        kroot: DSMKROOTMessage containing K_ROOT and all chain parameters.
               K_ROOT must have been verified a priori (ECDSA-P256 signature).
    """

    def __init__(self, kroot: DSMKROOTMessage) -> None:
        self.kroot: DSMKROOTMessage = kroot
        self.chain_id: int = kroot.cidkr
        self.hash_func: HashFunction = kroot.hash_func
        self.mac_func = kroot.mac_func
        self.key_size: int = kroot.key_size_bytes
        self.tag_size_bits: int = kroot.tag_size_bits
        self.gst0: int = kroot.gst0
        # Index 0 → K_ROOT (pre-verified anchor)
        self._verified: dict[int, bytes] = {0: kroot.kroot}
        self._last_idx: int = 0
        self._last_key: bytes = kroot.kroot
        _log.info(
            "TESLAChain[CID=%d] GST0=%d KeySz=%dB TagSz=%db",
            self.chain_id,
            self.gst0,
            self.key_size,
            self.tag_size_bits,
        )

    def key_index(self, gst_sf: int) -> int:
        """Map a subframe GST to its TESLA key chain index.

        Chain index: idx = (GST_SF − gst0) / 30 + 1

        Raises:
            ValueError: if gst_sf < gst0 (before chain epoch).
        """
        if gst_sf < self.gst0:
            raise ValueError(f"GST {gst_sf} < chain start {self.gst0}")
        return int((gst_sf - self.gst0) / SUBFRAME_DURATION_S) + 1

    def verify_key(self, key: bytes, gst_sf: int) -> bool:
        """Verify a disclosed TESLA key by tracing backward to the last anchor.

        Computes F^{idx − last_idx}(key) and compares against the most
        recently verified key.  On success, advances the verified frontier
        to idx so subsequent verifications are incremental (O(1) per step).

        Args:
            key:    Disclosed K_idx bytes.
            gst_sf: GST of the subframe associated with this key.

        Returns:
            True iff the key lies on the authenticated hash chain.
        """
        idx = self.key_index(gst_sf)

        if idx in self._verified:
            return self._verified[idx] == key

        if idx <= self._last_idx:
            _log.warning("TESLA replay: idx=%d <= last_idx=%d", idx, self._last_idx)
            return False

        # F^{idx − last_idx}(key) must equal _last_key
        computed = key
        for _ in range(idx - self._last_idx):
            computed = self._F(computed)

        if computed == self._last_key:
            self._verified[idx] = key
            self._last_key = key
            self._last_idx = idx
            _log.info("✓ TESLA key idx=%d GST=%d", idx, gst_sf)
            return True

        _log.error("✗ TESLA key FAIL idx=%d GST=%d", idx, gst_sf)
        return False

    def get_key(self, idx: int) -> bytes | None:
        """Return cached verified key at chain index ``idx``, or ``None``."""
        return self._verified.get(idx)

    def key_at_gst(self, gst_sf: int) -> bytes | None:
        """Return cached verified key for the given GST, or ``None``."""
        try:
            return self.get_key(self.key_index(gst_sf))
        except ValueError:
            return None

    def _F(self, key: bytes) -> bytes:
        """One-way hash step: F(K_i) = trunc(key_size, Hash(K_i || alpha || CID)).

        ICD §5.3.1 derivation:
          - alpha binding: prevents cross-chain preimage attacks
          - CID   binding: scopes chain to its identifier
        """
        data = key + self.kroot.alpha + bytes([self.chain_id & 0xFF])
        digest = (
            hashlib.sha256(data)
            if self.hash_func == HashFunction.SHA_256
            else hashlib.sha3_256(data)
        ).digest()
        return digest[: self.key_size]


# ---------------------------------------------------------------------------
# MAC tag computation
# ---------------------------------------------------------------------------


def compute_chain_mac_tag(
    key: bytes,
    svid: int,
    gst_sf: int,
    adkd: int,
    cop: int,
    nma_status: int,
    nav_data: bytes,
    tag_size_bits: int = 40,
) -> bytes:
    """Compute truncated HMAC-SHA256 authentication tag (ICD §5.4.1).

    MAC input layout:
        SVID[1B] || GST_sf[4B, big-endian] || CTR[1B] || NMAS[1B] || nav_data

    where:
        CTR  = ADKD[4b, hi] || COP[4b, lo]
        NMAS = nma_status & 0x03 (zero-padded to 1 byte)

    Args:
        key:           TESLA key K_i (key_size_bytes long).
        svid:          Galileo SVID (1–36).
        gst_sf:        GST at subframe start [s].
        adkd:          ADKD type (0=INAV_CED, 4=INAV_TIMING, 12=SLOW_MAC).
        cop:           Continuity-of-protection nibble (0–15).
        nma_status:    NMA status (0–3).
        nav_data:      Navigation data payload bytes.
        tag_size_bits: Truncation length in bits (must be a multiple of 8).

    Returns:
        ``tag_size_bits // 8`` bytes of truncated HMAC-SHA256.
    """
    ctr = ((adkd & 0xF) << 4) | (cop & 0xF)
    mac_input = (
        struct.pack("B", svid & 0xFF)
        + struct.pack(">I", gst_sf & 0xFFFFFFFF)
        + struct.pack("B", ctr)
        + struct.pack("B", nma_status & 0x3)
        + nav_data
    )
    digest = hmac.new(key, mac_input, hashlib.sha256).digest()
    return digest[: tag_size_bits // 8]


# ---------------------------------------------------------------------------
# Chain generator (simulation / testing helper)
# ---------------------------------------------------------------------------


def generate_tesla_chain(
    n: int,
    template: DSMKROOTMessage,
    seed: int = 42,
) -> tuple[DSMKROOTMessage, list[bytes]]:
    """Pre-compute a full TESLA key chain for testing and simulation.

    Generates n+1 keys [K_0, K_1, ..., K_n] where:
      - K_n is a random seed (most-future key, kept secret by the satellite authority)
      - K_{i} = F(K_{i+1}) for i in [0, n-1]  (backward derivation)
      - K_0 = K_ROOT (chain anchor, authenticated by DSM-KROOT signature)

    In the real protocol K_n is pre-generated offline; K_0 is signed and
    published in DSM-KROOT; K_1, K_2, ..., K_n are disclosed progressively.

    Subframe mapping:
      Subframe sf_idx has key index idx = sf_idx + 1.
      Its MAC tag is computed with K_{idx}.
      The subframe discloses K_{idx − TESLA_DELAY} (None for idx <= TESLA_DELAY).

    Args:
        n:        Number of subframes (keys K_1 … K_n are operational keys).
        template: DSMKROOTMessage supplying alpha, cidkr, hash_func, key_size_bytes.
                  The ``kroot`` field is replaced with the derived K_0.
        seed:     RNG seed for reproducible K_n generation.

    Returns:
        Tuple ``(kroot_msg, keys)`` where:
          - ``kroot_msg`` is a DSMKROOTMessage with ``kroot`` = K_0
          - ``keys[i]`` = K_i for i in [0, n]
    """
    rng = np.random.default_rng(seed)
    keys: list[bytes] = [b""] * (n + 1)
    # K_n: random seed (highest index = most-future, never transmitted)
    keys[n] = bytes(rng.integers(0, 256, template.key_size_bytes, dtype=np.uint8))

    # Derive backward: K_{i} = F(K_{i+1}) using the same _F logic as TESLAChain
    for i in range(n - 1, -1, -1):
        data = keys[i + 1] + template.alpha + bytes([template.cidkr & 0xFF])
        digest = (
            hashlib.sha256(data)
            if template.hash_func == HashFunction.SHA_256
            else hashlib.sha3_256(data)
        ).digest()
        keys[i] = digest[: template.key_size_bytes]

    # K_0 = K_ROOT is the chain anchor embedded in the DSMKROOTMessage
    kroot_msg = dataclasses.replace(template, kroot=keys[0])
    return kroot_msg, keys


# ---------------------------------------------------------------------------
# OSNMA chain verification engine
# ---------------------------------------------------------------------------


class OSNMAChainVerifier:
    """OSNMA verification engine backed by TESLAChain.

    Implements the three-check TESLA authentication protocol per subframe:

      1. **key_valid**    — TESLAChain.verify_key() confirms F^{steps}(K_i) == K_anchor
      2. **receipt_safe** — buffered subframe arrived before key-disclosure GST
      3. **mac_valid**    — recomputed HMAC matches the buffered MAC tag

    Ingestion::

        kroot_msg = DSMKROOTMessage(...)     # pre-verified K_ROOT
        verifier = OSNMAChainVerifier(kroot_msg)

        for msg, recv_t in stream:
            result = verifier.receive(msg, recv_t)
            if result is not None and result.authenticated:
                print(f"SVID {msg.svid} authenticated at GST {result.gst_sf_auth}")

    Notes:
        - Returns ``None`` for the first TESLA_DELAY subframes (no key disclosed).
        - Returns ``None`` if gst_auth falls before the chain epoch (gst0).
        - Thread-safety: instances are NOT thread-safe.

    Args:
        kroot_msg:    Fully validated DSMKROOTMessage (K_ROOT pre-verified by ECDSA).
        tesla_delay:  TESLA disclosure delay in subframes (default 1, per ICD).
        nma_accept:   Acceptable NMA status values.
                      Defaults to ``{NMA_STATUS_TEST, NMA_STATUS_OPERATIONAL}``.
    """

    def __init__(
        self,
        kroot_msg: DSMKROOTMessage,
        tesla_delay: int = TESLA_DELAY,
        nma_accept: frozenset[int] | None = None,
    ) -> None:
        self._chain = TESLAChain(kroot_msg)
        self._delay = tesla_delay
        self._nma_accept: frozenset[int] = (
            nma_accept
            if nma_accept is not None
            else frozenset({NMA_STATUS_TEST, NMA_STATUS_OPERATIONAL})
        )
        # Buffer: (svid, gst_sf) → (NavAuthMessage, recv_time_gst)
        self._buf: dict[tuple[int, int], tuple[NavAuthMessage, float]] = {}
        # Cumulative per-SVID auth state (once True, stays True)
        self._auth_state: dict[int, bool] = {}

    # ------------------------------------------------------------------
    # Main ingestion path
    # ------------------------------------------------------------------

    def receive(
        self,
        msg: NavAuthMessage,
        recv_time_gst: float,
    ) -> ChainAuthResult | None:
        """Process one I/NAV subframe and attempt TESLA authentication.

        The subframe is always buffered for receipt-safety checking.
        A :class:`ChainAuthResult` is returned only when ``msg.tesla_key``
        is not ``None`` (i.e., a TESLA key is disclosed) and the associated
        authenticated subframe falls within the chain epoch.

        Args:
            msg:            One I/NAV subframe with OSNMA fields.
            recv_time_gst:  GST [s] at which this subframe was received.

        Returns:
            :class:`ChainAuthResult`, or ``None`` if no authentication attempt
            was made (early subframes, or gst_auth before chain start).
        """
        # Always buffer — needed for receipt-safety lookup in future subframes
        self._buf[(msg.svid, msg.gst_sf)] = (msg, recv_time_gst)

        if msg.tesla_key is None:
            return None  # no key disclosed yet

        # GST of the subframe being authenticated (buffered TESLA_DELAY epochs ago)
        gst_auth = msg.gst_sf - self._delay * SUBFRAME_DURATION_S

        try:
            key_idx = self._chain.key_index(gst_auth)
        except ValueError:
            # gst_auth before chain epoch — nothing to authenticate
            _log.debug(
                "SVID=%d: gst_auth=%d < chain gst0=%d; skipping",
                msg.svid,
                gst_auth,
                self._chain.gst0,
            )
            return None

        # ── 1. TESLA key chain verification ────────────────────────────
        key_valid = self._chain.verify_key(msg.tesla_key, gst_auth)

        # ── 2. Receipt safety ──────────────────────────────────────────
        # The buffered subframe at gst_auth must have been received before
        # the disclosing subframe at msg.gst_sf arrived.
        buf_entry = self._buf.get((msg.svid, gst_auth))
        buffered, buf_recv = buf_entry if buf_entry else (None, None)
        receipt_safe = (
            buffered is not None and buf_recv is not None and buf_recv < float(msg.gst_sf)
        )

        # ── 3. MAC tag verification ────────────────────────────────────
        # The buffered subframe's MAC was computed with K_{key_idx} = tesla_key.
        mac_valid = False
        if key_valid and buffered is not None and msg.nma_status in self._nma_accept:
            expected = compute_chain_mac_tag(
                key=msg.tesla_key,
                svid=msg.svid,
                gst_sf=gst_auth,
                adkd=buffered.adkd,
                cop=buffered.cop,
                nma_status=buffered.nma_status,
                nav_data=buffered.nav_data,
                tag_size_bits=self._chain.tag_size_bits,
            )
            mac_valid = buffered.mac_tag == expected

        authenticated = key_valid and receipt_safe and mac_valid
        if authenticated:
            self._auth_state[msg.svid] = True

        return ChainAuthResult(
            svid=msg.svid,
            gst_sf_auth=gst_auth,
            gst_sf_disclose=msg.gst_sf,
            key_idx=key_idx,
            key_valid=key_valid,
            receipt_safe=receipt_safe,
            mac_valid=mac_valid,
            authenticated=authenticated,
        )

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    def authenticated_svids(self, svids: list[int]) -> list[bool]:
        """Return per-SVID cumulative authentication flags.

        A SVID is True if at least one subframe from it produced
        ``authenticated=True`` (monotonically non-decreasing).

        Args:
            svids: Ordered list of Galileo SVIDs to query.

        Returns:
            ``list[bool]`` of length ``len(svids)``.
        """
        return [self._auth_state.get(svid, False) for svid in svids]

    def reset(self) -> None:
        """Clear buffer and authentication state for reprocessing."""
        self._buf.clear()
        self._auth_state.clear()
        # Re-initialise chain (preserves K_ROOT anchor)
        self._chain = TESLAChain(self._chain.kroot)
