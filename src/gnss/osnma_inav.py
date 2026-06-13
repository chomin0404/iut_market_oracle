"""Galileo I/NAV OSNMA verification engine (research model).

Protocol reference: Galileo OSNMA SIS ICD OS-SIS-ICD-OSNMA v1.1 (simplified).

Subframe / page timing:
  1 page     = 2 s  (word type 0 carries OSNMA data)
  1 subframe = 15 pages = 30 s
  HKROOT block: transmitted in page 0 of each subframe (NMA status + K_ROOT + sig)
  MACK block:   transmitted in pages 1–14 of each subframe (tag-0 + disclosed key)

TESLA key derivation (per ICD §2.7):
  K_i = trunc_ks( SHA-256( K_{i+1} || GST_sf_i[4B, big-endian] || alpha[6B] ) )
  GST_sf_i : Galileo System Time at the start of subframe i [seconds]
  alpha     : 6-byte nonce from HKROOT (prevents cross-chain preimage attacks)
  Disclosure: K_{i−TESLA_DELAY} is disclosed in subframe i (TESLA_DELAY = 1).

Key differences from gnss/core.py TESLAKeyChain:
  core.py        — K_i = SHA-256(K_{i+1} || LE32(i))   (index-only derivation)
  osnma_inav.py  — K_i = SHA-256(K_{i+1} || GST_sf_i[BE] || alpha)
                   GST-binding prevents temporal-replay across time windows.
                   alpha-binding prevents cross-chain tag reuse.

MAC tag computation (tag-0 self-authentication, ICD §2.6.2):
  mac_in  = SVID[1B] || GST_sf[4B,BE] || CTR[1B] || NMAS[1B] || nav_data
  CTR     = ADKD[4b,hi] || COP[4b,lo]
  mac_out = trunc_40( HMAC-SHA256( K_i, mac_in ) )

Three-check authentication (per epoch / subframe):
  1. key_valid    — K_{i−TESLA_DELAY} lies on chain anchored at K_ROOT
  2. receipt_safe — subframe i−TESLA_DELAY was received before GST_sf(i)
  3. mac_valid    — tag-0 in subframe i−TESLA_DELAY matches HMAC recomputation

Produces list[bool] per SVID, compatible with OSNMALayer (layers/authentication.py)
and ResilienceTwin Pillar 1.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import struct
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Protocol constants (Galileo OSNMA SIS ICD §2)
# ---------------------------------------------------------------------------

PAGES_PER_SUBFRAME: int = 15  # 15 pages × 2 s = 30 s/subframe
PAGE_DURATION_S: int = 2  # seconds per I/NAV page
SUBFRAME_DURATION_S: int = 30  # seconds per subframe

TESLA_DELAY: int = 1  # K_{i−1} disclosed in subframe i
KEY_SIZE_BITS: int = 128  # TESLA key size [bits]
KEY_SIZE_BYTES: int = KEY_SIZE_BITS // 8
MAC_TAG_BITS: int = 40  # MACK tag truncation [bits]
MAC_TAG_BYTES: int = MAC_TAG_BITS // 8

ALPHA_BYTES: int = 6  # HKROOT nonce size [bytes]
NAV_DATA_BYTES: int = 32  # Navigation data payload per subframe [bytes]

# NMA status (2-bit HKROOT field)
NMA_STATUS_TEST: int = 0
NMA_STATUS_OPERATIONAL: int = 1
NMA_STATUS_DONT_USE: int = 2

# ADKD (Authentication Data and Key Derivation) types
ADKD_INAV_CED: int = 0  # I/NAV clock & ephemeris data
ADKD_INAV_TIMING: int = 4  # I/NAV timing parameters
ADKD_SLOW_MAC: int = 12  # Slow MAC (cross-constellation)

DEFAULT_GST_START: int = 0  # default GST epoch for simulation [s]
DEFAULT_SEED: int = 42

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HkrootMessage:
    """HKROOT block — transmitted in page 0 of each I/NAV subframe.

    Carries NMA status, TESLA chain parameters, K_ROOT, and root signature.
    Fields follow Galileo OSNMA SIS ICD §2.5 (simplified for research use).

    In the real ICD, HKROOT is 104 bits spread over the first I/NAV page:
        NMA_H (8b) | TESLA_H (8b) | kroot_wn (12b) | kroot_tow (20b) |
        alpha (48b) | K_ROOT (key_size_bits) | DS (signature)
    """

    nma_status: int  # 2-bit NMA status: TEST=0, OPERATIONAL=1, DONT_USE=2
    chain_id: int  # 2-bit: identifies which OSNMA chain is in force
    nb_dk: int  # key-size selector (0→96b, 1→104b, 2→128b)
    pkid: int  # public key ID used to sign K_ROOT (0–2)
    kroot_wn: int  # Galileo week number for K_ROOT epoch
    kroot_tow: int  # time of week for K_ROOT epoch [s]
    alpha: bytes  # 6-byte (48-bit) nonce — salt for key derivation
    kroot: bytes  # K_ROOT bytes (KEY_SIZE_BYTES = 16)
    ds: bytes  # digital signature over HKROOT fields


@dataclass(frozen=True)
class MackTagEntry:
    """One MACK authentication tag in the MACK block.

    In the real ICD, each tag entry occupies:
        ADKD[4b] | COP[4b] | tag[40b]  = 6 bytes per entry
    """

    adkd: int  # Authentication Data and Key Derivation type
    cop: int  # Continuity / offset padding (4-bit)
    tag: bytes  # 40-bit MAC tag (MAC_TAG_BYTES = 5 bytes)


@dataclass(frozen=True)
class MackMessage:
    """MACK block — distributed over pages 1–14 of each I/NAV subframe.

    Carries:
      tag-0      — self-authentication MAC tag for the data in *this* subframe
      tags       — cross-authentication tags (may be empty in simplified model)
      tesla_key  — K_{i−TESLA_DELAY} disclosed in this subframe (None for i < DELAY)
      key_id     — chain index of the disclosed key

    Verification flow:
        Receiver buffers (SVID, sf_i) when it arrives.
        In sf_{i+TESLA_DELAY}: tesla_key = K_i is disclosed.
        Receiver then verifies tag-0 from sf_i using K_i.
    """

    gst_sf: int  # GST at the start of this subframe [s]
    tag0: bytes  # self-auth MAC tag (MAC_TAG_BYTES = 5 bytes)
    tag0_adkd: int  # ADKD type for tag-0 (typically ADKD_INAV_CED = 0)
    tags: list[MackTagEntry]
    tesla_key: bytes | None  # disclosed K_{key_id}, or None if key_id < 0
    key_id: int  # chain index of the disclosed key (sf_idx − TESLA_DELAY)


@dataclass
class SubframeData:
    """Aggregated I/NAV OSNMA data for one (SVID, subframe) pair.

    Produced by INavOSNMASimulator or ingested from a real receiver.
    """

    svid: int  # Galileo SVID (1–36)
    subframe_idx: int  # 0-based subframe counter
    gst_sf: int  # GST at subframe start [s]
    nav_data: bytes  # navigation data being authenticated (NAV_DATA_BYTES)
    hkroot: HkrootMessage
    mack: MackMessage
    recv_time_gst: float  # GST when this subframe was received [s]


@dataclass(frozen=True)
class SubframeVerifyResult:
    """Per-(SVID, subframe) OSNMA verification outcome."""

    svid: int
    subframe_idx: int
    gst_sf: int
    key_valid: bool  # disclosed TESLA key lies on authenticated chain
    mac_valid: bool  # tag-0 MAC matches HMAC recomputation
    receipt_safe: bool  # subframe received before key disclosure GST
    nma_ok: bool  # NMA status is OPERATIONAL or TEST
    authenticated: bool  # key_valid AND mac_valid AND receipt_safe AND nma_ok


# ---------------------------------------------------------------------------
# GST-aware TESLA key chain
# ---------------------------------------------------------------------------


class GSTTESLAChain:
    """Hash-chain key generation anchored to Galileo System Time.

    Key derivation (Galileo OSNMA SIS ICD §2.7, simplified):
        K_i = trunc_ks( SHA-256( K_{i+1} || GST_sf_i[4B,BE] || alpha[6B] ) )

    The GST and alpha fields prevent two attack classes that the index-only
    TESLAKeyChain in gnss/core.py cannot resist:
      - Temporal replay: same chain index replayed at a different GST epoch
        → rejected because GST_sf_i differs in the derivation
      - Cross-chain tag reuse: tags computed under a different alpha value
        → rejected because alpha is chain-specific and carried in HKROOT

    Chain direction (right = highest index = root):
        K_0 ←H— K_1 ←H— … ←H— K_{n−1}   (K_{n−1} = K_ROOT, never transmitted)
    """

    KEY_BYTES: int = KEY_SIZE_BYTES

    def __init__(
        self,
        n: int,
        gst_start: int,
        alpha: bytes,
        seed: int = DEFAULT_SEED,
    ) -> None:
        """
        Args:
            n:          chain length (total number of subframes + safety margin)
            gst_start:  GST [s] at subframe index 0
            alpha:      ALPHA_BYTES-byte nonce from HKROOT
            seed:       RNG seed for root key K_{n-1} generation
        """
        if len(alpha) != ALPHA_BYTES:
            raise ValueError(f"alpha must be {ALPHA_BYTES} bytes, got {len(alpha)}")
        rng = np.random.default_rng(seed)
        self._n = n
        self._gst_start = gst_start
        self._alpha = alpha
        self._keys: list[bytes] = [b""] * n
        # K_{n-1} is the root — sampled randomly, signed by OSNMA authority
        self._keys[n - 1] = bytes(rng.integers(0, 256, self.KEY_BYTES, dtype=np.uint8))
        # Derive backward: K_{n-2}, …, K_0
        for i in range(n - 2, -1, -1):
            gst_i = gst_start + i * SUBFRAME_DURATION_S
            self._keys[i] = self.hash_step(self._keys[i + 1], gst_i, alpha)

    @staticmethod
    def hash_step(k_succ: bytes, gst_sf: int, alpha: bytes) -> bytes:
        """K_i = trunc_ks( SHA-256( K_{i+1} || GST_sf_i[4B,BE] || alpha ) )."""
        msg = k_succ + struct.pack(">I", gst_sf & 0xFFFFFFFF) + alpha
        return hashlib.sha256(msg).digest()[: GSTTESLAChain.KEY_BYTES]

    @property
    def root(self) -> bytes:
        """K_ROOT = K_{n−1} (highest-index key, never transmitted)."""
        return self._keys[-1]

    def get_key(self, idx: int) -> bytes:
        """Return K_{idx}."""
        if idx < 0 or idx >= self._n:
            raise IndexError(f"Key index {idx} out of range [0, {self._n})")
        return self._keys[idx]

    def gst_of(self, idx: int) -> int:
        """Return GST [s] at the start of subframe idx."""
        return self._gst_start + idx * SUBFRAME_DURATION_S

    def verify_key(
        self,
        key: bytes,
        idx: int,
        anchor_idx: int,
        anchor_key: bytes,
    ) -> bool:
        """Verify that key[idx] lies on the chain anchored at anchor_key[anchor_idx].

        Invariant: anchor_idx > idx (keys are derived backward from root).
        Returns False immediately if idx >= anchor_idx.
        """
        if idx >= anchor_idx:
            return False
        current = anchor_key
        for i in range(anchor_idx - 1, idx - 1, -1):
            gst_i = self._gst_start + i * SUBFRAME_DURATION_S
            current = self.hash_step(current, gst_i, self._alpha)
        return current == key


# ---------------------------------------------------------------------------
# MAC tag computation
# ---------------------------------------------------------------------------


def compute_mac_tag(
    key: bytes,
    svid: int,
    gst_sf: int,
    adkd: int,
    cop: int,
    nma_status: int,
    nav_data: bytes,
) -> bytes:
    """Compute 40-bit OSNMA MAC tag (truncated HMAC-SHA256).

    MAC input layout (Galileo OSNMA SIS ICD §2.6.2):
        SVID[1B] || GST_sf[4B,BE] || CTR[1B] || NMAS[1B] || nav_data

    CTR  = ADKD[4b,hi] || COP[4b,lo]
    NMAS = nma_status[2b] zero-padded to 1 byte

    Args:
        key:        TESLA key K_i (KEY_SIZE_BYTES)
        svid:       Galileo SVID (1–36)
        gst_sf:     GST at subframe start [s]
        adkd:       ADKD type (e.g. ADKD_INAV_CED = 0)
        cop:        continuity / offset padding nibble (0–15)
        nma_status: NMA status from HKROOT
        nav_data:   navigation data bytes (NAV_DATA_BYTES)

    Returns:
        MAC_TAG_BYTES (5) bytes of truncated HMAC-SHA256.
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
    return digest[:MAC_TAG_BYTES]


# ---------------------------------------------------------------------------
# I/NAV OSNMA verification engine
# ---------------------------------------------------------------------------


class INavOSNMAEngine:
    """Galileo I/NAV OSNMA verification engine.

    Processes SubframeData objects one at a time and performs the three-check
    OSNMA verification protocol:

        1. key_valid    — K_{key_id} lies on the chain anchored at K_ROOT
        2. receipt_safe — subframe key_id was received before GST_sf(current)
        3. mac_valid    — tag-0 in the buffered subframe matches HMAC recomputation

    Produces per-SVID authentication flags (list[bool]) compatible with
    OSNMALayer.assess(osnma_auth=...) in layers/authentication.py.

    Typical usage:
        engine = INavOSNMAEngine(**sim.engine_params)
        for sf in subframes:
            result = engine.verify_subframe(sf)
        flags = engine.authenticated_svids(svid_list)  # → list[bool]
    """

    def __init__(
        self,
        kroot: bytes,
        kroot_idx: int,
        gst_start: int,
        alpha: bytes,
        nma_status_accept: frozenset[int] | None = None,
    ) -> None:
        """
        Args:
            kroot:              K_ROOT bytes (KEY_SIZE_BYTES), verified a priori
            kroot_idx:          Chain index of K_ROOT (must be > all subframe indices)
            gst_start:          GST [s] at subframe index 0
            alpha:              ALPHA_BYTES-byte nonce from HKROOT
            nma_status_accept:  Acceptable NMA status values;
                                defaults to {NMA_STATUS_TEST, NMA_STATUS_OPERATIONAL}
        """
        if len(alpha) != ALPHA_BYTES:
            raise ValueError(f"alpha must be {ALPHA_BYTES} bytes, got {len(alpha)}")
        self._kroot = kroot
        self._kroot_idx = kroot_idx
        self._gst_start = gst_start
        self._alpha = alpha
        self._accept_status: frozenset[int] = (
            nma_status_accept
            if nma_status_accept is not None
            else frozenset({NMA_STATUS_TEST, NMA_STATUS_OPERATIONAL})
        )
        # Anchored verified keys: chain_index → key bytes
        self._verified_keys: dict[int, bytes] = {kroot_idx: kroot}
        # Buffer: (svid, sf_idx) → SubframeData
        self._buffer: dict[tuple[int, int], SubframeData] = {}
        # Already-verified buffer entries (avoid double-counting)
        self._verified_set: set[tuple[int, int]] = set()
        # Cumulative per-SVID authentication state
        self._auth_map: dict[int, bool] = {}

    # ------------------------------------------------------------------
    # Internal key verification
    # ------------------------------------------------------------------

    def _verify_key(self, key: bytes, idx: int) -> bool:
        """Verify key[idx] against the nearest verified anchor with index > idx."""
        anchor_idx: int | None = None
        anchor_key: bytes | None = None
        for ae, ak in self._verified_keys.items():
            if ae > idx and (anchor_idx is None or ae < anchor_idx):
                anchor_idx, anchor_key = ae, ak
        if anchor_idx is None or anchor_key is None:
            return False
        current = anchor_key
        for i in range(anchor_idx - 1, idx - 1, -1):
            gst_i = self._gst_start + i * SUBFRAME_DURATION_S
            current = GSTTESLAChain.hash_step(current, gst_i, self._alpha)
        return current == key

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify_subframe(self, sf: SubframeData) -> SubframeVerifyResult:
        """Verify one subframe's OSNMA data and update internal state.

        The subframe is always buffered; verification is attempted only when
        the disclosed TESLA key (mack.key_id ≥ 0) has arrived.

        Returns:
            SubframeVerifyResult with per-check flags.
        """
        buf_key = (sf.svid, sf.subframe_idx)
        self._buffer[buf_key] = sf

        mack = sf.mack
        hkroot = sf.hkroot

        # NMA status check
        nma_ok = hkroot.nma_status in self._accept_status

        # Early exit: no key disclosed in this subframe (first TESLA_DELAY subframes)
        if mack.tesla_key is None or mack.key_id < 0:
            return SubframeVerifyResult(
                svid=sf.svid,
                subframe_idx=sf.subframe_idx,
                gst_sf=sf.gst_sf,
                key_valid=False,
                mac_valid=False,
                receipt_safe=False,
                nma_ok=nma_ok,
                authenticated=False,
            )

        # 1. TESLA key chain verification
        key_valid = self._verify_key(mack.tesla_key, mack.key_id)
        if key_valid:
            self._verified_keys[mack.key_id] = mack.tesla_key

        # 2. Receipt safety
        # Subframe key_id must have been received before the disclosure GST.
        # Disclosure happens at GST_sf(i) where i = key_id + TESLA_DELAY.
        # sf.gst_sf == GST_sf(i) is the current subframe's start time.
        auth_buf_key = (sf.svid, mack.key_id)
        buffered_sf = self._buffer.get(auth_buf_key)
        receipt_safe = buffered_sf is not None and buffered_sf.recv_time_gst < sf.gst_sf

        # 3. MAC tag-0 verification
        # tag-0 in the buffered subframe was computed as:
        #   HMAC(K_{key_id}, SVID || GST_sf_{key_id} || CTR || NMAS || nav_data)
        # We recompute and compare.
        mac_valid = False
        if key_valid and buffered_sf is not None:
            expected_tag = compute_mac_tag(
                key=mack.tesla_key,
                svid=sf.svid,
                gst_sf=buffered_sf.gst_sf,
                adkd=buffered_sf.mack.tag0_adkd,
                cop=0,
                nma_status=buffered_sf.hkroot.nma_status,
                nav_data=buffered_sf.nav_data,
            )
            mac_valid = buffered_sf.mack.tag0 == expected_tag

        authenticated = key_valid and mac_valid and receipt_safe and nma_ok
        # Update cumulative state: once authenticated, stays authenticated
        self._auth_map[sf.svid] = self._auth_map.get(sf.svid, False) or authenticated
        self._verified_set.add(buf_key)

        return SubframeVerifyResult(
            svid=sf.svid,
            subframe_idx=sf.subframe_idx,
            gst_sf=sf.gst_sf,
            key_valid=key_valid,
            mac_valid=mac_valid,
            receipt_safe=receipt_safe,
            nma_ok=nma_ok,
            authenticated=authenticated,
        )

    def authenticated_svids(self, svids: list[int]) -> list[bool]:
        """Return per-SVID authentication flags for the given SVID list.

        True = at least one subframe from this SVID was successfully verified.
        The returned list is in the same order as `svids`, compatible with
        OSNMALayer.assess(osnma_auth=...).

        Args:
            svids: ordered list of Galileo SVIDs to query

        Returns:
            list[bool] of length len(svids)
        """
        return [self._auth_map.get(svid, False) for svid in svids]

    def reset(self) -> None:
        """Clear buffer and authentication state (call before re-running on new data)."""
        self._verified_keys = {self._kroot_idx: self._kroot}
        self._buffer.clear()
        self._verified_set.clear()
        self._auth_map.clear()


# ---------------------------------------------------------------------------
# Navigation data helper
# ---------------------------------------------------------------------------


def make_inav_nav_data(svid: int, sf_idx: int) -> bytes:
    """Deterministic dummy I/NAV navigation data (32 bytes).

    In a real receiver, this would be the CED (clock+ephemeris data) or
    timing parameters extracted from the I/NAV word type fields.

    Deterministic via SHA-256(SVID || subframe_idx) so tests are reproducible.
    """
    return hashlib.sha256(struct.pack(">II", svid, sf_idx)).digest()[:NAV_DATA_BYTES]


# ---------------------------------------------------------------------------
# I/NAV OSNMA simulator (for testing and research)
# ---------------------------------------------------------------------------


class INavOSNMASimulator:
    """Generates synthetic Galileo I/NAV subframes for testing INavOSNMAEngine.

    Creates a GSTTESLAChain and signs K_ROOT with a deterministic dummy
    signature, then produces SubframeData objects with correct HKROOT + MACK
    fields.  The protocol follows the three-check verification flow described
    in the module docstring.

    Tampering parameters on make_subframe() allow controlled injection of:
      - random tag-0 (tamper_tag0)         → mac_valid = False
      - wrong nav_data (tamper_nav_data)   → mac_valid = False
      - wrong TESLA key (tamper_tesla_key) → key_valid = False

    Example::

        sim = INavOSNMASimulator(svids=[1, 2, 3], n_subframes=10)
        engine = INavOSNMAEngine(**sim.engine_params)

        for svid in sim.svids:
            for sf_idx in range(sim.n_subframes):
                sf = sim.make_subframe(svid, sf_idx)
                engine.verify_subframe(sf)

        flags = engine.authenticated_svids(sim.svids)  # all True for sf_idx ≥ 1
    """

    def __init__(
        self,
        svids: list[int],
        n_subframes: int,
        gst_start: int = DEFAULT_GST_START,
        nma_status: int = NMA_STATUS_OPERATIONAL,
        seed: int = DEFAULT_SEED,
    ) -> None:
        """
        Args:
            svids:        list of Galileo SVIDs to simulate
            n_subframes:  number of subframes per SVID
            gst_start:    GST [s] at subframe 0
            nma_status:   NMA status value embedded in HKROOT
            seed:         RNG seed for alpha and K_ROOT generation
        """
        rng = np.random.default_rng(seed)
        self._svids = list(svids)
        self._n = n_subframes
        self._gst_start = gst_start
        self._nma_status = nma_status

        # Random 6-byte alpha nonce
        self._alpha: bytes = bytes(rng.integers(0, 256, ALPHA_BYTES, dtype=np.uint8))

        # TESLA chain: n_subframes + TESLA_DELAY + 1 slots so K_ROOT index
        # is strictly greater than any subframe index
        chain_len = n_subframes + TESLA_DELAY + 1
        self._chain = GSTTESLAChain(
            n=chain_len,
            gst_start=gst_start,
            alpha=self._alpha,
            seed=seed,
        )
        self._kroot_idx: int = chain_len - 1
        self._kroot: bytes = self._chain.root

        # Dummy root signature: SHA-256(K_ROOT || gst_start[4B])
        # In a real system: ECDSA-P256 or RLWE (see gnss/pqc.py)
        self._ds: bytes = hashlib.sha256(
            self._kroot + struct.pack(">I", gst_start & 0xFFFFFFFF)
        ).digest()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def svids(self) -> list[int]:
        return list(self._svids)

    @property
    def n_subframes(self) -> int:
        return self._n

    @property
    def engine_params(self) -> dict:
        """Keyword arguments for INavOSNMAEngine(**sim.engine_params)."""
        return {
            "kroot": self._kroot,
            "kroot_idx": self._kroot_idx,
            "gst_start": self._gst_start,
            "alpha": self._alpha,
        }

    # ------------------------------------------------------------------
    # Subframe generation
    # ------------------------------------------------------------------

    def make_subframe(
        self,
        svid: int,
        sf_idx: int,
        recv_delay_s: float = 1.0,
        tamper_tag0: bool = False,
        tamper_nav_data: bool = False,
        tamper_tesla_key: bool = False,
        late_recv_delay_s: float | None = None,
    ) -> SubframeData:
        """Generate one SubframeData with correct (or tampered) OSNMA fields.

        Args:
            svid:               Galileo SVID
            sf_idx:             subframe index (0-based)
            recv_delay_s:       reception delay after subframe GST start [s].
                                Must be < SUBFRAME_DURATION_S for receipt safety.
            tamper_tag0:        replace tag-0 with random bytes → mac_valid=False
            tamper_nav_data:    replace nav_data with random bytes → mac_valid=False
            tamper_tesla_key:   replace disclosed TESLA key with zeros → key_valid=False
            late_recv_delay_s:  override recv_time to simulate late arrival
                                (e.g. SUBFRAME_DURATION_S + 5) → receipt_safe=False
        """
        gst_sf = self._gst_start + sf_idx * SUBFRAME_DURATION_S
        # Always compute tag0 over the authentic nav_data.
        # When tamper_nav_data=True the SubframeData payload is replaced with
        # random bytes *after* tag0 is sealed — this models an adversary who
        # modifies the navigation message content without access to K_i.
        authentic_nav_data = make_inav_nav_data(svid, sf_idx)
        nav_data = os.urandom(NAV_DATA_BYTES) if tamper_nav_data else authentic_nav_data

        # Disclosed key: K_{sf_idx − TESLA_DELAY}
        key_id = sf_idx - TESLA_DELAY
        tesla_key: bytes | None = None
        if key_id >= 0:
            if tamper_tesla_key:
                tesla_key = bytes(KEY_SIZE_BYTES)  # all-zero key → wrong hash
            else:
                tesla_key = self._chain.get_key(key_id)

        hkroot = HkrootMessage(
            nma_status=self._nma_status,
            chain_id=0,
            nb_dk=2,
            pkid=0,
            kroot_wn=self._gst_start // (7 * 24 * 3600),
            kroot_tow=self._gst_start % (7 * 24 * 3600),
            alpha=self._alpha,
            kroot=self._kroot,
            ds=self._ds,
        )

        # tag-0 = HMAC(K_{sf_idx}, SVID, GST_sf, ADKD_INAV_CED, COP=0, NMAS, nav_data)
        # K_{sf_idx} will be disclosed in subframe sf_idx + TESLA_DELAY.
        mac_key = self._chain.get_key(sf_idx)
        tag0 = compute_mac_tag(
            key=mac_key,
            svid=svid,
            gst_sf=gst_sf,
            adkd=ADKD_INAV_CED,
            cop=0,
            nma_status=self._nma_status,
            nav_data=authentic_nav_data,  # tag0 always seals the authentic data
        )
        if tamper_tag0:
            tag0 = os.urandom(MAC_TAG_BYTES)

        mack = MackMessage(
            gst_sf=gst_sf,
            tag0=tag0,
            tag0_adkd=ADKD_INAV_CED,
            tags=[],
            tesla_key=tesla_key,
            key_id=key_id,
        )

        if late_recv_delay_s is not None:
            recv_time = float(gst_sf) + late_recv_delay_s
        else:
            recv_time = float(gst_sf) + recv_delay_s

        return SubframeData(
            svid=svid,
            subframe_idx=sf_idx,
            gst_sf=gst_sf,
            nav_data=nav_data,
            hkroot=hkroot,
            mack=mack,
            recv_time_gst=recv_time,
        )


# ---------------------------------------------------------------------------
# End-to-end simulation helper
# ---------------------------------------------------------------------------


def run_inav_simulation(
    svids: list[int],
    n_subframes: int = 10,
    attack_prob: float = 0.0,
    seed: int = DEFAULT_SEED,
) -> dict[int, list[SubframeVerifyResult]]:
    """Run an end-to-end Galileo I/NAV OSNMA simulation.

    Each (SVID, subframe) is either genuine or randomly tampered (tag-0 replaced
    with random bytes), simulating a modified-replay attack.

    Args:
        svids:        list of SVIDs to simulate
        n_subframes:  subframes per SVID
        attack_prob:  probability of replacing tag-0 with random bytes per subframe
        seed:         RNG seed

    Returns:
        Dict mapping SVID → list[SubframeVerifyResult] (length n_subframes).
    """
    rng = np.random.default_rng(seed)
    sim = INavOSNMASimulator(svids=svids, n_subframes=n_subframes, seed=seed)
    engine = INavOSNMAEngine(**sim.engine_params)

    results: dict[int, list[SubframeVerifyResult]] = {svid: [] for svid in svids}

    for sf_idx in range(n_subframes):
        for svid in svids:
            tampered = rng.random() < attack_prob
            sf = sim.make_subframe(svid, sf_idx, tamper_tag0=tampered)
            result = engine.verify_subframe(sf)
            results[svid].append(result)

    return results
