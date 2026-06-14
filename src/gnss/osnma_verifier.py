"""Galileo OSNMA end-to-end verification engine.

Extends osnma_inav.py with three new capabilities:

  1. ECDSA-P256 K_ROOT digital signature verification
     The DS field in DSM-KROOT is a 64-byte raw signature (r || s).
     This module converts to DER for the cryptography library, then verifies
     against the Galileo GSC public key.

  2. MACK cross-authentication tag verification
     Each subframe can carry cross-auth tags that authenticate navigation data
     from other satellites.  The MAC input follows ICD §5.4.2.

  3. DecodedSubframe ↔ SubframeData adapter (parser ↔ engine bridge)
     INavAccumulator produces DecodedSubframe (parser format).
     INavOSNMAEngine consumes SubframeData (engine format).
     _adapt_subframe() bridges the two.

  4. OSNMAVerifier — unified orchestrator exposing two ingestion paths:
       process_subframe(DecodedSubframe, ...)  — direct subframe API
       add_page(OSNMAPage, ...)               — full raw-page pipeline

Protocol reference: Galileo OSNMA SIS ICD OS-SIS-ICD-OSNMA v1.1 (simplified).

Key representation note:
  INavOSNMAEngine uses ``gst_sf`` as total GST seconds from epoch
  (= wn * 604800 + tow), NOT the packed 32-bit WN||TOW integer from pack_gst().
  _adapt_subframe() uses gst_to_seconds_total() to convert correctly.
"""

from __future__ import annotations

from dataclasses import dataclass

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ec import ECDSA, EllipticCurvePublicKey
from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature
from cryptography.hazmat.primitives.hashes import SHA256
from cryptography.hazmat.primitives.serialization import load_pem_public_key

from gnss.osnma_inav import (
    NMA_STATUS_OPERATIONAL,
    NMA_STATUS_TEST,
    SUBFRAME_DURATION_S,
    TESLA_DELAY,
    HkrootMessage,
    INavOSNMAEngine,
    MackMessage,
    MackTagEntry,
    SubframeData,
    SubframeVerifyResult,
    compute_mac_tag,
)
from gnss.parser.hkroot_parser import ParsedHkroot, parse_dsm_kroot
from gnss.parser.inav_parser import DecodedSubframe, INavAccumulator, OSNMAPage
from gnss.parser.mack_parser import ParsedMack
from gnss.utils.gst_utils import gst_to_seconds_total

# ---------------------------------------------------------------------------
# DSM-KROOT layout constants
# ---------------------------------------------------------------------------

#: Bits in DSM-KROOT before the K_ROOT field.
#: nb_dk[4]+pkid[4]+cidx[4]+hf[4]+mf[4]+ks[4]+ts[4]+maclt[8]+rsvd[12]+wn_k[12]+tow_k[20]+alpha[48]
_DSM_KROOT_HEADER_BITS: int = 128

#: ECDSA-P256 raw signature size (r || s), 32 bytes each = 64 bytes.
_DS_ECDSA_P256_BYTES: int = 64


# ---------------------------------------------------------------------------
# 1. ECDSA-P256 K_ROOT digital signature verification
# ---------------------------------------------------------------------------


def verify_kroot_ds(
    dsm_payload: bytes,
    ds: bytes,
    pubkey_pem: bytes,
    key_size_bits: int = 128,
) -> bool:
    """Verify the ECDSA-P256 digital signature over the DSM-KROOT payload.

    Signed region:
        DSM-KROOT bytes from bit 0 up to (but not including) the DS field.
        Byte boundary = (_DSM_KROOT_HEADER_BITS + key_size_bits) // 8.

    The ``ds`` parameter is expected in the ICD raw format: r || s (64 bytes,
    32 bytes per integer, big-endian).  Internally converted to DER for the
    ``cryptography`` library.

    Args:
        dsm_payload:   126-byte assembled DSM-KROOT from DsmKroot.assembled_bytes().
        ds:            64-byte raw ECDSA-P256 signature (DS field from ParsedHkroot).
        pubkey_pem:    PEM-encoded ECDSA-P256 public key (from Galileo GSC PKI).
        key_size_bits: TESLA key size in bits (from ParsedHkroot.key_size_bits).

    Returns:
        True if the signature verifies, False on any mismatch or error.
    """
    try:
        signed_n = (_DSM_KROOT_HEADER_BITS + key_size_bits) // 8
        signed_data = dsm_payload[:signed_n]

        # Convert raw (r || s) → DER-encoded ECDSA signature
        half = len(ds) // 2
        r = int.from_bytes(ds[:half], "big")
        s = int.from_bytes(ds[half:], "big")
        der_sig = encode_dss_signature(r, s)

        pubkey = load_pem_public_key(pubkey_pem)
        if not isinstance(pubkey, EllipticCurvePublicKey):
            return False
        pubkey.verify(der_sig, signed_data, ECDSA(SHA256()))
        return True
    except (InvalidSignature, ValueError, TypeError):
        return False


# ---------------------------------------------------------------------------
# 2. Cross-authentication tag verification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CrossTagResult:
    """Result of one MACK cross-authentication tag verification.

    Attributes:
        prn_a:        Authenticated satellite SVID (1-36).
        adkd:         ADKD type (0=CED, 4=timing, 12=slow-MAC).
        cop:          Continuity-of-protection nibble (0-15).
        tag_valid:    True if the MAC tag matches recomputation.
        has_nav_data: True if nav_data for prn_a was available.
    """

    prn_a: int
    adkd: int
    cop: int
    tag_valid: bool
    has_nav_data: bool


def verify_cross_tags(
    mack: ParsedMack,
    tesla_key: bytes,
    gst_sf_auth: int,
    nma_status: int,
    nav_data_map: dict[int, bytes],
) -> list[CrossTagResult]:
    """Verify all MACK cross-authentication tags for one subframe.

    Each cross-auth tag authenticates navigation data from a different satellite
    (prn_a) using the same TESLA key as tag-0, but bound to prn_a's SVID.

    MAC input (ICD §5.4.2):
        prn_a[1B] || GST_sf_auth[4B,BE] || CTR[1B] || NMAS[1B] || nav_data_A

    where GST_sf_auth is the total-seconds GST of the *authenticated* (delayed)
    subframe, not the current subframe that discloses the TESLA key.

    Args:
        mack:          Decoded MACK section (from gnss.parser.mack_parser).
        tesla_key:     Disclosed TESLA key K_{key_id}.
        gst_sf_auth:   GST [total seconds] of the authenticated subframe
                       = current subframe GST − TESLA_DELAY × SUBFRAME_DURATION_S.
        nma_status:    NMA status from HKROOT.
        nav_data_map:  Mapping SVID → nav_data for SVIDs seen in the authenticated epoch.

    Returns:
        List[CrossTagResult], one per cross-auth entry in mack.cross_tags.
    """
    results: list[CrossTagResult] = []
    for entry in mack.cross_tags:
        nav_data = nav_data_map.get(entry.prn_a)
        has_nav = nav_data is not None
        tag_valid = False
        if has_nav and nav_data is not None:
            expected = compute_mac_tag(
                key=tesla_key,
                svid=entry.prn_a,
                gst_sf=gst_sf_auth,
                adkd=entry.adkd,
                cop=entry.cop,
                nma_status=nma_status,
                nav_data=nav_data,
            )
            tag_valid = entry.tag == expected
        results.append(
            CrossTagResult(
                prn_a=entry.prn_a,
                adkd=entry.adkd,
                cop=entry.cop,
                tag_valid=tag_valid,
                has_nav_data=has_nav,
            )
        )
    return results


# ---------------------------------------------------------------------------
# 3. DecodedSubframe → SubframeData adapter
# ---------------------------------------------------------------------------


def _adapt_subframe(
    sf: DecodedSubframe,
    dsm_kroot: ParsedHkroot,
    nav_data: bytes,
    recv_time_gst: float,
) -> SubframeData:
    """Convert a parser DecodedSubframe into the engine SubframeData format.

    Time representation:
        INavOSNMAEngine uses total-seconds-from-GST-epoch for all time values.
        gst_to_seconds_total(wn, tow) = wn * 604800 + tow.

    Args:
        sf:             Decoded subframe from INavAccumulator.
        dsm_kroot:      Fully assembled DSM-KROOT (for complete KROOT fields).
        nav_data:       Navigation data authenticated by tag-0 in this subframe.
        recv_time_gst:  Reception time in total GST seconds.

    Returns:
        SubframeData ready for INavOSNMAEngine.verify_subframe().
    """
    gst_sf_s = gst_to_seconds_total(sf.wn, sf.tow_sf)
    key_id = sf.subframe_idx - TESLA_DELAY

    hkroot_msg = HkrootMessage(
        nma_status=sf.hkroot_section.nma_status,
        chain_id=sf.hkroot_section.chain_id,
        nb_dk=dsm_kroot.nb_dk,
        pkid=dsm_kroot.pkid,
        kroot_wn=dsm_kroot.wn_k,
        kroot_tow=dsm_kroot.tow_k,
        alpha=dsm_kroot.alpha,
        kroot=dsm_kroot.kroot,
        ds=dsm_kroot.ds,
    )
    mack_msg = MackMessage(
        gst_sf=gst_sf_s,
        tag0=sf.mack.tag0.tag,
        tag0_adkd=sf.mack.tag0.adkd,
        tags=[MackTagEntry(adkd=t.adkd, cop=t.cop, tag=t.tag) for t in sf.mack.cross_tags],
        tesla_key=sf.mack.tesla_key,
        key_id=key_id,
    )
    return SubframeData(
        svid=sf.svid,
        subframe_idx=sf.subframe_idx,
        gst_sf=gst_sf_s,
        nav_data=nav_data,
        hkroot=hkroot_msg,
        mack=mack_msg,
        recv_time_gst=recv_time_gst,
    )


# ---------------------------------------------------------------------------
# 4. Unified verification report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OSNMAVerifyReport:
    """Complete OSNMA verification report for one (svid, subframe) pair.

    Combines the three-check INavOSNMAEngine result with ECDSA DS verification
    and cross-authentication tag results.

    Attributes:
        svid:           Galileo SVID (1-36).
        subframe_idx:   0-based subframe index from GST epoch.
        gst_sf:         GST at subframe start [total seconds].
        key_valid:      TESLA key lies on the authenticated hash chain.
        mac_valid:      Self-auth MAC tag-0 matches HMAC recomputation.
        receipt_safe:   Subframe was received before key-disclosure GST.
        nma_ok:         NMA status is within the acceptable set.
        ds_valid:       K_ROOT ECDSA-P256 signature verified.
                        None if no public key is configured.
        cross_tags:     Cross-authentication results (empty if no cross-tags).
        authenticated:  key_valid AND mac_valid AND receipt_safe AND nma_ok.
    """

    svid: int
    subframe_idx: int
    gst_sf: int
    key_valid: bool
    mac_valid: bool
    receipt_safe: bool
    nma_ok: bool
    ds_valid: bool | None
    cross_tags: tuple[CrossTagResult, ...]
    authenticated: bool


# ---------------------------------------------------------------------------
# 5. Top-level OSNMAVerifier
# ---------------------------------------------------------------------------


class OSNMAVerifier:
    """End-to-end Galileo OSNMA verifier.

    Two ingestion paths
    -------------------
    **Direct subframe path** (pre-decoded data or tests)::

        verifier = OSNMAVerifier(pubkey_pem=pem_bytes)
        verifier.set_kroot(dsm_kroot, kroot_idx)

        for sf, nav, t in zip(decoded_subframes, nav_datas, recv_times):
            report = verifier.process_subframe(sf, nav, t)
            if report is not None:
                print(report.authenticated)

    **Full raw-page path** (real receiver data)::

        verifier = OSNMAVerifier(pubkey_pem=pem_bytes)
        for page in receiver_pages:
            report = verifier.add_page(page, nav_data=..., recv_time_gst=...)
            if report is not None:
                print(report.authenticated)

    K_ROOT initialization
    ---------------------
    In the raw-page path, the engine is initialised automatically once all 14
    DSM-KROOT blocks have arrived (~7 minutes of satellite reception).
    For faster bootstrap, call set_kroot() with a pre-verified K_ROOT (e.g.
    from Galileo HAS or a cached PKI root).

    kroot_idx semantics
    -------------------
    kroot_idx is the TESLA chain index of K_ROOT.  It must be strictly greater
    than any subframe index that will be verified.  For real data:
        kroot_idx = (wn_k * 604800 + tow_k) // SUBFRAME_DURATION_S
    For simulated data (INavOSNMASimulator), pass the simulator's kroot_idx
    directly via set_kroot().

    Args:
        pubkey_pem:         PEM-encoded ECDSA-P256 public key for K_ROOT DS
                            verification.  None = skip DS check (ds_valid=None).
        key_size_bits:      TESLA key size in bits (default 128).
        tag_size_bits:      MAC tag size in bits (default 40).
        gst_start:          GST epoch for TESLA chain derivation [total seconds]
                            (default 0 = absolute GST epoch).
        nma_status_accept:  Acceptable NMA status values.
                            Defaults to {NMA_STATUS_TEST, NMA_STATUS_OPERATIONAL}.
    """

    def __init__(
        self,
        pubkey_pem: bytes | None = None,
        key_size_bits: int = 128,
        tag_size_bits: int = 40,
        gst_start: int = 0,
        nma_status_accept: frozenset[int] | None = None,
    ) -> None:
        self._pubkey_pem = pubkey_pem
        self._key_size_bits = key_size_bits
        self._tag_size_bits = tag_size_bits
        self._gst_start = gst_start
        self._nma_status_accept: frozenset[int] = (
            nma_status_accept
            if nma_status_accept is not None
            else frozenset({NMA_STATUS_TEST, NMA_STATUS_OPERATIONAL})
        )

        # Per-SVID accumulators (raw-page path)
        self._accumulators: dict[int, INavAccumulator] = {}

        # Core engine (initialised by set_kroot or first complete DSM-KROOT)
        self._engine: INavOSNMAEngine | None = None
        self._dsm_kroot: ParsedHkroot | None = None
        self._dsm_kroot_raw: bytes | None = None
        self._ds_valid_cached: bool | None = None  # cached — verified once per KROOT

        # Nav-data buffer: (svid, sf_idx) → nav_data (needed for cross-tag lookup)
        self._nav_data_buf: dict[tuple[int, int], bytes] = {}

    # ------------------------------------------------------------------
    # K_ROOT injection
    # ------------------------------------------------------------------

    def set_kroot(
        self,
        dsm_kroot: ParsedHkroot,
        kroot_idx: int,
        dsm_kroot_raw: bytes | None = None,
    ) -> None:
        """Inject a pre-verified DSM-KROOT and initialise the TESLA engine.

        Use when K_ROOT arrives out-of-band (Galileo HAS, cached PKI bootstrap,
        or in tests with INavOSNMASimulator.engine_params['kroot_idx']).

        Args:
            dsm_kroot:      Fully parsed DSM-KROOT message.
            kroot_idx:      Chain index of dsm_kroot.kroot.  Must be > all
                            subframe indices that will be verified.
            dsm_kroot_raw:  Optional 126-byte raw DSM payload for DS verification.
                            If None, ds_valid will be None in reports.
        """
        self._dsm_kroot = dsm_kroot
        self._dsm_kroot_raw = dsm_kroot_raw
        self._ds_valid_cached = None
        self._engine = INavOSNMAEngine(
            kroot=dsm_kroot.kroot,
            kroot_idx=kroot_idx,
            gst_start=self._gst_start,
            alpha=dsm_kroot.alpha,
            nma_status_accept=self._nma_status_accept,
        )

    # ------------------------------------------------------------------
    # Direct subframe path
    # ------------------------------------------------------------------

    def process_subframe(
        self,
        sf: DecodedSubframe,
        nav_data: bytes,
        recv_time_gst: float,
    ) -> OSNMAVerifyReport | None:
        """Verify a pre-assembled DecodedSubframe.

        Returns None if no K_ROOT has been configured yet.

        Args:
            sf:             Decoded subframe from INavAccumulator.
            nav_data:       Navigation data for (sf.svid, sf.subframe_idx).
            recv_time_gst:  Reception time in total GST seconds.

        Returns:
            OSNMAVerifyReport or None.
        """
        if self._engine is None or self._dsm_kroot is None:
            return None

        # Buffer nav_data for cross-tag lookup in subsequent subframes
        self._nav_data_buf[(sf.svid, sf.subframe_idx)] = nav_data

        sf_data = _adapt_subframe(sf, self._dsm_kroot, nav_data, recv_time_gst)
        core: SubframeVerifyResult = self._engine.verify_subframe(sf_data)

        # DS verification — done once and cached
        ds_valid = self._verify_ds_once()

        # Cross-authentication tags (requires disclosed TESLA key)
        cross_results: list[CrossTagResult] = []
        if core.key_valid and sf.mack.tesla_key is not None:
            sf_idx_auth = sf.subframe_idx - TESLA_DELAY
            gst_sf_auth = gst_to_seconds_total(sf.wn, sf.tow_sf) - TESLA_DELAY * SUBFRAME_DURATION_S
            nav_map: dict[int, bytes] = {
                svid: nd for (svid, idx), nd in self._nav_data_buf.items() if idx == sf_idx_auth
            }
            cross_results = verify_cross_tags(
                mack=sf.mack,
                tesla_key=sf.mack.tesla_key,
                gst_sf_auth=gst_sf_auth,
                nma_status=sf.hkroot_section.nma_status,
                nav_data_map=nav_map,
            )

        return OSNMAVerifyReport(
            svid=core.svid,
            subframe_idx=core.subframe_idx,
            gst_sf=core.gst_sf,
            key_valid=core.key_valid,
            mac_valid=core.mac_valid,
            receipt_safe=core.receipt_safe,
            nma_ok=core.nma_ok,
            ds_valid=ds_valid,
            cross_tags=tuple(cross_results),
            authenticated=core.authenticated,
        )

    # ------------------------------------------------------------------
    # Raw-page path
    # ------------------------------------------------------------------

    def add_page(
        self,
        page: OSNMAPage,
        nav_data: bytes,
        recv_time_gst: float,
    ) -> OSNMAVerifyReport | None:
        """Feed one I/NAV OSNMA page; return a report when a subframe completes.

        Internally uses INavAccumulator to assemble 15 pages into a DecodedSubframe,
        then calls process_subframe().

        Returns None until:
          - all 15 pages of a subframe have arrived, AND
          - DSM-KROOT is complete (or set_kroot() was called in advance).

        Args:
            page:           One I/NAV OSNMA page (5 bytes, 40 bits of OSNMA data).
            nav_data:       Navigation data for (page.svid, current subframe).
            recv_time_gst:  Reception time in total GST seconds.
        """
        acc = self._get_accumulator(page.svid)
        decoded = acc.add_page(page)
        if decoded is None:
            return None  # subframe not yet complete

        # Buffer nav_data now (before process_subframe, which also buffers it)
        self._nav_data_buf[(page.svid, decoded.subframe_idx)] = nav_data

        # First-time engine initialization from completed DSM-KROOT
        if self._engine is None:
            self._try_init_engine_from_dsm(acc)

        return self.process_subframe(decoded, nav_data, recv_time_gst)

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    def authenticated_svids(self, svids: list[int]) -> list[bool]:
        """Per-SVID authentication flags, compatible with OSNMALayer.assess().

        Args:
            svids: Ordered list of Galileo SVIDs.

        Returns:
            list[bool] of length len(svids).
        """
        if self._engine is None:
            return [False] * len(svids)
        return self._engine.authenticated_svids(svids)

    def reset(self) -> None:
        """Clear all state for reprocessing a new stream."""
        self._accumulators.clear()
        self._engine = None
        self._dsm_kroot = None
        self._dsm_kroot_raw = None
        self._ds_valid_cached = None
        self._nav_data_buf.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_accumulator(self, svid: int) -> INavAccumulator:
        if svid not in self._accumulators:
            self._accumulators[svid] = INavAccumulator(
                svid=svid,
                key_size_bits=self._key_size_bits,
                tag_size_bits=self._tag_size_bits,
            )
        return self._accumulators[svid]

    def _try_init_engine_from_dsm(self, acc: INavAccumulator) -> None:
        """Try to initialize engine from first complete DSM-KROOT in accumulator."""
        for dsm_builder in acc.completed_dsm().values():
            try:
                raw = dsm_builder.assembled_bytes()
                dsm_kroot = parse_dsm_kroot(raw)
                kroot_gst_s = gst_to_seconds_total(dsm_kroot.wn_k, dsm_kroot.tow_k)
                kroot_idx = kroot_gst_s // SUBFRAME_DURATION_S
                self._dsm_kroot = dsm_kroot
                self._dsm_kroot_raw = raw
                self._ds_valid_cached = None
                self._engine = INavOSNMAEngine(
                    kroot=dsm_kroot.kroot,
                    kroot_idx=kroot_idx,
                    gst_start=self._gst_start,
                    alpha=dsm_kroot.alpha,
                    nma_status_accept=self._nma_status_accept,
                )
                return  # use first complete DSM-KROOT found
            except (RuntimeError, ValueError):
                continue

    def _verify_ds_once(self) -> bool | None:
        """Verify K_ROOT DS once and cache the result."""
        if self._pubkey_pem is None:
            return None
        if self._ds_valid_cached is not None:
            return self._ds_valid_cached
        if self._dsm_kroot is None or self._dsm_kroot_raw is None:
            return None
        self._ds_valid_cached = verify_kroot_ds(
            dsm_payload=self._dsm_kroot_raw,
            ds=self._dsm_kroot.ds,
            pubkey_pem=self._pubkey_pem,
            key_size_bits=self._dsm_kroot.key_size_bits,
        )
        return self._ds_valid_cached
