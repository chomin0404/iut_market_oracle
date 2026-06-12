"""Tests for gnss.osnma_verifier.

Covers:
  1. verify_kroot_ds — ECDSA-P256 digital signature verification
  2. verify_cross_tags — MACK cross-authentication tag verification
  3. _adapt_subframe — DecodedSubframe ↔ SubframeData adapter
  4. OSNMAVerifier — end-to-end orchestrator (process_subframe + add_page paths)
"""

from __future__ import annotations

from cryptography.hazmat.primitives.asymmetric.ec import (
    ECDSA,
    SECP256R1,
    generate_private_key,
)
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature
from cryptography.hazmat.primitives.hashes import SHA256
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from gnss.osnma_inav import (
    ADKD_INAV_CED,
    NMA_STATUS_OPERATIONAL,
    SUBFRAME_DURATION_S,
    TESLA_DELAY,
    INavOSNMASimulator,
    _compute_mac_tag,
    make_inav_nav_data,
)
from gnss.osnma_verifier import (
    OSNMAVerifier,
    OSNMAVerifyReport,
    _adapt_subframe,
    verify_cross_tags,
    verify_kroot_ds,
)
from gnss.parser.hkroot_parser import HkrootSection, ParsedHkroot
from gnss.parser.inav_parser import DecodedSubframe, OSNMAPage
from gnss.parser.mack_parser import ParsedMack, TagEntry
from gnss.utils.gst_utils import gst_to_seconds_total

# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------


def _ecdsa_keygen():
    """Return (private_key, public_pem_bytes) for an ECDSA-P256 key pair."""
    priv = generate_private_key(SECP256R1())
    pub_pem = priv.public_key().public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)
    return priv, pub_pem


def _sign_raw(priv_key, data: bytes) -> bytes:
    """Sign ``data`` with ECDSA-P256; return 64-byte raw r‖s (ICD format)."""
    der_sig = priv_key.sign(data, ECDSA(SHA256()))
    r, s = decode_dss_signature(der_sig)
    return r.to_bytes(32, "big") + s.to_bytes(32, "big")


def _build_parsed_hkroot(
    sim: INavOSNMASimulator,
    ds_override: bytes | None = None,
) -> ParsedHkroot:
    """Build a minimal ParsedHkroot from simulator parameters."""
    return ParsedHkroot(
        nb_dk=2,
        pkid=0,
        cidx=0,
        hf=0,
        mf=0,
        ks=2,  # ks=2 → 128-bit key
        ts=4,  # ts=4 → 40-bit tag
        maclt=0,
        wn_k=0,
        tow_k=0,
        alpha=sim._alpha,  # noqa: SLF001
        kroot=sim._kroot,  # noqa: SLF001
        ds=ds_override if ds_override is not None else sim._ds,  # noqa: SLF001
        key_size_bits=128,
        tag_size_bits=40,
    )


def _build_decoded_subframe(
    sim: INavOSNMASimulator,
    svid: int,
    sf_idx: int,
    tamper_tag0: bool = False,
) -> tuple[DecodedSubframe, bytes]:
    """Build a DecodedSubframe + nav_data from simulator.

    Returns (decoded_subframe, nav_data).
    """
    sf = sim.make_subframe(svid, sf_idx, tamper_tag0=tamper_tag0)
    gst_sf = sf.gst_sf  # total seconds from GST epoch (gst_start=0 default)
    wn = gst_sf // 604800
    tow_sf = gst_sf % 604800

    hkroot_sec = HkrootSection(
        nma_status=sf.hkroot.nma_status,
        chain_id=sf.hkroot.chain_id,
        chain_in_force=False,
        cidx=0,
        cpks=0,
        dsm_id=0,
        dsm_block_id=0,
        dsm_data=b"\x00" * 9,
    )
    tag0_entry = TagEntry(tag=sf.mack.tag0, adkd=sf.mack.tag0_adkd, cop=0, prn_a=0)
    mack_parsed = ParsedMack(
        nma_status=sf.hkroot.nma_status,
        chain_id=sf.hkroot.chain_id,
        has_key=sf.mack.tesla_key is not None,
        tag0=tag0_entry,
        cross_tags=(),
        tesla_key=sf.mack.tesla_key,
    )
    decoded = DecodedSubframe(
        svid=svid,
        wn=wn,
        tow_sf=tow_sf,
        subframe_idx=sf_idx,
        hkroot_section=hkroot_sec,
        mack=mack_parsed,
        raw_600=b"\x00" * 75,
    )
    return decoded, sf.nav_data


def _pack_msb(fields: list[tuple[int, int]]) -> bytes:
    """Pack (value, n_bits) pairs MSB-first into bytes."""
    total_bits = sum(n for _, n in fields)
    acc = 0
    for val, nbits in fields:
        acc = (acc << nbits) | (val & ((1 << nbits) - 1))
    return acc.to_bytes((total_bits + 7) // 8, "big")


def _encode_raw600(
    nma_status: int,
    chain_id: int,
    tag0: bytes,
    adkd0: int,
    cop0: int,
    tesla_key: bytes | None,
    dsm_id: int = 0,
    dsm_block_id: int = 0,
    dsm_data: bytes = b"\x00" * 9,
) -> bytes:
    """Encode a 75-byte OSNMA block for INavAccumulator ingestion.

    Bit layout follows Galileo OSNMA SIS ICD:
      HKROOT (104 bits): NMA_H[16] + TESLA_H[8] + DSM_H[8] + DSM_DATA[72]
      MACK   (496 bits): HF[12] + tag0[40] + tag0_info[8] + fill[308] + key[128]
    """
    # HKROOT section
    hkroot_fields: list[tuple[int, int]] = [
        (nma_status, 2),
        (chain_id, 2),
        (0, 1),  # chain_in_force
        (0, 11),  # reserved
        (0, 4),  # cidx
        (0, 4),  # cpks
        (dsm_id, 4),
        (dsm_block_id, 4),
        (int.from_bytes(dsm_data, "big"), 72),
    ]
    hkroot_bytes = _pack_msb(hkroot_fields)  # 13 bytes

    # MACK section
    has_key = 1 if tesla_key is not None else 0
    tag0_int = int.from_bytes(tag0, "big")
    # key_start_bit = MACK_BITS(496) − key_size_bits(128) = 368
    # bits consumed: HF(12) + tag0(40) + tag0_info(8) = 60
    # fill: 368 − 60 = 308 bits (zero-filled cross-tag region)
    mack_fields: list[tuple[int, int]] = [
        (nma_status, 2),
        (chain_id, 2),
        (has_key, 1),
        (0, 7),  # reserved
        (tag0_int, 40),
        (adkd0, 4),
        (cop0, 4),
        (0, 308),  # zero cross-tag region
        (int.from_bytes(tesla_key, "big") if tesla_key is not None else 0, 128),
    ]
    mack_bytes = _pack_msb(mack_fields)  # 62 bytes

    raw_600 = hkroot_bytes + mack_bytes
    assert len(raw_600) == 75, f"Expected 75 bytes, got {len(raw_600)}"
    return raw_600


# ---------------------------------------------------------------------------
# 1. verify_kroot_ds
# ---------------------------------------------------------------------------


class TestVerifyKrootDs:
    def test_valid_signature(self):
        priv, pub_pem = _ecdsa_keygen()
        dsm_payload = bytes(range(126))
        key_size_bits = 128
        signed_n = (128 + key_size_bits) // 8  # 32 bytes
        ds = _sign_raw(priv, dsm_payload[:signed_n])
        assert verify_kroot_ds(dsm_payload, ds, pub_pem, key_size_bits) is True

    def test_wrong_payload_rejected(self):
        priv, pub_pem = _ecdsa_keygen()
        dsm_payload = bytes(range(126))
        signed_n = (128 + 128) // 8
        ds = _sign_raw(priv, dsm_payload[:signed_n])
        bad_payload = bytearray(dsm_payload)
        bad_payload[0] ^= 0xFF
        assert verify_kroot_ds(bytes(bad_payload), ds, pub_pem, 128) is False

    def test_wrong_signature_rejected(self):
        priv, pub_pem = _ecdsa_keygen()
        dsm_payload = bytes(range(126))
        ds = _sign_raw(priv, dsm_payload[:32])
        bad_ds = bytearray(ds)
        bad_ds[31] ^= 0xFF  # corrupt r component
        assert verify_kroot_ds(dsm_payload, bytes(bad_ds), pub_pem, 128) is False

    def test_wrong_pubkey_rejected(self):
        priv, _ = _ecdsa_keygen()
        _, wrong_pub_pem = _ecdsa_keygen()
        dsm_payload = bytes(range(126))
        ds = _sign_raw(priv, dsm_payload[:32])
        assert verify_kroot_ds(dsm_payload, ds, wrong_pub_pem, 128) is False

    def test_key_size_bits_96(self):
        """key_size_bits=96 → signed region = (128+96)//8 = 28 bytes."""
        priv, pub_pem = _ecdsa_keygen()
        dsm_payload = bytes(range(126))
        key_size_bits = 96
        signed_n = (128 + key_size_bits) // 8  # 28 bytes
        ds = _sign_raw(priv, dsm_payload[:signed_n])
        assert verify_kroot_ds(dsm_payload, ds, pub_pem, key_size_bits) is True

    def test_all_zero_signature_returns_false(self):
        _, pub_pem = _ecdsa_keygen()
        assert verify_kroot_ds(b"\x00" * 126, b"\x00" * 64, pub_pem, 128) is False

    def test_key_size_bits_256(self):
        """key_size_bits=256 → signed region = (128+256)//8 = 48 bytes."""
        priv, pub_pem = _ecdsa_keygen()
        dsm_payload = bytes(range(126))
        key_size_bits = 256
        signed_n = (128 + key_size_bits) // 8  # 48 bytes
        ds = _sign_raw(priv, dsm_payload[:signed_n])
        assert verify_kroot_ds(dsm_payload, ds, pub_pem, key_size_bits) is True


# ---------------------------------------------------------------------------
# 2. verify_cross_tags
# ---------------------------------------------------------------------------


class TestVerifyCrossTags:
    def setup_method(self):
        import os

        self.tesla_key = os.urandom(16)
        self.gst_sf_auth = 60  # total seconds from GST epoch
        self.nma_status = NMA_STATUS_OPERATIONAL
        self.nav_data_svid2 = make_inav_nav_data(2, 2)
        self.cross_tag = _compute_mac_tag(
            key=self.tesla_key,
            svid=2,
            gst_sf=self.gst_sf_auth,
            adkd=ADKD_INAV_CED,
            cop=0,
            nma_status=self.nma_status,
            nav_data=self.nav_data_svid2,
        )

    def _mack_with_cross(self, tag_override: bytes | None = None) -> ParsedMack:
        tag = tag_override if tag_override is not None else self.cross_tag
        tag0 = TagEntry(tag=b"\x00" * 5, adkd=0, cop=0, prn_a=0)
        entry = TagEntry(tag=tag, adkd=ADKD_INAV_CED, cop=0, prn_a=2)
        return ParsedMack(
            nma_status=self.nma_status,
            chain_id=0,
            has_key=True,
            tag0=tag0,
            cross_tags=(entry,),
            tesla_key=self.tesla_key,
        )

    def test_valid_cross_tag(self):
        results = verify_cross_tags(
            mack=self._mack_with_cross(),
            tesla_key=self.tesla_key,
            gst_sf_auth=self.gst_sf_auth,
            nma_status=self.nma_status,
            nav_data_map={2: self.nav_data_svid2},
        )
        assert len(results) == 1
        r = results[0]
        assert r.prn_a == 2
        assert r.adkd == ADKD_INAV_CED
        assert r.has_nav_data is True
        assert r.tag_valid is True

    def test_wrong_nav_data(self):
        results = verify_cross_tags(
            mack=self._mack_with_cross(),
            tesla_key=self.tesla_key,
            gst_sf_auth=self.gst_sf_auth,
            nma_status=self.nma_status,
            nav_data_map={2: b"\xff" * 32},
        )
        assert results[0].has_nav_data is True
        assert results[0].tag_valid is False

    def test_missing_nav_data(self):
        results = verify_cross_tags(
            mack=self._mack_with_cross(),
            tesla_key=self.tesla_key,
            gst_sf_auth=self.gst_sf_auth,
            nma_status=self.nma_status,
            nav_data_map={},
        )
        assert results[0].has_nav_data is False
        assert results[0].tag_valid is False

    def test_tampered_tag(self):
        bad_tag = bytes(b ^ 0xFF for b in self.cross_tag)
        results = verify_cross_tags(
            mack=self._mack_with_cross(tag_override=bad_tag),
            tesla_key=self.tesla_key,
            gst_sf_auth=self.gst_sf_auth,
            nma_status=self.nma_status,
            nav_data_map={2: self.nav_data_svid2},
        )
        assert results[0].tag_valid is False

    def test_no_cross_tags_returns_empty(self):
        tag0 = TagEntry(tag=b"\x00" * 5, adkd=0, cop=0, prn_a=0)
        mack = ParsedMack(
            nma_status=self.nma_status,
            chain_id=0,
            has_key=True,
            tag0=tag0,
            cross_tags=(),
            tesla_key=self.tesla_key,
        )
        results = verify_cross_tags(mack, self.tesla_key, self.gst_sf_auth, self.nma_status, {})
        assert results == []

    def test_multiple_cross_tags(self):
        nav_data_3 = make_inav_nav_data(3, 2)
        tag_3 = _compute_mac_tag(
            key=self.tesla_key,
            svid=3,
            gst_sf=self.gst_sf_auth,
            adkd=ADKD_INAV_CED,
            cop=0,
            nma_status=self.nma_status,
            nav_data=nav_data_3,
        )
        tag0 = TagEntry(tag=b"\x00" * 5, adkd=0, cop=0, prn_a=0)
        mack = ParsedMack(
            nma_status=self.nma_status,
            chain_id=0,
            has_key=True,
            tag0=tag0,
            cross_tags=(
                TagEntry(tag=self.cross_tag, adkd=ADKD_INAV_CED, cop=0, prn_a=2),
                TagEntry(tag=tag_3, adkd=ADKD_INAV_CED, cop=0, prn_a=3),
            ),
            tesla_key=self.tesla_key,
        )
        results = verify_cross_tags(
            mack,
            self.tesla_key,
            self.gst_sf_auth,
            self.nma_status,
            {2: self.nav_data_svid2, 3: nav_data_3},
        )
        assert len(results) == 2
        assert all(r.tag_valid for r in results)

    def test_wrong_tesla_key_invalidates_tag(self):
        """Using a different key produces a tag mismatch."""
        import os

        wrong_key = os.urandom(16)
        results = verify_cross_tags(
            mack=self._mack_with_cross(),
            tesla_key=wrong_key,  # different from self.tesla_key
            gst_sf_auth=self.gst_sf_auth,
            nma_status=self.nma_status,
            nav_data_map={2: self.nav_data_svid2},
        )
        assert results[0].tag_valid is False

    def test_wrong_gst_sf_auth_invalidates_tag(self):
        results = verify_cross_tags(
            mack=self._mack_with_cross(),
            tesla_key=self.tesla_key,
            gst_sf_auth=self.gst_sf_auth + 30,  # shifted by one subframe
            nma_status=self.nma_status,
            nav_data_map={2: self.nav_data_svid2},
        )
        assert results[0].tag_valid is False


# ---------------------------------------------------------------------------
# 3. _adapt_subframe
# ---------------------------------------------------------------------------


class TestAdaptSubframe:
    def test_gst_conversion_total_seconds(self):
        """gst_sf in SubframeData must equal gst_to_seconds_total(wn, tow_sf)."""
        sim = INavOSNMASimulator(svids=[1], n_subframes=5)
        dsm_kroot = _build_parsed_hkroot(sim)
        sf, nav_data = _build_decoded_subframe(sim, 1, sf_idx=3)
        recv_time = float(gst_to_seconds_total(sf.wn, sf.tow_sf) + 1)

        sf_data = _adapt_subframe(sf, dsm_kroot, nav_data, recv_time)

        expected_gst = gst_to_seconds_total(sf.wn, sf.tow_sf)
        assert sf_data.gst_sf == expected_gst

    def test_key_id_is_sf_idx_minus_tesla_delay(self):
        sim = INavOSNMASimulator(svids=[1], n_subframes=5)
        dsm_kroot = _build_parsed_hkroot(sim)
        sf_idx = 3
        sf, nav_data = _build_decoded_subframe(sim, 1, sf_idx)
        sf_data = _adapt_subframe(sf, dsm_kroot, nav_data, 1.0)

        assert sf_data.mack.key_id == sf_idx - TESLA_DELAY

    def test_sf_idx_zero_has_no_key(self):
        """sf_idx=0 → key_id=−1 → tesla_key=None."""
        sim = INavOSNMASimulator(svids=[1], n_subframes=5)
        dsm_kroot = _build_parsed_hkroot(sim)
        sf, nav_data = _build_decoded_subframe(sim, 1, sf_idx=0)
        sf_data = _adapt_subframe(sf, dsm_kroot, nav_data, 1.0)

        assert sf_data.mack.key_id == -1
        assert sf_data.mack.tesla_key is None

    def test_fields_propagated_correctly(self):
        sim = INavOSNMASimulator(svids=[2], n_subframes=5)
        dsm_kroot = _build_parsed_hkroot(sim)
        sf_idx = 2
        sf, nav_data = _build_decoded_subframe(sim, 2, sf_idx)
        recv_time = float(gst_to_seconds_total(sf.wn, sf.tow_sf) + 1)

        sf_data = _adapt_subframe(sf, dsm_kroot, nav_data, recv_time)

        assert sf_data.svid == 2
        assert sf_data.subframe_idx == sf_idx
        assert sf_data.nav_data == nav_data
        assert sf_data.hkroot.alpha == dsm_kroot.alpha
        assert sf_data.hkroot.kroot == dsm_kroot.kroot
        assert sf_data.hkroot.nma_status == sf.hkroot_section.nma_status
        assert sf_data.mack.tesla_key == sf.mack.tesla_key
        assert sf_data.recv_time_gst == recv_time

    def test_kroot_fields_come_from_dsm_kroot(self):
        """HkrootMessage carries fields from dsm_kroot, not hkroot_section."""
        sim = INavOSNMASimulator(svids=[1], n_subframes=5)
        dsm_kroot = _build_parsed_hkroot(sim)
        sf, nav_data = _build_decoded_subframe(sim, 1, sf_idx=2)
        sf_data = _adapt_subframe(sf, dsm_kroot, nav_data, 1.0)

        assert sf_data.hkroot.nb_dk == dsm_kroot.nb_dk
        assert sf_data.hkroot.pkid == dsm_kroot.pkid
        assert sf_data.hkroot.kroot_wn == dsm_kroot.wn_k
        assert sf_data.hkroot.kroot_tow == dsm_kroot.tow_k
        assert sf_data.hkroot.ds == dsm_kroot.ds


# ---------------------------------------------------------------------------
# 4. OSNMAVerifier
# ---------------------------------------------------------------------------


class TestOSNMAVerifier:
    """Tests for OSNMAVerifier using INavOSNMASimulator as ground truth."""

    def _setup(
        self, n_subframes: int = 6
    ) -> tuple[OSNMAVerifier, INavOSNMASimulator, ParsedHkroot]:
        sim = INavOSNMASimulator(svids=[1, 2, 3], n_subframes=n_subframes)
        dsm_kroot = _build_parsed_hkroot(sim)
        kroot_idx = sim.engine_params["kroot_idx"]
        verifier = OSNMAVerifier()
        verifier.set_kroot(dsm_kroot, kroot_idx)
        return verifier, sim, dsm_kroot

    # ------------------------------------------------------------------
    # No K_ROOT guards
    # ------------------------------------------------------------------

    def test_no_kroot_returns_none(self):
        sim = INavOSNMASimulator(svids=[1], n_subframes=5)
        verifier = OSNMAVerifier()  # no set_kroot
        sf, nav_data = _build_decoded_subframe(sim, 1, 1)
        assert verifier.process_subframe(sf, nav_data, 1.0) is None

    def test_authenticated_svids_without_kroot(self):
        verifier = OSNMAVerifier()
        assert verifier.authenticated_svids([1, 2, 3]) == [False, False, False]

    # ------------------------------------------------------------------
    # sf_idx=0 — no disclosed key yet
    # ------------------------------------------------------------------

    def test_sf_idx_zero_not_authenticated(self):
        verifier, sim, _ = self._setup()
        sf, nav_data = _build_decoded_subframe(sim, 1, 0)
        result = verifier.process_subframe(sf, nav_data, recv_time_gst=0.5)
        assert result is not None
        assert not result.authenticated
        assert not result.key_valid

    # ------------------------------------------------------------------
    # Genuine authentication
    # ------------------------------------------------------------------

    def test_genuine_authenticated_after_key_disclosure(self):
        """sf_idx=0 buffered → sf_idx=1 discloses K_0 → authenticated=True."""
        verifier, sim, _ = self._setup()
        sf0, nav0 = _build_decoded_subframe(sim, 1, 0)
        verifier.process_subframe(sf0, nav0, recv_time_gst=0.5)

        sf1, nav1 = _build_decoded_subframe(sim, 1, 1)
        result = verifier.process_subframe(sf1, nav1, recv_time_gst=31.0)

        assert result is not None
        assert result.key_valid
        assert result.mac_valid
        assert result.receipt_safe
        assert result.nma_ok
        assert result.authenticated

    def test_multi_svid_all_authenticated(self):
        """Three SVIDs verified independently."""
        verifier, sim, _ = self._setup()
        for svid in [1, 2, 3]:
            sf0, nav0 = _build_decoded_subframe(sim, svid, 0)
            verifier.process_subframe(sf0, nav0, recv_time_gst=0.5)
            sf1, nav1 = _build_decoded_subframe(sim, svid, 1)
            result = verifier.process_subframe(sf1, nav1, recv_time_gst=31.0)
            assert result is not None
            assert result.authenticated, f"SVID {svid} not authenticated"

    def test_authenticated_svids_flags(self):
        verifier, sim, _ = self._setup()
        # Authenticate only svid=1
        for sf_idx in range(2):
            sf, nav = _build_decoded_subframe(sim, 1, sf_idx)
            verifier.process_subframe(sf, nav, recv_time_gst=float(sf_idx * 30 + 0.5))

        flags = verifier.authenticated_svids([1, 2])
        assert flags[0] is True  # svid=1 authenticated
        assert flags[1] is False  # svid=2 not submitted

    # ------------------------------------------------------------------
    # Attack scenarios
    # ------------------------------------------------------------------

    def test_tampered_tag0_mac_invalid(self):
        verifier, sim, _ = self._setup()
        sf0, nav0 = _build_decoded_subframe(sim, 1, 0, tamper_tag0=True)
        verifier.process_subframe(sf0, nav0, recv_time_gst=0.5)

        sf1, nav1 = _build_decoded_subframe(sim, 1, 1)
        result = verifier.process_subframe(sf1, nav1, recv_time_gst=31.0)

        assert result is not None
        assert result.key_valid  # key chain is valid
        assert not result.mac_valid  # tampered tag0
        assert not result.authenticated

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def test_reset_clears_engine(self):
        verifier, sim, _ = self._setup()
        for sf_idx in range(2):
            sf, nav = _build_decoded_subframe(sim, 1, sf_idx)
            verifier.process_subframe(sf, nav, float(sf_idx * 30 + 0.5))

        verifier.reset()
        sf2, nav2 = _build_decoded_subframe(sim, 1, 2)
        assert verifier.process_subframe(sf2, nav2, 61.0) is None

    def test_reset_clears_authenticated_svids(self):
        verifier, sim, _ = self._setup()
        for sf_idx in range(2):
            sf, nav = _build_decoded_subframe(sim, 1, sf_idx)
            verifier.process_subframe(sf, nav, float(sf_idx * 30 + 0.5))

        verifier.reset()
        assert verifier.authenticated_svids([1]) == [False]

    # ------------------------------------------------------------------
    # Digital signature (ds_valid)
    # ------------------------------------------------------------------

    def test_ds_valid_none_without_pubkey(self):
        verifier, sim, _ = self._setup()
        for sf_idx in range(2):
            sf, nav = _build_decoded_subframe(sim, 1, sf_idx)
            result = verifier.process_subframe(sf, nav, float(sf_idx * 30 + 0.5))

        assert result is not None  # type: ignore[union-attr]
        assert result.ds_valid is None

    def test_ds_valid_true_with_correct_pubkey(self):
        priv, pub_pem = _ecdsa_keygen()
        sim = INavOSNMASimulator(svids=[1], n_subframes=5)
        dsm_kroot = _build_parsed_hkroot(sim)

        # Build raw payload + real signature
        key_size_bits = 128
        raw_payload = b"\xab" * 126
        raw_sig = _sign_raw(priv, raw_payload[: (128 + key_size_bits) // 8])
        dsm_kroot_signed = ParsedHkroot(
            nb_dk=dsm_kroot.nb_dk,
            pkid=dsm_kroot.pkid,
            cidx=dsm_kroot.cidx,
            hf=dsm_kroot.hf,
            mf=dsm_kroot.mf,
            ks=dsm_kroot.ks,
            ts=dsm_kroot.ts,
            maclt=dsm_kroot.maclt,
            wn_k=dsm_kroot.wn_k,
            tow_k=dsm_kroot.tow_k,
            alpha=dsm_kroot.alpha,
            kroot=dsm_kroot.kroot,
            ds=raw_sig,
            key_size_bits=key_size_bits,
            tag_size_bits=dsm_kroot.tag_size_bits,
        )
        kroot_idx = sim.engine_params["kroot_idx"]
        verifier = OSNMAVerifier(pubkey_pem=pub_pem)
        verifier.set_kroot(dsm_kroot_signed, kroot_idx, dsm_kroot_raw=raw_payload)

        result = None
        for sf_idx in range(2):
            sf, nav = _build_decoded_subframe(sim, 1, sf_idx)
            result = verifier.process_subframe(sf, nav, float(sf_idx * 30 + 0.5))

        assert result is not None
        assert result.ds_valid is True

    def test_ds_valid_false_with_wrong_pubkey(self):
        priv, _ = _ecdsa_keygen()
        _, wrong_pub_pem = _ecdsa_keygen()
        sim = INavOSNMASimulator(svids=[1], n_subframes=5)
        dsm_kroot = _build_parsed_hkroot(sim)
        raw_payload = b"\xab" * 126
        raw_sig = _sign_raw(priv, raw_payload[:32])
        dsm_kroot_signed = ParsedHkroot(
            nb_dk=dsm_kroot.nb_dk,
            pkid=dsm_kroot.pkid,
            cidx=dsm_kroot.cidx,
            hf=dsm_kroot.hf,
            mf=dsm_kroot.mf,
            ks=dsm_kroot.ks,
            ts=dsm_kroot.ts,
            maclt=dsm_kroot.maclt,
            wn_k=dsm_kroot.wn_k,
            tow_k=dsm_kroot.tow_k,
            alpha=dsm_kroot.alpha,
            kroot=dsm_kroot.kroot,
            ds=raw_sig,
            key_size_bits=128,
            tag_size_bits=40,
        )
        verifier = OSNMAVerifier(pubkey_pem=wrong_pub_pem)
        verifier.set_kroot(
            dsm_kroot_signed, sim.engine_params["kroot_idx"], dsm_kroot_raw=raw_payload
        )

        result = None
        for sf_idx in range(2):
            sf, nav = _build_decoded_subframe(sim, 1, sf_idx)
            result = verifier.process_subframe(sf, nav, float(sf_idx * 30 + 0.5))

        assert result is not None
        assert result.ds_valid is False

    def test_ds_valid_cached_across_subframes(self):
        """DS is verified only once; subsequent subframes reuse the cached result."""
        priv, pub_pem = _ecdsa_keygen()
        sim = INavOSNMASimulator(svids=[1], n_subframes=8)
        dsm_kroot = _build_parsed_hkroot(sim)
        raw_payload = b"\xab" * 126
        raw_sig = _sign_raw(priv, raw_payload[:32])
        dsm_kroot_signed = ParsedHkroot(
            nb_dk=dsm_kroot.nb_dk,
            pkid=dsm_kroot.pkid,
            cidx=dsm_kroot.cidx,
            hf=dsm_kroot.hf,
            mf=dsm_kroot.mf,
            ks=dsm_kroot.ks,
            ts=dsm_kroot.ts,
            maclt=dsm_kroot.maclt,
            wn_k=dsm_kroot.wn_k,
            tow_k=dsm_kroot.tow_k,
            alpha=dsm_kroot.alpha,
            kroot=dsm_kroot.kroot,
            ds=raw_sig,
            key_size_bits=128,
            tag_size_bits=40,
        )
        verifier = OSNMAVerifier(pubkey_pem=pub_pem)
        verifier.set_kroot(
            dsm_kroot_signed, sim.engine_params["kroot_idx"], dsm_kroot_raw=raw_payload
        )

        ds_values = []
        for sf_idx in range(5):
            sf, nav = _build_decoded_subframe(sim, 1, sf_idx)
            result = verifier.process_subframe(sf, nav, float(sf_idx * 30 + 0.5))
            if result is not None and result.ds_valid is not None:
                ds_values.append(result.ds_valid)

        # All non-None ds_valid values must be consistent (caching doesn't flip)
        assert len(set(ds_values)) <= 1, f"ds_valid inconsistent: {ds_values}"

    # ------------------------------------------------------------------
    # OSNMAVerifyReport structure
    # ------------------------------------------------------------------

    def test_report_type_and_fields(self):
        verifier, sim, _ = self._setup()
        sf0, nav0 = _build_decoded_subframe(sim, 1, 0)
        verifier.process_subframe(sf0, nav0, 0.5)
        sf1, nav1 = _build_decoded_subframe(sim, 1, 1)
        result = verifier.process_subframe(sf1, nav1, 31.0)

        assert isinstance(result, OSNMAVerifyReport)
        assert result.svid == 1
        assert result.subframe_idx == 1
        assert isinstance(result.gst_sf, int)
        assert isinstance(result.cross_tags, tuple)
        assert result.cross_tags == ()  # no cross tags in simulator default output

    # ------------------------------------------------------------------
    # set_kroot — re-initialization
    # ------------------------------------------------------------------

    def test_set_kroot_reinitializes_engine(self):
        """Calling set_kroot() a second time starts a fresh chain."""
        sim1 = INavOSNMASimulator(svids=[1], n_subframes=5)
        sim2 = INavOSNMASimulator(svids=[1], n_subframes=5, seed=99)

        dsm1 = _build_parsed_hkroot(sim1)
        dsm2 = _build_parsed_hkroot(sim2)

        verifier = OSNMAVerifier()
        verifier.set_kroot(dsm1, sim1.engine_params["kroot_idx"])

        # Verify sf from sim1
        sf0, nav0 = _build_decoded_subframe(sim1, 1, 0)
        verifier.process_subframe(sf0, nav0, 0.5)
        sf1, nav1 = _build_decoded_subframe(sim1, 1, 1)
        r1 = verifier.process_subframe(sf1, nav1, 31.0)
        assert r1 is not None and r1.authenticated

        # Switch to sim2 — old sim1 subframes should fail
        verifier.set_kroot(dsm2, sim2.engine_params["kroot_idx"])
        sf0_2, nav0_2 = _build_decoded_subframe(sim2, 1, 0)
        verifier.process_subframe(sf0_2, nav0_2, 0.5)
        sf1_2, nav1_2 = _build_decoded_subframe(sim2, 1, 1)
        r2 = verifier.process_subframe(sf1_2, nav1_2, 31.0)
        assert r2 is not None and r2.authenticated


# ---------------------------------------------------------------------------
# 5. add_page path
# ---------------------------------------------------------------------------


class TestOSNMAVerifierAddPage:
    """Tests for OSNMAVerifier.add_page() (full raw-page ingestion path)."""

    def _make_pages(
        self,
        svid: int,
        sf_idx: int,
        raw_600: bytes,
        wn: int = 0,
    ) -> list[OSNMAPage]:
        """Construct 15 OSNMAPage objects from a 75-byte OSNMA block."""
        tow_sf = sf_idx * SUBFRAME_DURATION_S
        pages = []
        for page_idx in range(15):
            osnma_bits = raw_600[page_idx * 5 : page_idx * 5 + 5]
            pages.append(
                OSNMAPage(
                    svid=svid,
                    wn=wn,
                    tow=tow_sf + page_idx * 2,  # 2 s per page
                    page_idx=page_idx,
                    osnma_bits=osnma_bits,
                )
            )
        return pages

    def test_fewer_than_15_pages_returns_none(self):
        """INavAccumulator does not emit until all 15 pages are received."""
        sim = INavOSNMASimulator(svids=[1], n_subframes=3)
        dsm_kroot = _build_parsed_hkroot(sim)
        kroot_idx = sim.engine_params["kroot_idx"]
        verifier = OSNMAVerifier()
        verifier.set_kroot(dsm_kroot, kroot_idx)

        # Dummy raw_600 — just enough to build pages; we won't complete the subframe
        raw_600 = b"\x00" * 75
        pages = self._make_pages(1, 0, raw_600)[:14]  # one short
        nav_data = make_inav_nav_data(1, 0)

        for page in pages:
            result = verifier.add_page(page, nav_data, recv_time_gst=0.5)
            assert result is None, "Expected None before 15 pages"

    def test_15_pages_completes_subframe(self):
        """After 15 pages a report is returned (sf_idx=0: not authenticated)."""
        sim = INavOSNMASimulator(svids=[1], n_subframes=5)
        dsm_kroot = _build_parsed_hkroot(sim)
        kroot_idx = sim.engine_params["kroot_idx"]
        verifier = OSNMAVerifier()
        verifier.set_kroot(dsm_kroot, kroot_idx)

        # sf_idx=0: no key disclosed, tag0 is correct
        sf0_data = sim.make_subframe(1, sf_idx=0)
        raw_600_sf0 = _encode_raw600(
            nma_status=sf0_data.hkroot.nma_status,
            chain_id=sf0_data.hkroot.chain_id,
            tag0=sf0_data.mack.tag0,
            adkd0=sf0_data.mack.tag0_adkd,
            cop0=0,
            tesla_key=sf0_data.mack.tesla_key,  # None for sf_idx=0
        )
        nav_data = sf0_data.nav_data
        pages = self._make_pages(1, 0, raw_600_sf0)
        result = None
        for page in pages:
            result = verifier.add_page(page, nav_data, recv_time_gst=0.5)

        assert result is not None
        assert result.svid == 1
        assert result.subframe_idx == 0
        assert not result.authenticated  # no key yet

    def test_two_subframes_via_add_page_authenticated(self):
        """sf_idx=0 followed by sf_idx=1 through add_page → authenticated."""
        sim = INavOSNMASimulator(svids=[1], n_subframes=5)
        dsm_kroot = _build_parsed_hkroot(sim)
        kroot_idx = sim.engine_params["kroot_idx"]
        verifier = OSNMAVerifier()
        verifier.set_kroot(dsm_kroot, kroot_idx)

        # sf_idx=0
        sf0_data = sim.make_subframe(1, sf_idx=0)
        raw_600_0 = _encode_raw600(
            nma_status=sf0_data.hkroot.nma_status,
            chain_id=sf0_data.hkroot.chain_id,
            tag0=sf0_data.mack.tag0,
            adkd0=sf0_data.mack.tag0_adkd,
            cop0=0,
            tesla_key=None,  # sf_idx=0 has no key
        )
        for page in self._make_pages(1, 0, raw_600_0):
            verifier.add_page(page, sf0_data.nav_data, recv_time_gst=0.5)

        # sf_idx=1 — discloses K_0 (key_id=0)
        sf1_data = sim.make_subframe(1, sf_idx=1)
        raw_600_1 = _encode_raw600(
            nma_status=sf1_data.hkroot.nma_status,
            chain_id=sf1_data.hkroot.chain_id,
            tag0=sf1_data.mack.tag0,
            adkd0=sf1_data.mack.tag0_adkd,
            cop0=0,
            tesla_key=sf1_data.mack.tesla_key,
        )
        result = None
        for page in self._make_pages(1, 1, raw_600_1):
            result = verifier.add_page(page, sf1_data.nav_data, recv_time_gst=31.0)

        assert result is not None
        assert result.key_valid
        assert result.mac_valid
        assert result.authenticated
