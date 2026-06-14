"""OSNMA end-to-end integration tests.

Pipeline under test:
    ECDSAVerifier.verify_kroot()
        → generate_tesla_chain()
        → OSNMAChainVerifier (TESLA key + receipt-safety + MAC)
        → OSNMAMacEngine (MACK-section §5.6.3 MAC format)

These tests ensure the three independent modules cooperate correctly
over a realistic subframe sequence without mocking the cryptographic
primitives they use internally.
"""

from __future__ import annotations

import dataclasses

import pytest
from cryptography.hazmat.primitives.asymmetric.ec import (
    ECDSA,
    SECP256R1,
    generate_private_key,
)
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature
from cryptography.hazmat.primitives.hashes import SHA256

from core.data_structures import (
    ADKD,
    DSMKROOTMessage,
    ECDSAType,
    HashFunction,
    MACFunction,
)
from gnss.chain_verifier import (
    ADKD_INAV_CED,
    NMA_STATUS_OPERATIONAL,
    SUBFRAME_DURATION_S,
    TESLA_DELAY,
    NavAuthMessage,
    OSNMAChainVerifier,
    compute_chain_mac_tag,
    generate_tesla_chain,
)
from gnss.ecdsa_verifier import _SCALAR_BYTES, ECDSAVerifier
from gnss.mac_engine import OSNMAMacEngine

# ---------------------------------------------------------------------------
# Shared constants and helpers
# ---------------------------------------------------------------------------

_GST0: int = 345_600  # arbitrary chain epoch (GST seconds)
_ALPHA: bytes = bytes(range(6))
_CID: int = 2
_KEY_BYTES: int = 16
_TAG_BITS: int = 40
_N_SF: int = 6  # number of simulated subframes
_NAV_DATA: bytes = b"\xDE\xAD\xBE\xEF" * 4  # 16-byte dummy nav payload


def _make_signed_kroot(priv_key, body: bytes = b"\x00" * 40) -> tuple[DSMKROOTMessage, int]:
    """Build a DSMKROOTMessage with a valid ECDSA-P256 signature.

    Returns (kroot_msg_placeholder, nma_hdr_byte).
    The ``kroot`` field is a placeholder — callers must replace it via
    generate_tesla_chain() to obtain a properly derived K_ROOT.
    """
    nma_hdr_byte = 0x05
    m_kroot = bytes([nma_hdr_byte]) + body
    der_sig = priv_key.sign(m_kroot, ECDSA(SHA256()))
    r, s = decode_dss_signature(der_sig)
    scalar = _SCALAR_BYTES[ECDSAType.P256]
    raw_sig = r.to_bytes(scalar, "big") + s.to_bytes(scalar, "big")

    msg = DSMKROOTMessage(
        cidkr=_CID,
        hash_func=HashFunction.SHA_256,
        mac_func=MACFunction.HMAC_SHA_256,
        key_size_bytes=_KEY_BYTES,
        tag_size_bits=_TAG_BITS,
        gst0=_GST0,
        alpha=_ALPHA,
        kroot=b"\x00" * _KEY_BYTES,  # placeholder
        ds=raw_sig,
        m_kroot_body=body,
    )
    return msg, nma_hdr_byte


def _build_subframe_sequence(
    kroot_msg: DSMKROOTMessage,
    keys: list[bytes],
    svid: int = 1,
    n_sf: int = _N_SF,
    nav_data: bytes = _NAV_DATA,
) -> list[NavAuthMessage]:
    """Build a valid subframe sequence for one SVID.

    Subframe at sf_idx (gst_sf = gst0 + sf_idx * 30):
      key_index = (gst_sf - gst0) / 30 + 1 = sf_idx + 1
      mac_tag   = HMAC(K_{sf_idx+1}, ...)          (key for THIS subframe)
      tesla_key = K_{sf_idx} = keys[sf_idx]         (disclosed TESLA_DELAY=1 earlier)

    For sf_idx=0: gst_auth = gst0 - 30 < gst0 → verifier returns None (pre-epoch).
    For sf_idx=1: gst_auth = gst0 → verifier authenticates gst0 subframe with keys[1].
    """
    msgs: list[NavAuthMessage] = []
    for sf_idx in range(n_sf):
        gst_sf = kroot_msg.gst0 + sf_idx * SUBFRAME_DURATION_S
        mac_tag = compute_chain_mac_tag(
            key=keys[sf_idx + 1],
            svid=svid,
            gst_sf=gst_sf,
            adkd=ADKD_INAV_CED,
            cop=0,
            nma_status=NMA_STATUS_OPERATIONAL,
            nav_data=nav_data,
            tag_size_bits=kroot_msg.tag_size_bits,
        )
        # Always disclose keys[sf_idx] = K_{sf_idx}; for sf_idx=0 this is K_ROOT
        # but gst_auth = gst0 - 30 < gst0, so verifier returns None (pre-epoch guard).
        msgs.append(
            NavAuthMessage(
                svid=svid,
                gst_sf=gst_sf,
                nav_data=nav_data,
                mac_tag=mac_tag,
                tesla_key=keys[sf_idx],
                adkd=ADKD_INAV_CED,
                cop=0,
                nma_status=NMA_STATUS_OPERATIONAL,
            )
        )
    return msgs


# ---------------------------------------------------------------------------
# Module-scoped fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def p256_priv():
    return generate_private_key(SECP256R1())


@pytest.fixture(scope="module")
def signed_kroot_pair(p256_priv):
    """(placeholder_kroot_msg, nma_hdr_byte) — kroot field is a placeholder."""
    return _make_signed_kroot(p256_priv)


@pytest.fixture(scope="module")
def full_chain(signed_kroot_pair):
    """(kroot_msg_with_derived_K0, keys[0..N_SF]) from generate_tesla_chain."""
    placeholder, _ = signed_kroot_pair
    return generate_tesla_chain(n=_N_SF, template=placeholder, seed=99)


# ---------------------------------------------------------------------------
# 1. ECDSAVerifier.verify_kroot() → generate_tesla_chain integration
# ---------------------------------------------------------------------------


class TestECDSAToChainHandoff:
    """Verify that the DSMKROOTMessage produced by generate_tesla_chain retains
    all fields needed for ECDSAVerifier.verify_kroot() to succeed."""

    def test_ecdsa_verifier_accepts_kroot_with_derived_K0(
        self, p256_priv, signed_kroot_pair, full_chain
    ):
        """ECDSAVerifier succeeds on the kroot_msg whose K_ROOT was derived by
        generate_tesla_chain (K_ROOT field changed, ECDSA fields unchanged)."""
        _, nma_hdr_byte = signed_kroot_pair
        kroot_msg, _ = full_chain
        verifier = ECDSAVerifier(ECDSAType.P256)
        assert verifier.verify_kroot(kroot_msg, nma_hdr_byte, p256_priv.public_key())

    def test_wrong_key_rejected_after_chain_derivation(
        self, signed_kroot_pair, full_chain
    ):
        """A different ECDSA key pair cannot verify the same signature."""
        _, nma_hdr_byte = signed_kroot_pair
        kroot_msg, _ = full_chain
        other_priv = generate_private_key(SECP256R1())
        verifier = ECDSAVerifier(ECDSAType.P256)
        assert not verifier.verify_kroot(kroot_msg, nma_hdr_byte, other_priv.public_key())

    def test_tampered_nma_header_rejected(self, p256_priv, signed_kroot_pair, full_chain):
        """Different NMA header byte produces different M_KROOT → signature fails."""
        _, nma_hdr_byte = signed_kroot_pair
        kroot_msg, _ = full_chain
        verifier = ECDSAVerifier(ECDSAType.P256)
        wrong_hdr = (nma_hdr_byte ^ 0xFF) & 0xFF
        assert not verifier.verify_kroot(kroot_msg, wrong_hdr, p256_priv.public_key())


# ---------------------------------------------------------------------------
# 2. generate_tesla_chain → OSNMAChainVerifier end-to-end
# ---------------------------------------------------------------------------


class TestChainVerifierE2E:
    """Feed a realistic subframe sequence into OSNMAChainVerifier and confirm
    all three checks (key_valid, receipt_safe, mac_valid) pass."""

    def test_first_subframe_returns_none(self, full_chain):
        """No tesla_key in sf_idx=0 → receive() returns None."""
        kroot_msg, keys = full_chain
        msgs = _build_subframe_sequence(kroot_msg, keys)
        cv = OSNMAChainVerifier(kroot_msg)
        result = cv.receive(msgs[0], recv_time_gst=float(msgs[0].gst_sf))
        assert result is None

    def test_second_subframe_produces_authenticated_result(self, full_chain):
        """sf_idx=1 discloses K_0 which authenticates sf_idx=0."""
        kroot_msg, keys = full_chain
        msgs = _build_subframe_sequence(kroot_msg, keys)
        cv = OSNMAChainVerifier(kroot_msg)
        # Feed sf_idx=0 first (buffers it)
        cv.receive(msgs[0], recv_time_gst=float(msgs[0].gst_sf))
        # Feed sf_idx=1 — discloses K_0 = kroot
        result = cv.receive(msgs[1], recv_time_gst=float(msgs[1].gst_sf))
        assert result is not None
        assert result.key_valid
        assert result.receipt_safe
        assert result.mac_valid
        assert result.authenticated

    def test_full_sequence_all_authenticated(self, full_chain):
        """All subframes from sf_idx=1 onward authenticate successfully."""
        kroot_msg, keys = full_chain
        msgs = _build_subframe_sequence(kroot_msg, keys)
        cv = OSNMAChainVerifier(kroot_msg)
        authenticated_count = 0
        for i, msg in enumerate(msgs):
            result = cv.receive(msg, recv_time_gst=float(msg.gst_sf))
            if result is not None and result.authenticated:
                authenticated_count += 1
        # First subframe never authenticates (no prior key); rest should all pass
        assert authenticated_count == _N_SF - TESLA_DELAY

    def test_tampered_tesla_key_fails_key_valid(self, full_chain):
        """Replacing the disclosed K_0 with random bytes → key_valid=False."""
        kroot_msg, keys = full_chain
        msgs = _build_subframe_sequence(kroot_msg, keys)
        cv = OSNMAChainVerifier(kroot_msg)
        cv.receive(msgs[0], recv_time_gst=float(msgs[0].gst_sf))
        # Tamper the disclosed key in sf_idx=1
        bad_msg = NavAuthMessage(
            svid=msgs[1].svid,
            gst_sf=msgs[1].gst_sf,
            nav_data=msgs[1].nav_data,
            mac_tag=msgs[1].mac_tag,
            tesla_key=bytes([0xFF] * _KEY_BYTES),
            adkd=msgs[1].adkd,
            cop=msgs[1].cop,
            nma_status=msgs[1].nma_status,
        )
        result = cv.receive(bad_msg, recv_time_gst=float(bad_msg.gst_sf))
        assert result is not None
        assert not result.key_valid
        assert not result.authenticated

    def test_tampered_mac_tag_fails_mac_valid(self, full_chain):
        """Wrong MAC tag → mac_valid=False even when key is correct."""
        kroot_msg, keys = full_chain
        msgs = _build_subframe_sequence(kroot_msg, keys)
        cv = OSNMAChainVerifier(kroot_msg)
        # Buffer sf_idx=0 with a bad MAC tag
        bad_sf0 = NavAuthMessage(
            svid=msgs[0].svid,
            gst_sf=msgs[0].gst_sf,
            nav_data=msgs[0].nav_data,
            mac_tag=bytes([0x00] * (_TAG_BITS // 8)),
            tesla_key=None,
            adkd=msgs[0].adkd,
            cop=msgs[0].cop,
            nma_status=msgs[0].nma_status,
        )
        cv.receive(bad_sf0, recv_time_gst=float(bad_sf0.gst_sf))
        result = cv.receive(msgs[1], recv_time_gst=float(msgs[1].gst_sf))
        assert result is not None
        assert result.key_valid
        assert not result.mac_valid
        assert not result.authenticated

    def test_authenticated_svids_cumulative(self, full_chain):
        """Once authenticated, SVID flag stays True after reset-free replay."""
        kroot_msg, keys = full_chain
        msgs = _build_subframe_sequence(kroot_msg, keys, svid=5)
        cv = OSNMAChainVerifier(kroot_msg)
        for msg in msgs:
            cv.receive(msg, recv_time_gst=float(msg.gst_sf))
        flags = cv.authenticated_svids([5, 99])
        assert flags[0] is True
        assert flags[1] is False


# ---------------------------------------------------------------------------
# 3. OSNMAMacEngine + chain_verifier MAC cross-validation
# ---------------------------------------------------------------------------


class TestMacEngineAndChainVerifierConsistency:
    """OSNMAMacEngine (MACK §5.6.3 format) and compute_chain_mac_tag (§5.4.1
    format) use different MAC input layouts — they must NOT produce the same
    tag for the same inputs.  This guards against accidental merging of the
    two distinct protocols."""

    def test_mac_engine_compute_is_deterministic(self, full_chain):
        kroot_msg, keys = full_chain
        engine = OSNMAMacEngine(MACFunction.HMAC_SHA_256, tag_size_bits=_TAG_BITS)
        tag1 = engine.compute_tag(
            key=keys[1],
            nav_data=_NAV_DATA,
            prn_a=1,
            prn_d=1,
            gst_sf=_GST0 + SUBFRAME_DURATION_S,
            adkd=ADKD.INAV_CED,
            ctr=0,
        )
        tag2 = engine.compute_tag(
            key=keys[1],
            nav_data=_NAV_DATA,
            prn_a=1,
            prn_d=1,
            gst_sf=_GST0 + SUBFRAME_DURATION_S,
            adkd=ADKD.INAV_CED,
            ctr=0,
        )
        assert tag1 == tag2

    def test_mac_engine_verify_accepts_own_tag(self, full_chain):
        kroot_msg, keys = full_chain
        engine = OSNMAMacEngine(MACFunction.HMAC_SHA_256, tag_size_bits=_TAG_BITS)
        gst = _GST0 + SUBFRAME_DURATION_S
        tag = engine.compute_tag(
            key=keys[1], nav_data=_NAV_DATA, prn_a=2, prn_d=2,
            gst_sf=gst, adkd=ADKD.INAV_CED, ctr=0,
        )
        assert engine.verify_tag(
            received=tag, key=keys[1], nav_data=_NAV_DATA, prn_a=2, prn_d=2,
            gst_sf=gst, adkd=ADKD.INAV_CED, ctr=0,
        )

    def test_mac_engine_and_chain_mac_differ(self, full_chain):
        """The two MAC formats must produce different byte strings for the same key."""
        kroot_msg, keys = full_chain
        gst = _GST0 + SUBFRAME_DURATION_S
        engine = OSNMAMacEngine(MACFunction.HMAC_SHA_256, tag_size_bits=_TAG_BITS)
        engine_tag = engine.compute_tag(
            key=keys[1], nav_data=_NAV_DATA, prn_a=1, prn_d=1,
            gst_sf=gst, adkd=ADKD.INAV_CED, ctr=0,
        )
        chain_tag = compute_chain_mac_tag(
            key=keys[1], svid=1, gst_sf=gst,
            adkd=ADKD_INAV_CED, cop=0, nma_status=NMA_STATUS_OPERATIONAL,
            nav_data=_NAV_DATA, tag_size_bits=_TAG_BITS,
        )
        assert engine_tag != chain_tag, (
            "OSNMAMacEngine and compute_chain_mac_tag must use distinct MAC inputs"
        )

    def test_mac_engine_wrong_key_rejected(self, full_chain):
        kroot_msg, keys = full_chain
        engine = OSNMAMacEngine(MACFunction.HMAC_SHA_256, tag_size_bits=_TAG_BITS)
        gst = _GST0 + SUBFRAME_DURATION_S
        tag = engine.compute_tag(
            key=keys[1], nav_data=_NAV_DATA, prn_a=1, prn_d=1,
            gst_sf=gst, adkd=ADKD.INAV_CED, ctr=0,
        )
        assert not engine.verify_tag(
            received=tag, key=keys[2], nav_data=_NAV_DATA, prn_a=1, prn_d=1,
            gst_sf=gst, adkd=ADKD.INAV_CED, ctr=0,
        )


# ---------------------------------------------------------------------------
# 4. Full three-module pipeline
# ---------------------------------------------------------------------------


class TestFullOSNMAPipeline:
    """Smoke test: ECDSA verify → chain derive → TESLA verify → MAC verify."""

    def test_pipeline_authenticates_all_subframes(self, p256_priv, signed_kroot_pair):
        """Full pipeline from fresh ECDSA key through subframe authentication."""
        placeholder, nma_hdr_byte = signed_kroot_pair
        kroot_msg, keys = generate_tesla_chain(n=4, template=placeholder, seed=77)

        # Step 1 — ECDSA: confirm K_ROOT signature is valid
        ecdsa_verifier = ECDSAVerifier(ECDSAType.P256)
        assert ecdsa_verifier.verify_kroot(kroot_msg, nma_hdr_byte, p256_priv.public_key())

        # Step 2 — Chain verifier: feed subframes
        cv = OSNMAChainVerifier(kroot_msg)
        msgs = _build_subframe_sequence(kroot_msg, keys, n_sf=4)
        results = [cv.receive(msg, recv_time_gst=float(msg.gst_sf)) for msg in msgs]

        authenticated = [r for r in results if r is not None and r.authenticated]
        # 4 subframes, TESLA_DELAY=1 → 3 authenticated
        assert len(authenticated) == 4 - TESLA_DELAY

    def test_tampered_kroot_ds_prevents_chain_trust(self, p256_priv, signed_kroot_pair):
        """If the ECDSA signature check fails, a system should not trust the chain.
        The chain itself cannot detect the compromise — but the ECDSA gate should."""
        placeholder, nma_hdr_byte = signed_kroot_pair
        kroot_msg, _ = generate_tesla_chain(n=3, template=placeholder, seed=13)

        # Tamper the DS field
        bad_ds = bytes([0x00] * len(kroot_msg.ds))
        bad_kroot = dataclasses.replace(kroot_msg, ds=bad_ds)

        ecdsa_verifier = ECDSAVerifier(ECDSAType.P256)
        # Gate must reject
        assert not ecdsa_verifier.verify_kroot(
            bad_kroot, nma_hdr_byte, p256_priv.public_key()
        )
