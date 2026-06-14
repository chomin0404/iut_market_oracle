"""Tests for gnss.chain_verifier.

Covers:
  1. TESLAChain — key_index, verify_key (正常・リプレイ・改ざん)
  2. compute_chain_mac_tag — 入力長・決定論性・切り捨て
  3. generate_tesla_chain — チェーン長・F の一貫性
  4. OSNMAChainVerifier.receive — 3チェック (key / receipt / mac) を独立に検証
  5. OSNMAChainVerifier.authenticated_svids — 累積フラグ
  6. OSNMAChainVerifier.reset — 状態クリア
"""

from __future__ import annotations

import hashlib
import hmac
import os
import struct

import pytest

from core.data_structures import DSMKROOTMessage, HashFunction, MACFunction
from gnss.chain_verifier import (
    ADKD_INAV_CED,
    NMA_STATUS_OPERATIONAL,
    SUBFRAME_DURATION_S,
    TESLA_DELAY,
    ChainAuthResult,
    NavAuthMessage,
    OSNMAChainVerifier,
    TESLAChain,
    compute_chain_mac_tag,
    generate_tesla_chain,
)

# ---------------------------------------------------------------------------
# Fixtures / shared helpers
# ---------------------------------------------------------------------------

_ALPHA: bytes = bytes(range(6))  # deterministic 6-byte nonce
_CHAIN_ID: int = 1
_KEY_BYTES: int = 16  # 128-bit keys
_TAG_BITS: int = 40  # 5-byte tags
_GST0: int = 0  # chain starts at GST = 0

_TEMPLATE = DSMKROOTMessage(
    cidkr=_CHAIN_ID,
    hash_func=HashFunction.SHA_256,
    mac_func=MACFunction.HMAC_SHA_256,
    key_size_bytes=_KEY_BYTES,
    tag_size_bits=_TAG_BITS,
    gst0=_GST0,
    alpha=_ALPHA,
    kroot=b"\x00" * _KEY_BYTES,  # placeholder; replaced by generate_tesla_chain
    ds=b"\x00" * 64,
)

_N_SF = 8  # number of test subframes


def _build_chain(n: int = _N_SF, seed: int = 42) -> tuple[DSMKROOTMessage, list[bytes]]:
    """Return (kroot_msg, keys) where keys[i] = K_i."""
    return generate_tesla_chain(n=n, template=_TEMPLATE, seed=seed)


def _make_nav_data(svid: int, sf_idx: int) -> bytes:
    """Deterministic 32-byte nav data."""
    return hashlib.sha256(struct.pack(">II", svid, sf_idx)).digest()


def _make_msgs(
    svid: int,
    n_sf: int,
    keys: list[bytes],
    *,
    tamper_mac_at: int | None = None,
    tamper_key_at: int | None = None,
    late_recv_at: int | None = None,
) -> tuple[list[NavAuthMessage], list[float]]:
    """Build synthetic NavAuthMessage list and recv_times.

    key index for sf_idx: idx = sf_idx + 1
    MAC computed with K_{idx}; disclosed key = K_{idx - TESLA_DELAY} (None if idx < TESLA_DELAY).
    """
    msgs: list[NavAuthMessage] = []
    recv_times: list[float] = []

    for sf_idx in range(n_sf):
        gst_sf = _GST0 + sf_idx * SUBFRAME_DURATION_S
        key_idx = sf_idx + 1
        nav_data = _make_nav_data(svid, sf_idx)

        # MAC with current subframe's key K_{key_idx}
        mac_tag = compute_chain_mac_tag(
            key=keys[key_idx],
            svid=svid,
            gst_sf=gst_sf,
            adkd=ADKD_INAV_CED,
            cop=0,
            nma_status=NMA_STATUS_OPERATIONAL,
            nav_data=nav_data,
            tag_size_bits=_TAG_BITS,
        )
        if tamper_mac_at == sf_idx:
            mac_tag = os.urandom(_TAG_BITS // 8)

        # Disclose K_{key_idx - TESLA_DELAY} when key_idx >= TESLA_DELAY
        disclosed_idx = key_idx - TESLA_DELAY
        tesla_key: bytes | None = keys[disclosed_idx] if disclosed_idx >= 1 else None
        if tamper_key_at == sf_idx and tesla_key is not None:
            tesla_key = os.urandom(_KEY_BYTES)

        # Normal recv_time: gst_sf + 1 s (well within subframe window)
        if late_recv_at == sf_idx:
            recv_t = float(gst_sf + SUBFRAME_DURATION_S + 5)  # after next subframe
        else:
            recv_t = float(gst_sf) + 1.0

        msgs.append(
            NavAuthMessage(
                svid=svid,
                gst_sf=gst_sf,
                nav_data=nav_data,
                mac_tag=mac_tag,
                tesla_key=tesla_key,
            )
        )
        recv_times.append(recv_t)

    return msgs, recv_times


def _run_verifier(
    kroot_msg: DSMKROOTMessage,
    msgs: list[NavAuthMessage],
    recv_times: list[float],
) -> list[ChainAuthResult | None]:
    """Run OSNMAChainVerifier on a list of messages; return all results."""
    verifier = OSNMAChainVerifier(kroot_msg)
    return [verifier.receive(msg, t) for msg, t in zip(msgs, recv_times)]


# ---------------------------------------------------------------------------
# 1. TESLAChain
# ---------------------------------------------------------------------------


class TestTESLAChain:
    def setup_method(self) -> None:
        self.kroot_msg, self.keys = _build_chain()
        self.chain = TESLAChain(self.kroot_msg)

    def test_key_index_boundary(self) -> None:
        # sf_idx=0 → key_idx=1
        assert self.chain.key_index(_GST0) == 1
        assert self.chain.key_index(_GST0 + SUBFRAME_DURATION_S) == 2
        assert self.chain.key_index(_GST0 + 7 * SUBFRAME_DURATION_S) == 8

    def test_key_index_before_chain_raises(self) -> None:
        with pytest.raises(ValueError, match="chain start"):
            self.chain.key_index(_GST0 - 1)

    def test_verify_key_sequential(self) -> None:
        """Each key K_i verifies against the previous anchor K_{i-1}."""
        for i in range(1, _N_SF + 1):
            gst_sf = _GST0 + (i - 1) * SUBFRAME_DURATION_S
            assert self.chain.verify_key(self.keys[i], gst_sf), f"K_{i} failed"

    def test_verify_key_skip_steps(self) -> None:
        """Skip-ahead verification: K_5 verifies directly from K_0 anchor."""
        gst_sf = _GST0 + 4 * SUBFRAME_DURATION_S
        assert self.chain.verify_key(self.keys[5], gst_sf)

    def test_verify_key_wrong_key_fails(self) -> None:
        gst_sf = _GST0  # idx = 1
        bad_key = os.urandom(_KEY_BYTES)
        assert not self.chain.verify_key(bad_key, gst_sf)

    def test_verify_key_replay_rejected(self) -> None:
        """Once last_idx advances, lower-index keys are rejected."""
        # Verify K_3 first (advances last_idx to 3)
        self.chain.verify_key(self.keys[3], _GST0 + 2 * SUBFRAME_DURATION_S)
        # K_1 (idx=1) is now a replay
        assert not self.chain.verify_key(self.keys[1], _GST0)

    def test_get_key_after_verify(self) -> None:
        gst_sf = _GST0  # idx = 1
        self.chain.verify_key(self.keys[1], gst_sf)
        assert self.chain.get_key(1) == self.keys[1]

    def test_get_key_unverified_returns_none(self) -> None:
        assert self.chain.get_key(99) is None

    def test_kroot_always_in_verified(self) -> None:
        assert self.chain.get_key(0) == self.kroot_msg.kroot

    def test_key_at_gst_delegates_correctly(self) -> None:
        gst_sf = _GST0 + 2 * SUBFRAME_DURATION_S  # idx = 3
        self.chain.verify_key(self.keys[3], gst_sf)
        assert self.chain.key_at_gst(gst_sf) == self.keys[3]

    def test_key_at_gst_before_chain_returns_none(self) -> None:
        assert self.chain.key_at_gst(_GST0 - 30) is None

    def test_f_is_deterministic(self) -> None:
        """Same input to _F always produces the same output."""
        key = os.urandom(_KEY_BYTES)
        result_a = self.chain._F(key)
        result_b = self.chain._F(key)
        assert result_a == result_b

    def test_f_output_length(self) -> None:
        result = self.chain._F(os.urandom(_KEY_BYTES))
        assert len(result) == _KEY_BYTES


# ---------------------------------------------------------------------------
# 2. compute_chain_mac_tag
# ---------------------------------------------------------------------------


class TestComputeChainMacTag:
    def test_output_length(self) -> None:
        key = os.urandom(_KEY_BYTES)
        tag = compute_chain_mac_tag(
            key=key,
            svid=1,
            gst_sf=0,
            adkd=0,
            cop=0,
            nma_status=1,
            nav_data=b"\x00" * 32,
            tag_size_bits=40,
        )
        assert len(tag) == 5

    def test_deterministic(self) -> None:
        key = b"\xab" * _KEY_BYTES
        kwargs = dict(
            key=key,
            svid=3,
            gst_sf=60,
            adkd=0,
            cop=2,
            nma_status=1,
            nav_data=b"\xff" * 32,
            tag_size_bits=40,
        )
        assert compute_chain_mac_tag(**kwargs) == compute_chain_mac_tag(**kwargs)

    def test_different_svid_different_tag(self) -> None:
        key = b"\x11" * _KEY_BYTES
        t1 = compute_chain_mac_tag(
            key=key,
            svid=1,
            gst_sf=0,
            adkd=0,
            cop=0,
            nma_status=1,
            nav_data=b"\x00" * 32,
            tag_size_bits=40,
        )
        t2 = compute_chain_mac_tag(
            key=key,
            svid=2,
            gst_sf=0,
            adkd=0,
            cop=0,
            nma_status=1,
            nav_data=b"\x00" * 32,
            tag_size_bits=40,
        )
        assert t1 != t2

    def test_different_gst_different_tag(self) -> None:
        key = b"\x22" * _KEY_BYTES
        t1 = compute_chain_mac_tag(
            key=key,
            svid=1,
            gst_sf=0,
            adkd=0,
            cop=0,
            nma_status=1,
            nav_data=b"\x00" * 32,
            tag_size_bits=40,
        )
        t2 = compute_chain_mac_tag(
            key=key,
            svid=1,
            gst_sf=30,
            adkd=0,
            cop=0,
            nma_status=1,
            nav_data=b"\x00" * 32,
            tag_size_bits=40,
        )
        assert t1 != t2

    def test_tag_truncation_20bit(self) -> None:
        key = os.urandom(_KEY_BYTES)
        tag = compute_chain_mac_tag(
            key=key,
            svid=1,
            gst_sf=0,
            adkd=0,
            cop=0,
            nma_status=1,
            nav_data=b"\x00" * 32,
            tag_size_bits=20,
        )
        assert len(tag) == 20 // 8

    def test_mac_input_layout(self) -> None:
        """Verify MAC input matches ICD §5.4.1 layout manually."""
        key = b"\x55" * _KEY_BYTES
        svid, gst_sf, adkd, cop, nma_status = 7, 300, 0, 3, 1
        nav_data = b"\xab" * 32
        ctr = ((adkd & 0xF) << 4) | (cop & 0xF)
        manual_input = (
            struct.pack("B", svid)
            + struct.pack(">I", gst_sf)
            + struct.pack("B", ctr)
            + struct.pack("B", nma_status)
            + nav_data
        )
        expected = hmac.new(key, manual_input, hashlib.sha256).digest()[:5]
        result = compute_chain_mac_tag(
            key=key,
            svid=svid,
            gst_sf=gst_sf,
            adkd=adkd,
            cop=cop,
            nma_status=nma_status,
            nav_data=nav_data,
        )
        assert result == expected


# ---------------------------------------------------------------------------
# 3. generate_tesla_chain
# ---------------------------------------------------------------------------


class TestGenerateTeslaChain:
    def test_chain_length(self) -> None:
        _, keys = _build_chain(n=5)
        assert len(keys) == 6  # K_0 … K_5

    def test_kroot_is_k0(self) -> None:
        kroot_msg, keys = _build_chain()
        assert kroot_msg.kroot == keys[0]

    def test_f_consistency(self) -> None:
        """F(K_{i+1}) == K_i for all i."""
        kroot_msg, keys = _build_chain()
        chain = TESLAChain(kroot_msg)
        for i in range(_N_SF):
            computed = chain._F(keys[i + 1])
            assert computed == keys[i], f"F(K_{i + 1}) != K_{i}"

    def test_different_seeds_differ(self) -> None:
        _, keys_a = _build_chain(seed=1)
        _, keys_b = _build_chain(seed=2)
        assert keys_a[0] != keys_b[0]

    def test_template_fields_preserved(self) -> None:
        kroot_msg, _ = _build_chain()
        assert kroot_msg.cidkr == _TEMPLATE.cidkr
        assert kroot_msg.alpha == _TEMPLATE.alpha
        assert kroot_msg.key_size_bytes == _TEMPLATE.key_size_bytes


# ---------------------------------------------------------------------------
# 4. OSNMAChainVerifier — 3-check authentication
# ---------------------------------------------------------------------------


class TestOSNMAChainVerifier:
    def setup_method(self) -> None:
        self.kroot_msg, self.keys = _build_chain()
        self.svid = 5

    # -- Basic genuine authentication --

    def test_first_sf_returns_none(self) -> None:
        """sf_idx=0 has no prior key to disclose; receive returns None."""
        msgs, recv_times = _make_msgs(self.svid, 1, self.keys)
        results = _run_verifier(self.kroot_msg, msgs, recv_times)
        assert results[0] is None

    def test_genuine_authenticates_from_sf1(self) -> None:
        """sf_idx=1 discloses K_1, authenticating the MAC from sf_idx=0."""
        msgs, recv_times = _make_msgs(self.svid, _N_SF, self.keys)
        results = _run_verifier(self.kroot_msg, msgs, recv_times)

        # sf_idx=1 is first to produce a result
        assert results[1] is not None
        assert results[1].authenticated
        assert results[1].key_valid
        assert results[1].receipt_safe
        assert results[1].mac_valid

    def test_all_genuine_subframes_authenticate(self) -> None:
        msgs, recv_times = _make_msgs(self.svid, _N_SF, self.keys)
        results = _run_verifier(self.kroot_msg, msgs, recv_times)
        auth_results = [r for r in results if r is not None]
        assert all(r.authenticated for r in auth_results)

    # -- Check 1: key_valid --

    def test_wrong_key_fails_key_valid(self) -> None:
        """Tampered tesla_key at sf_idx=1 → key_valid=False."""
        msgs, recv_times = _make_msgs(self.svid, _N_SF, self.keys, tamper_key_at=1)
        results = _run_verifier(self.kroot_msg, msgs, recv_times)
        r = results[1]
        assert r is not None
        assert not r.key_valid
        assert not r.authenticated

    def test_wrong_key_leaves_mac_check_incomplete(self) -> None:
        """When key_valid=False, mac_valid must also be False (key unavailable)."""
        msgs, recv_times = _make_msgs(self.svid, _N_SF, self.keys, tamper_key_at=2)
        results = _run_verifier(self.kroot_msg, msgs, recv_times)
        r = results[2]
        assert r is not None
        assert not r.key_valid
        assert not r.mac_valid

    # -- Check 2: receipt_safe --

    def test_late_receipt_fails_receipt_safe(self) -> None:
        """sf_idx=0 received late (after sf_idx=1) → receipt_safe=False at sf_idx=1."""
        msgs, recv_times = _make_msgs(self.svid, _N_SF, self.keys, late_recv_at=0)
        results = _run_verifier(self.kroot_msg, msgs, recv_times)
        r = results[1]
        assert r is not None
        assert not r.receipt_safe
        assert not r.authenticated

    def test_late_receipt_key_still_valid(self) -> None:
        """Receipt-safety failure does not invalidate the key chain check."""
        msgs, recv_times = _make_msgs(self.svid, _N_SF, self.keys, late_recv_at=0)
        results = _run_verifier(self.kroot_msg, msgs, recv_times)
        r = results[1]
        assert r is not None
        assert r.key_valid  # key chain is independent of timing

    # -- Check 3: mac_valid --

    def test_tampered_mac_fails_mac_valid(self) -> None:
        """Random mac_tag at sf_idx=0 → mac_valid=False when verified at sf_idx=1."""
        msgs, recv_times = _make_msgs(self.svid, _N_SF, self.keys, tamper_mac_at=0)
        results = _run_verifier(self.kroot_msg, msgs, recv_times)
        r = results[1]
        assert r is not None
        assert not r.mac_valid
        assert not r.authenticated

    def test_tampered_mac_keeps_key_and_receipt_valid(self) -> None:
        msgs, recv_times = _make_msgs(self.svid, _N_SF, self.keys, tamper_mac_at=0)
        results = _run_verifier(self.kroot_msg, msgs, recv_times)
        r = results[1]
        assert r is not None
        assert r.key_valid
        assert r.receipt_safe

    # -- ChainAuthResult fields --

    def test_result_fields_populated(self) -> None:
        msgs, recv_times = _make_msgs(self.svid, _N_SF, self.keys)
        results = _run_verifier(self.kroot_msg, msgs, recv_times)
        r = results[1]  # first result
        assert r is not None
        assert r.svid == self.svid
        assert r.gst_sf_auth == _GST0  # authenticated sf_idx=0
        assert r.gst_sf_disclose == _GST0 + SUBFRAME_DURATION_S  # disclosed at sf_idx=1
        assert r.key_idx == 1  # K_1 was disclosed

    def test_result_key_idx_increments(self) -> None:
        msgs, recv_times = _make_msgs(self.svid, _N_SF, self.keys)
        results = _run_verifier(self.kroot_msg, msgs, recv_times)
        non_none = [r for r in results if r is not None]
        for i, r in enumerate(non_none):
            assert r.key_idx == i + 1

    # -- Multi-SVID --

    def test_multiple_svids_independent(self) -> None:
        """Different SVIDs are tracked independently."""
        _, keys_a = _build_chain(seed=10)
        _, keys_b = _build_chain(seed=20)
        kroot_msg, _ = _build_chain(seed=10)
        verifier = OSNMAChainVerifier(kroot_msg)

        msgs_a, recv_a = _make_msgs(svid=1, n_sf=_N_SF, keys=keys_a)
        msgs_b, recv_b = _make_msgs(svid=2, n_sf=_N_SF, keys=keys_b)

        # Interleave SVID=1 and SVID=2 messages
        for (ma, ta), (mb, tb) in zip(zip(msgs_a, recv_a), zip(msgs_b, recv_b)):
            verifier.receive(ma, ta)
            verifier.receive(mb, tb)

        # SVID=1 should authenticate (same chain); SVID=2 uses different keys → mac fails
        flags = verifier.authenticated_svids([1, 2])
        assert flags[0] is True  # SVID=1 uses same kroot_msg
        assert flags[1] is False  # SVID=2 has different keys → mac_valid=False


# ---------------------------------------------------------------------------
# 5. authenticated_svids
# ---------------------------------------------------------------------------


class TestAuthenticatedSvids:
    def test_empty_before_any_message(self) -> None:
        kroot_msg, _ = _build_chain()
        verifier = OSNMAChainVerifier(kroot_msg)
        assert verifier.authenticated_svids([1, 2, 3]) == [False, False, False]

    def test_true_after_authentication(self) -> None:
        kroot_msg, keys = _build_chain()
        msgs, recv_times = _make_msgs(svid=7, n_sf=_N_SF, keys=keys)
        results = _run_verifier(kroot_msg, msgs, recv_times)
        # Verify at least one result is authenticated
        assert any(r is not None and r.authenticated for r in results)
        # Create verifier again and replay to check authenticated_svids
        verifier = OSNMAChainVerifier(kroot_msg)
        for msg, t in zip(msgs, recv_times):
            verifier.receive(msg, t)
        assert verifier.authenticated_svids([7]) == [True]

    def test_false_for_unknown_svid(self) -> None:
        kroot_msg, keys = _build_chain()
        verifier = OSNMAChainVerifier(kroot_msg)
        msgs, recv_times = _make_msgs(svid=3, n_sf=_N_SF, keys=keys)
        for msg, t in zip(msgs, recv_times):
            verifier.receive(msg, t)
        assert verifier.authenticated_svids([99]) == [False]

    def test_monotone_non_decreasing(self) -> None:
        """Once True, authenticated_svids never reverts to False."""
        kroot_msg, keys = _build_chain()
        verifier = OSNMAChainVerifier(kroot_msg)
        msgs, recv_times = _make_msgs(svid=5, n_sf=_N_SF, keys=keys)
        for msg, t in zip(msgs, recv_times):
            verifier.receive(msg, t)
        # After full run, flag is True; it must stay True
        assert verifier.authenticated_svids([5]) == [True]


# ---------------------------------------------------------------------------
# 6. reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_clears_auth_state(self) -> None:
        kroot_msg, keys = _build_chain()
        verifier = OSNMAChainVerifier(kroot_msg)
        msgs, recv_times = _make_msgs(svid=4, n_sf=_N_SF, keys=keys)
        for msg, t in zip(msgs, recv_times):
            verifier.receive(msg, t)
        assert verifier.authenticated_svids([4]) == [True]
        verifier.reset()
        assert verifier.authenticated_svids([4]) == [False]

    def test_reset_clears_buffer(self) -> None:
        """After reset, receipt_safe fails for messages that were only in pre-reset buffer."""
        kroot_msg, keys = _build_chain()
        verifier = OSNMAChainVerifier(kroot_msg)
        msgs, recv_times = _make_msgs(svid=4, n_sf=_N_SF, keys=keys)
        # Deliver sf_idx=0 (buffered)
        verifier.receive(msgs[0], recv_times[0])
        verifier.reset()
        # sf_idx=1 discloses K_1 but sf_idx=0 is no longer in buffer
        result = verifier.receive(msgs[1], recv_times[1])
        assert result is not None
        assert not result.receipt_safe  # buffer was cleared
        assert not result.authenticated

    def test_reset_reinitialises_chain(self) -> None:
        """After reset, chain verifies keys from scratch (K_ROOT anchor restored)."""
        kroot_msg, keys = _build_chain()
        verifier = OSNMAChainVerifier(kroot_msg)
        # Advance chain to idx=5 via sequential verification
        msgs, recv_times = _make_msgs(svid=4, n_sf=6, keys=keys)
        for msg, t in zip(msgs, recv_times):
            verifier.receive(msg, t)
        verifier.reset()
        # After reset: chain last_idx should be 0 again (K_ROOT only)
        assert verifier._chain._last_idx == 0

    def test_authentication_works_after_reset(self) -> None:
        kroot_msg, keys = _build_chain()
        verifier = OSNMAChainVerifier(kroot_msg)
        msgs, recv_times = _make_msgs(svid=4, n_sf=_N_SF, keys=keys)
        # First run
        for msg, t in zip(msgs, recv_times):
            verifier.receive(msg, t)
        verifier.reset()
        # Second run — should authenticate again from scratch
        for msg, t in zip(msgs, recv_times):
            verifier.receive(msg, t)
        assert verifier.authenticated_svids([4]) == [True]


# ---------------------------------------------------------------------------
# 7. Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_sha3_256_chain_verify(self) -> None:
        """SHA3-256 hash function verifies correctly."""
        template_sha3 = DSMKROOTMessage(
            cidkr=0,
            hash_func=HashFunction.SHA3_256,
            mac_func=MACFunction.HMAC_SHA_256,
            key_size_bytes=16,
            tag_size_bits=40,
            gst0=0,
            alpha=_ALPHA,
            kroot=b"\x00" * 16,
            ds=b"\x00" * 64,
        )
        kroot_msg, keys = generate_tesla_chain(n=4, template=template_sha3, seed=7)
        verifier = OSNMAChainVerifier(kroot_msg)
        msgs, recv_times = _make_msgs(svid=1, n_sf=4, keys=keys)
        results = [verifier.receive(m, t) for m, t in zip(msgs, recv_times)]
        assert any(r is not None and r.authenticated for r in results)

    def test_gst0_nonzero(self) -> None:
        """Chain anchored at non-zero GST epoch authenticates correctly."""
        gst0 = 3600  # 1 hour
        template_nonzero = DSMKROOTMessage(
            cidkr=2,
            hash_func=HashFunction.SHA_256,
            mac_func=MACFunction.HMAC_SHA_256,
            key_size_bytes=16,
            tag_size_bits=40,
            gst0=gst0,
            alpha=_ALPHA,
            kroot=b"\x00" * 16,
            ds=b"\x00" * 64,
        )
        kroot_msg, keys = generate_tesla_chain(n=4, template=template_nonzero, seed=99)
        verifier = OSNMAChainVerifier(kroot_msg)
        svid = 3

        msgs: list[NavAuthMessage] = []
        recv_times: list[float] = []
        for sf_idx in range(4):
            gst_sf = gst0 + sf_idx * SUBFRAME_DURATION_S
            nav_data = _make_nav_data(svid, sf_idx)
            key_idx = sf_idx + 1
            mac_tag = compute_chain_mac_tag(
                key=keys[key_idx],
                svid=svid,
                gst_sf=gst_sf,
                adkd=ADKD_INAV_CED,
                cop=0,
                nma_status=NMA_STATUS_OPERATIONAL,
                nav_data=nav_data,
                tag_size_bits=40,
            )
            disclosed_idx = key_idx - TESLA_DELAY
            tesla_key = keys[disclosed_idx] if disclosed_idx >= 1 else None
            msgs.append(
                NavAuthMessage(
                    svid=svid,
                    gst_sf=gst_sf,
                    nav_data=nav_data,
                    mac_tag=mac_tag,
                    tesla_key=tesla_key,
                )
            )
            recv_times.append(float(gst_sf) + 1.0)

        results = [verifier.receive(m, t) for m, t in zip(msgs, recv_times)]
        assert any(r is not None and r.authenticated for r in results)

    def test_tesla_delay_2(self) -> None:
        """TESLA_DELAY=2 requires buffering over 2 subframes before authentication."""
        delay = 2
        kroot_msg, keys = _build_chain(n=_N_SF + delay)
        verifier = OSNMAChainVerifier(kroot_msg, tesla_delay=delay)

        msgs: list[NavAuthMessage] = []
        recv_times: list[float] = []
        svid = 9
        for sf_idx in range(_N_SF):
            gst_sf = _GST0 + sf_idx * SUBFRAME_DURATION_S
            nav_data = _make_nav_data(svid, sf_idx)
            key_idx = sf_idx + 1
            mac_tag = compute_chain_mac_tag(
                key=keys[key_idx],
                svid=svid,
                gst_sf=gst_sf,
                adkd=ADKD_INAV_CED,
                cop=0,
                nma_status=NMA_STATUS_OPERATIONAL,
                nav_data=nav_data,
                tag_size_bits=_TAG_BITS,
            )
            disclosed_idx = key_idx - delay
            tesla_key = keys[disclosed_idx] if disclosed_idx >= 1 else None
            msgs.append(
                NavAuthMessage(
                    svid=svid,
                    gst_sf=gst_sf,
                    nav_data=nav_data,
                    mac_tag=mac_tag,
                    tesla_key=tesla_key,
                )
            )
            recv_times.append(float(gst_sf) + 1.0)

        results = [verifier.receive(m, t) for m, t in zip(msgs, recv_times)]
        # First 2 subframes produce None (no prior key); rest should authenticate
        assert results[0] is None
        assert results[1] is None
        assert results[2] is not None
        assert results[2].authenticated

    def test_duplicate_receive_does_not_duplicate_auth(self) -> None:
        """Receiving the same message twice is idempotent (replay-safe)."""
        kroot_msg, keys = _build_chain()
        msgs, recv_times = _make_msgs(svid=6, n_sf=_N_SF, keys=keys)
        verifier = OSNMAChainVerifier(kroot_msg)
        # Deliver sf_idx=0 twice, then sf_idx=1
        verifier.receive(msgs[0], recv_times[0])
        verifier.receive(msgs[0], recv_times[0])  # duplicate
        result = verifier.receive(msgs[1], recv_times[1])
        # Should still authenticate (buffer just overwrites the same entry)
        assert result is not None
        assert result.authenticated
