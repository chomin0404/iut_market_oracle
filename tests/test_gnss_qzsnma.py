"""Tests for src/gnss/qzsnma.py — QZSS NMA authenticator.

Acceptance criteria:
    QZSNMAChain:
        - from_root_and_length builds correct chain length
        - key_at returns None for out-of-range indices
        - consecutive keys satisfy one-way hash property

    QZSNMAVerifier:
        - verify_tag returns authenticated=True on valid tag
        - verify_tag returns authenticated=False with wrong tag
        - verify_tag returns no_chain_registered for unknown PRN

    QZSNMALayer:
        - assess(None) returns auth_fraction=1.0, alert=False
        - assess([]) returns auth_fraction=1.0, alert=False
        - assess([True]*5) returns auth_fraction=1.0, alert=False
        - assess([True, False]*3) returns auth_fraction ≈ 0.5
        - assess([False]*5) returns alert=True
"""

from __future__ import annotations

import hashlib

import pytest

from gnss.qzsnma import (
    HASH_TRUNCATION_BYTES,
    KEY_SIZE_BYTES,
    QZSNMAChain,
    QZSNMALayer,
    QZSNMALayerResult,
    QZSNMAVerifier,
    _compute_mac_tag,
    _derive_parent_key,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ROOT_KEY = b"\x01" * KEY_SIZE_BYTES
_PRN = 193
_GST0 = 1_296_000  # arbitrary GPS time epoch


# ---------------------------------------------------------------------------
# Key derivation
# ---------------------------------------------------------------------------


class TestDeriveParentKey:
    def test_output_length(self) -> None:
        child = b"\xab" * KEY_SIZE_BYTES
        parent = _derive_parent_key(child)
        assert len(parent) == KEY_SIZE_BYTES

    def test_deterministic(self) -> None:
        child = b"\xcd" * KEY_SIZE_BYTES
        assert _derive_parent_key(child) == _derive_parent_key(child)

    def test_different_input_different_output(self) -> None:
        k1 = _derive_parent_key(b"\x00" * KEY_SIZE_BYTES)
        k2 = _derive_parent_key(b"\xff" * KEY_SIZE_BYTES)
        assert k1 != k2


class TestComputeMacTag:
    def test_output_length(self) -> None:
        tag = _compute_mac_tag(_ROOT_KEY, b"test data")
        assert len(tag) == HASH_TRUNCATION_BYTES

    def test_deterministic(self) -> None:
        msg = b"nav message bytes"
        tag1 = _compute_mac_tag(_ROOT_KEY, msg)
        tag2 = _compute_mac_tag(_ROOT_KEY, msg)
        assert tag1 == tag2

    def test_different_key_different_tag(self) -> None:
        msg = b"nav message bytes"
        k2 = b"\xff" * KEY_SIZE_BYTES
        assert _compute_mac_tag(_ROOT_KEY, msg) != _compute_mac_tag(k2, msg)


# ---------------------------------------------------------------------------
# QZSNMAChain
# ---------------------------------------------------------------------------


class TestQZSNMAChain:
    def test_chain_length(self) -> None:
        chain = QZSNMAChain.from_root_and_length(_PRN, _GST0, _ROOT_KEY, 10)
        assert len(chain) == 10

    def test_first_key_is_root(self) -> None:
        chain = QZSNMAChain.from_root_and_length(_PRN, _GST0, _ROOT_KEY, 5)
        assert chain.keys[0] == _ROOT_KEY

    def test_key_at_root(self) -> None:
        chain = QZSNMAChain.from_root_and_length(_PRN, _GST0, _ROOT_KEY, 5)
        assert chain.key_at(_GST0) == _ROOT_KEY

    def test_key_at_future(self) -> None:
        chain = QZSNMAChain.from_root_and_length(_PRN, _GST0, _ROOT_KEY, 5)
        # gst0 + 3 * 1.0 s = index 3
        assert chain.key_at(_GST0 + 3) == chain.keys[3]

    def test_key_at_out_of_range_returns_none(self) -> None:
        chain = QZSNMAChain.from_root_and_length(_PRN, _GST0, _ROOT_KEY, 5)
        assert chain.key_at(_GST0 + 10) is None

    def test_key_at_before_gst0_returns_none(self) -> None:
        chain = QZSNMAChain.from_root_and_length(_PRN, _GST0, _ROOT_KEY, 5)
        assert chain.key_at(_GST0 - 1) is None

    def test_one_way_hash_property(self) -> None:
        chain = QZSNMAChain.from_root_and_length(_PRN, _GST0, _ROOT_KEY, 5)
        for i in range(1, len(chain.keys)):
            # Consecutive keys must satisfy: key[i-1] == SHA256(key[i])[:16]
            # Actually the chain is built forward: key[i] = SHA256(key[i-1])[:16]
            expected = hashlib.sha256(chain.keys[i - 1]).digest()[:KEY_SIZE_BYTES]
            assert chain.keys[i] == expected

    def test_invalid_root_key_length(self) -> None:
        with pytest.raises(ValueError, match="root_key must be"):
            QZSNMAChain.from_root_and_length(_PRN, _GST0, b"\x00" * 8, 5)

    def test_invalid_chain_length_zero(self) -> None:
        with pytest.raises(ValueError, match="chain_length must be in"):
            QZSNMAChain.from_root_and_length(_PRN, _GST0, _ROOT_KEY, 0)


# ---------------------------------------------------------------------------
# QZSNMAVerifier
# ---------------------------------------------------------------------------


class TestQZSNMAVerifier:
    def _make_verifier_and_chain(self, chain_len: int = 10) -> tuple[QZSNMAVerifier, QZSNMAChain]:
        chain = QZSNMAChain.from_root_and_length(_PRN, _GST0, _ROOT_KEY, chain_len)
        verifier = QZSNMAVerifier()
        verifier.register_chain(chain)
        return verifier, chain

    def test_valid_tag(self) -> None:
        verifier, chain = self._make_verifier_and_chain()
        gst_auth = _GST0 + 2
        nav_data = b"fake galileo nav data"
        auth_key = chain.key_at(gst_auth)
        assert auth_key is not None
        tag = _compute_mac_tag(auth_key, nav_data)
        result = verifier.verify_tag(_PRN, gst_auth, nav_data, tag)
        assert result.authenticated is True
        assert result.reason == "ok"

    def test_wrong_tag_fails(self) -> None:
        verifier, chain = self._make_verifier_and_chain()
        gst_auth = _GST0 + 2
        nav_data = b"fake nav data"
        wrong_tag = b"\xff" * HASH_TRUNCATION_BYTES
        result = verifier.verify_tag(_PRN, gst_auth, nav_data, wrong_tag)
        assert result.authenticated is False
        assert result.reason == "tag_mismatch"

    def test_unknown_prn(self) -> None:
        verifier, _ = self._make_verifier_and_chain()
        result = verifier.verify_tag(999, _GST0, b"data", b"\x00" * HASH_TRUNCATION_BYTES)
        assert result.authenticated is False
        assert result.reason == "no_chain_registered"

    def test_key_index_out_of_range(self) -> None:
        verifier, _ = self._make_verifier_and_chain()
        gst_far_future = _GST0 + 9999
        result = verifier.verify_tag(_PRN, gst_far_future, b"data", b"\x00" * HASH_TRUNCATION_BYTES)
        assert result.authenticated is False
        assert result.reason == "key_index_out_of_range"


# ---------------------------------------------------------------------------
# QZSNMALayer
# ---------------------------------------------------------------------------


class TestQZSNMALayer:
    def test_none_returns_full_auth(self) -> None:
        layer = QZSNMALayer()
        res = layer.assess(None)
        assert isinstance(res, QZSNMALayerResult)
        assert res.auth_fraction == 1.0
        assert res.alert is False
        assert res.n_total == 0

    def test_empty_list_returns_full_auth(self) -> None:
        res = QZSNMALayer().assess([])
        assert res.auth_fraction == 1.0
        assert res.alert is False

    def test_all_auth_no_alert(self) -> None:
        res = QZSNMALayer().assess([True] * 5)
        assert res.auth_fraction == 1.0
        assert res.n_auth == 5
        assert res.alert is False

    def test_half_auth_no_alert_at_threshold(self) -> None:
        # exactly 50% → NOT less than threshold → no alert
        res = QZSNMALayer().assess([True, False, True, False])
        assert abs(res.auth_fraction - 0.5) < 1e-9
        assert res.alert is False  # 0.5 is not < 0.5

    def test_below_threshold_fires_alert(self) -> None:
        # 2/5 = 0.4 < 0.5
        res = QZSNMALayer().assess([True, False, False, False, False])
        assert res.auth_fraction < 0.5
        assert res.alert is True

    def test_none_authenticated_fires_alert(self) -> None:
        res = QZSNMALayer().assess([False] * 4)
        assert res.auth_fraction == 0.0
        assert res.p_spoof_contribution == 1.0
        assert res.alert is True
