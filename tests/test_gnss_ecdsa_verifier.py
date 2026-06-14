"""Tests for src/gnss/ecdsa_verifier.py — ECDSAVerifier (ICD §5.4.4 / §5.5.2)."""

from __future__ import annotations

import hashlib

import pytest
from cryptography.hazmat.primitives.asymmetric.ec import (
    ECDSA,
    SECP256R1,
    SECP521R1,
    generate_private_key,
)
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature
from cryptography.hazmat.primitives.hashes import SHA256, SHA512

from core.data_structures import (
    DSMKROOTMessage,
    DSMPKRMessage,
    ECDSAType,
    HashFunction,
    MACFunction,
)
from gnss.ecdsa_verifier import _SCALAR_BYTES, ECDSAVerifier

# ---------------------------------------------------------------------------
# Fixtures — keys
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def p256_key_pair():
    priv = generate_private_key(SECP256R1())
    return priv, priv.public_key()


@pytest.fixture(scope="module")
def p521_key_pair():
    priv = generate_private_key(SECP521R1())
    return priv, priv.public_key()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_kroot(
    ecdsa_type: ECDSAType,
    priv_key,
    nma_hdr_byte: int = 0x05,
    body: bytes = b"\x00" * 40,
) -> tuple[DSMKROOTMessage, int]:
    """Return (DSMKROOTMessage with valid DS, nma_hdr_byte)."""
    m_kroot = bytes([nma_hdr_byte & 0xFF]) + body
    hash_alg = SHA256() if ecdsa_type == ECDSAType.P256 else SHA512()
    der_sig = priv_key.sign(m_kroot, ECDSA(hash_alg))
    r, s = decode_dss_signature(der_sig)
    scalar = _SCALAR_BYTES[ecdsa_type]
    raw_sig = r.to_bytes(scalar, "big") + s.to_bytes(scalar, "big")

    kroot = DSMKROOTMessage(
        cidkr=0,
        hash_func=HashFunction.SHA_256,
        mac_func=MACFunction.HMAC_SHA_256,
        key_size_bytes=16,
        tag_size_bits=40,
        gst0=0,
        alpha=b"\x00" * 6,
        kroot=b"\xab" * 16,
        ds=raw_sig,
        m_kroot_body=body,
    )
    return kroot, nma_hdr_byte


def _build_merkle_tree(leaves: list[bytes]) -> list[list[bytes]]:
    """Build a complete Merkle tree; return levels bottom-up."""
    level = leaves[:]
    levels = [level]
    while len(level) > 1:
        next_level: list[bytes] = []
        for i in range(0, len(level), 2):
            left = level[i]
            right = level[i + 1] if i + 1 < len(level) else level[i]
            next_level.append(hashlib.sha256(left + right).digest())
        level = next_level
        levels.append(level)
    return levels


def _make_pkr(
    pkid: int,
    ecdsa_type: ECDSAType,
    pub_key_bytes: bytes,
    tree_leaves: list[bytes],
    levels: list[list[bytes]],
) -> DSMPKRMessage:
    """Build a DSMPKRMessage with correct Merkle path."""
    siblings: list[bytes] = []
    idx = pkid
    for level in levels[:-1]:  # skip root level
        sibling_idx = idx ^ 1  # toggle LSB
        if sibling_idx < len(level):
            siblings.append(level[sibling_idx])
        else:
            siblings.append(level[idx])  # duplicate if odd
        idx //= 2

    return DSMPKRMessage(
        pkid=pkid,
        pktype=ecdsa_type,
        public_key=pub_key_bytes,
        merkle_nodes=tuple(siblings),
    )


# ---------------------------------------------------------------------------
# ECDSAVerifier instantiation
# ---------------------------------------------------------------------------


class TestECDSAVerifierInit:
    def test_stores_ecdsa_type_p256(self):
        v = ECDSAVerifier(ECDSAType.P256)
        assert v.ecdsa_type == ECDSAType.P256

    def test_stores_ecdsa_type_p521(self):
        v = ECDSAVerifier(ECDSAType.P521)
        assert v.ecdsa_type == ECDSAType.P521


# ---------------------------------------------------------------------------
# verify_kroot — P-256
# ---------------------------------------------------------------------------


class TestVerifyKrootP256:
    def test_valid_signature(self, p256_key_pair):
        priv, pub = p256_key_pair
        kroot, hdr = _make_kroot(ECDSAType.P256, priv)
        v = ECDSAVerifier(ECDSAType.P256)
        assert v.verify_kroot(kroot, hdr, pub) is True

    def test_wrong_nma_header_fails(self, p256_key_pair):
        priv, pub = p256_key_pair
        kroot, hdr = _make_kroot(ECDSAType.P256, priv, nma_hdr_byte=0x05)
        v = ECDSAVerifier(ECDSAType.P256)
        assert v.verify_kroot(kroot, 0x06, pub) is False

    def test_tampered_body_fails(self, p256_key_pair):
        priv, pub = p256_key_pair
        kroot, hdr = _make_kroot(ECDSAType.P256, priv, body=b"\x00" * 40)
        # Rebuild with tampered body (keep same DS)
        tampered = DSMKROOTMessage(
            cidkr=kroot.cidkr,
            hash_func=kroot.hash_func,
            mac_func=kroot.mac_func,
            key_size_bytes=kroot.key_size_bytes,
            tag_size_bits=kroot.tag_size_bits,
            gst0=kroot.gst0,
            alpha=kroot.alpha,
            kroot=kroot.kroot,
            ds=kroot.ds,
            m_kroot_body=b"\xff" * 40,
        )
        v = ECDSAVerifier(ECDSAType.P256)
        assert v.verify_kroot(tampered, hdr, pub) is False

    def test_tampered_signature_fails(self, p256_key_pair):
        priv, pub = p256_key_pair
        kroot, hdr = _make_kroot(ECDSAType.P256, priv)
        bad_ds = bytes(b ^ 0xFF for b in kroot.ds)
        bad_kroot = DSMKROOTMessage(
            cidkr=kroot.cidkr,
            hash_func=kroot.hash_func,
            mac_func=kroot.mac_func,
            key_size_bytes=kroot.key_size_bytes,
            tag_size_bits=kroot.tag_size_bits,
            gst0=kroot.gst0,
            alpha=kroot.alpha,
            kroot=kroot.kroot,
            ds=bad_ds,
            m_kroot_body=kroot.m_kroot_body,
        )
        v = ECDSAVerifier(ECDSAType.P256)
        assert v.verify_kroot(bad_kroot, hdr, pub) is False

    def test_wrong_public_key_fails(self, p256_key_pair):
        priv, _pub = p256_key_pair
        other_pub = generate_private_key(SECP256R1()).public_key()
        kroot, hdr = _make_kroot(ECDSAType.P256, priv)
        v = ECDSAVerifier(ECDSAType.P256)
        assert v.verify_kroot(kroot, hdr, other_pub) is False

    def test_wrong_curve_key_returns_false(self, p256_key_pair, p521_key_pair):
        priv256, _ = p256_key_pair
        _, pub521 = p521_key_pair
        kroot, hdr = _make_kroot(ECDSAType.P256, priv256)
        v = ECDSAVerifier(ECDSAType.P256)
        assert v.verify_kroot(kroot, hdr, pub521) is False

    def test_short_ds_returns_false(self, p256_key_pair):
        priv, pub = p256_key_pair
        kroot, hdr = _make_kroot(ECDSAType.P256, priv)
        short_kroot = DSMKROOTMessage(
            cidkr=kroot.cidkr,
            hash_func=kroot.hash_func,
            mac_func=kroot.mac_func,
            key_size_bytes=kroot.key_size_bytes,
            tag_size_bits=kroot.tag_size_bits,
            gst0=kroot.gst0,
            alpha=kroot.alpha,
            kroot=kroot.kroot,
            ds=kroot.ds[:30],
            m_kroot_body=kroot.m_kroot_body,
        )
        v = ECDSAVerifier(ECDSAType.P256)
        assert v.verify_kroot(short_kroot, hdr, pub) is False


# ---------------------------------------------------------------------------
# verify_kroot — P-521
# ---------------------------------------------------------------------------


class TestVerifyKrootP521:
    def test_valid_signature(self, p521_key_pair):
        priv, pub = p521_key_pair
        kroot, hdr = _make_kroot(ECDSAType.P521, priv)
        v = ECDSAVerifier(ECDSAType.P521)
        assert v.verify_kroot(kroot, hdr, pub) is True

    def test_tampered_signature_fails(self, p521_key_pair):
        priv, pub = p521_key_pair
        kroot, hdr = _make_kroot(ECDSAType.P521, priv)
        bad_ds = bytes(b ^ 0xAA for b in kroot.ds)
        bad_kroot = DSMKROOTMessage(
            cidkr=kroot.cidkr,
            hash_func=kroot.hash_func,
            mac_func=kroot.mac_func,
            key_size_bytes=kroot.key_size_bytes,
            tag_size_bits=kroot.tag_size_bits,
            gst0=kroot.gst0,
            alpha=kroot.alpha,
            kroot=kroot.kroot,
            ds=bad_ds,
            m_kroot_body=kroot.m_kroot_body,
        )
        v = ECDSAVerifier(ECDSAType.P521)
        assert v.verify_kroot(bad_kroot, hdr, pub) is False

    def test_p256_key_with_p521_verifier_fails(self, p521_key_pair, p256_key_pair):
        priv521, _ = p521_key_pair
        _, pub256 = p256_key_pair
        kroot, hdr = _make_kroot(ECDSAType.P521, priv521)
        v = ECDSAVerifier(ECDSAType.P521)
        assert v.verify_kroot(kroot, hdr, pub256) is False

    def test_ds_length_is_132(self, p521_key_pair):
        priv, pub = p521_key_pair
        kroot, hdr = _make_kroot(ECDSAType.P521, priv)
        assert len(kroot.ds) == 132


# ---------------------------------------------------------------------------
# nma_hdr_byte masking
# ---------------------------------------------------------------------------


class TestNmaHeaderByte:
    def test_only_low_8_bits_used(self, p256_key_pair):
        priv, pub = p256_key_pair
        kroot, hdr = _make_kroot(ECDSAType.P256, priv, nma_hdr_byte=0x05)
        v = ECDSAVerifier(ECDSAType.P256)
        # Passing 0x105 should use only 0x05 (same result)
        assert v.verify_kroot(kroot, 0x105, pub) is True

    def test_different_hdr_bytes_disagree(self, p256_key_pair):
        priv, pub = p256_key_pair
        kroot, hdr = _make_kroot(ECDSAType.P256, priv, nma_hdr_byte=0x05)
        v = ECDSAVerifier(ECDSAType.P256)
        assert v.verify_kroot(kroot, 0x05, pub) is True
        assert v.verify_kroot(kroot, 0x06, pub) is False


# ---------------------------------------------------------------------------
# verify_public_key — Merkle Tree
# ---------------------------------------------------------------------------


class TestVerifyPublicKeyMerkle:
    @pytest.fixture(scope="class")
    def tree_of_4(self):
        """4-leaf Merkle tree with 4 P-256 public keys."""
        keys = [generate_private_key(SECP256R1()) for _ in range(4)]
        pub_bytes = [
            k.public_key().public_bytes(
                encoding=__import__(
                    "cryptography.hazmat.primitives.serialization",
                    fromlist=["Encoding"],
                ).Encoding.X962,
                format=__import__(
                    "cryptography.hazmat.primitives.serialization",
                    fromlist=["PublicFormat"],
                ).PublicFormat.UncompressedPoint,
            )
            for k in keys
        ]
        leaves = [
            hashlib.sha256(bytes([i, ECDSAType.P256.value]) + pub_bytes[i]).digest()
            for i in range(4)
        ]
        levels = _build_merkle_tree(leaves)
        root = levels[-1][0]
        return pub_bytes, leaves, levels, root

    def test_leaf_0_valid(self, tree_of_4):
        pub_bytes, _leaves, levels, root = tree_of_4
        pkr = _make_pkr(0, ECDSAType.P256, pub_bytes[0], _leaves, levels)
        v = ECDSAVerifier(ECDSAType.P256)
        assert v.verify_public_key(pkr, root) is True

    def test_leaf_1_valid(self, tree_of_4):
        pub_bytes, _leaves, levels, root = tree_of_4
        pkr = _make_pkr(1, ECDSAType.P256, pub_bytes[1], _leaves, levels)
        v = ECDSAVerifier(ECDSAType.P256)
        assert v.verify_public_key(pkr, root) is True

    def test_leaf_2_valid(self, tree_of_4):
        pub_bytes, _leaves, levels, root = tree_of_4
        pkr = _make_pkr(2, ECDSAType.P256, pub_bytes[2], _leaves, levels)
        v = ECDSAVerifier(ECDSAType.P256)
        assert v.verify_public_key(pkr, root) is True

    def test_leaf_3_valid(self, tree_of_4):
        pub_bytes, _leaves, levels, root = tree_of_4
        pkr = _make_pkr(3, ECDSAType.P256, pub_bytes[3], _leaves, levels)
        v = ECDSAVerifier(ECDSAType.P256)
        assert v.verify_public_key(pkr, root) is True

    def test_tampered_key_fails(self, tree_of_4):
        pub_bytes, _leaves, levels, root = tree_of_4
        pkr = _make_pkr(0, ECDSAType.P256, pub_bytes[0], _leaves, levels)
        bad_pkr = DSMPKRMessage(
            pkid=pkr.pkid,
            pktype=pkr.pktype,
            public_key=b"\xff" * len(pkr.public_key),
            merkle_nodes=pkr.merkle_nodes,
        )
        v = ECDSAVerifier(ECDSAType.P256)
        assert v.verify_public_key(bad_pkr, root) is False

    def test_wrong_root_fails(self, tree_of_4):
        pub_bytes, _leaves, levels, root = tree_of_4
        pkr = _make_pkr(0, ECDSAType.P256, pub_bytes[0], _leaves, levels)
        bad_root = bytes(b ^ 0xFF for b in root)
        v = ECDSAVerifier(ECDSAType.P256)
        assert v.verify_public_key(pkr, bad_root) is False

    def test_tampered_sibling_fails(self, tree_of_4):
        pub_bytes, _leaves, levels, root = tree_of_4
        pkr = _make_pkr(0, ECDSAType.P256, pub_bytes[0], _leaves, levels)
        bad_siblings = list(pkr.merkle_nodes)
        bad_siblings[0] = bytes(b ^ 0xAA for b in bad_siblings[0])
        bad_pkr = DSMPKRMessage(
            pkid=pkr.pkid,
            pktype=pkr.pktype,
            public_key=pkr.public_key,
            merkle_nodes=tuple(bad_siblings),
        )
        v = ECDSAVerifier(ECDSAType.P256)
        assert v.verify_public_key(bad_pkr, root) is False

    def test_wrong_pkid_fails(self, tree_of_4):
        """Using leaf-1 public key but claiming pkid=0 gives wrong leaf hash."""
        pub_bytes, _leaves, levels, root = tree_of_4
        pkr_0 = _make_pkr(0, ECDSAType.P256, pub_bytes[0], _leaves, levels)
        wrong_pkr = DSMPKRMessage(
            pkid=0,
            pktype=ECDSAType.P256,
            public_key=pub_bytes[1],  # wrong key for this slot
            merkle_nodes=pkr_0.merkle_nodes,
        )
        v = ECDSAVerifier(ECDSAType.P256)
        assert v.verify_public_key(wrong_pkr, root) is False


# ---------------------------------------------------------------------------
# Single-leaf degenerate tree
# ---------------------------------------------------------------------------


class TestMerkleDegenerate:
    def test_single_leaf_tree(self):
        """With one leaf the Merkle root equals the leaf hash itself."""
        priv = generate_private_key(SECP256R1())
        pub_bytes = priv.public_key().public_bytes(
            encoding=__import__(
                "cryptography.hazmat.primitives.serialization",
                fromlist=["Encoding"],
            ).Encoding.X962,
            format=__import__(
                "cryptography.hazmat.primitives.serialization",
                fromlist=["PublicFormat"],
            ).PublicFormat.UncompressedPoint,
        )
        leaf = hashlib.sha256(bytes([0, ECDSAType.P256.value]) + pub_bytes).digest()
        pkr = DSMPKRMessage(
            pkid=0,
            pktype=ECDSAType.P256,
            public_key=pub_bytes,
            merkle_nodes=(),  # no siblings — leaf IS the root
        )
        v = ECDSAVerifier(ECDSAType.P256)
        assert v.verify_public_key(pkr, leaf) is True

    def test_single_leaf_wrong_root_fails(self):
        priv = generate_private_key(SECP256R1())
        pub_bytes = priv.public_key().public_bytes(
            encoding=__import__(
                "cryptography.hazmat.primitives.serialization",
                fromlist=["Encoding"],
            ).Encoding.X962,
            format=__import__(
                "cryptography.hazmat.primitives.serialization",
                fromlist=["PublicFormat"],
            ).PublicFormat.UncompressedPoint,
        )
        pkr = DSMPKRMessage(
            pkid=0,
            pktype=ECDSAType.P256,
            public_key=pub_bytes,
            merkle_nodes=(),
        )
        v = ECDSAVerifier(ECDSAType.P256)
        assert v.verify_public_key(pkr, b"\x00" * 32) is False


# ---------------------------------------------------------------------------
# pktype.value used in leaf computation
# ---------------------------------------------------------------------------


class TestPktypeInLeaf:
    def test_different_pktype_gives_different_leaf(self):
        """Same key bytes with different pktype values must produce different leaf hashes."""
        key_bytes = b"\xab" * 65
        leaf_p256 = hashlib.sha256(bytes([0, ECDSAType.P256.value]) + key_bytes).digest()
        leaf_p521 = hashlib.sha256(bytes([0, ECDSAType.P521.value]) + key_bytes).digest()
        assert leaf_p256 != leaf_p521

    def test_pktype_value_included_in_leaf(self):
        """Verify that pktype.value (0 or 1) is used, not the enum object."""
        key_bytes = b"\xcd" * 65
        pkr = DSMPKRMessage(
            pkid=0,
            pktype=ECDSAType.P256,
            public_key=key_bytes,
            merkle_nodes=(),
        )
        expected_leaf = hashlib.sha256(bytes([0, 0]) + key_bytes).digest()
        v = ECDSAVerifier(ECDSAType.P256)
        assert v.verify_public_key(pkr, expected_leaf) is True


# ---------------------------------------------------------------------------
# _check_key_curve helper
# ---------------------------------------------------------------------------


class TestCheckKeyCurve:
    def test_p256_verifier_accepts_p256_key(self, p256_key_pair):
        _, pub = p256_key_pair
        v = ECDSAVerifier(ECDSAType.P256)
        assert v._check_key_curve(pub) is True

    def test_p256_verifier_rejects_p521_key(self, p521_key_pair):
        _, pub = p521_key_pair
        v = ECDSAVerifier(ECDSAType.P256)
        assert v._check_key_curve(pub) is False

    def test_p521_verifier_accepts_p521_key(self, p521_key_pair):
        _, pub = p521_key_pair
        v = ECDSAVerifier(ECDSAType.P521)
        assert v._check_key_curve(pub) is True

    def test_p521_verifier_rejects_p256_key(self, p256_key_pair):
        _, pub = p256_key_pair
        v = ECDSAVerifier(ECDSAType.P521)
        assert v._check_key_curve(pub) is False
