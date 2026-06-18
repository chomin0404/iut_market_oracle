"""GNSS OSNMA/TESLA simulation entities (T1500).

Provides the core classes used in the spoofing detection simulation:

    TESLAKeyChain      — hash-chain key generation and verification
    OSNMAAuthority     — ECDSA-P256 root-key signing (simulated GSA)
    OSNMATransmitter   — per-satellite OSNMA broadcaster
    OSNMAReceiver      — verifier (TESLA + receipt-safety + MAC + quantum fidelity)
    SpoofingAttacker   — five attack models
    make_eph           — deterministic dummy ephemeris generator
"""

from __future__ import annotations

import hashlib
import hmac
import os
import struct
from collections.abc import Callable
from typing import Protocol

import numpy as np
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import (
    decode_dss_signature,
    encode_dss_signature,
)

from gnss.core import (
    DISCLOSURE_DELAY,
    KEY_SIZE_BITS,
    MAC_SIZE_BITS,
    SUBFRAME_DURATION,
    NavMessage,
    VerificationResult,
)

# pqc is a research-only module — import lazily inside the functions that use it
# to keep this module importable without loading the Ring-LWE primitives at startup.


# ---------------------------------------------------------------------------
# TESLA key chain
# ---------------------------------------------------------------------------


class TESLAKeyChain:
    """Hash-chain key generation and single-key verification.

    .. deprecated::
        **Simulation use only.**  This class uses an index-only derivation
        (``K_i = SHA-256(K_{i+1} || LE32(i))``) that does NOT bind keys to
        Galileo System Time or the HKROOT alpha nonce.  It is therefore
        vulnerable to temporal-replay and cross-chain tag-reuse attacks.

        For ICD-compliant (GST-bound) key derivation use
        :class:`gnss.osnma_inav.GSTTESLAChain` instead.

    Chain structure (right = root):
        K_0 <--H-- K_1 <--H-- ... <--H-- K_{n-1}

    K_i = trunc_{ks}( SHA-256( K_{i+1} || LE32(i) ) )
    """

    KEY_BYTES: int = KEY_SIZE_BITS // 8

    def __init__(self, n: int, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        self._keys: list[bytes] = [b""] * n
        self._keys[n - 1] = bytes(rng.integers(0, 256, self.KEY_BYTES, dtype=np.uint8))
        for i in range(n - 2, -1, -1):
            self._keys[i] = self._derive(self._keys[i + 1], i)

    @staticmethod
    def _derive(k_succ: bytes, index: int) -> bytes:
        msg = k_succ + struct.pack("<I", index)
        return hashlib.sha256(msg).digest()[: TESLAKeyChain.KEY_BYTES]

    @property
    def root(self) -> bytes:
        return self._keys[-1]

    def get_key(self, index: int) -> bytes:
        if index < 0 or index >= len(self._keys):
            raise IndexError(f"Key index {index} out of range [0, {len(self._keys)})")
        return self._keys[index]

    def verify(self, key: bytes, index: int, anchor_index: int, anchor_key: bytes) -> bool:
        """Verify that key[index] lies on the chain anchored at anchor_key[anchor_index]."""
        if index >= anchor_index:
            return False
        current = anchor_key
        for i in range(anchor_index - 1, index - 1, -1):
            current = self._derive(current, i)
        return current == key


# ---------------------------------------------------------------------------
# OSNMA authority (simulated GSA — signs K_root with ECDSA-P256)
# ---------------------------------------------------------------------------


class OSNMAAuthority:
    """Holds an ECDSA-P256 key pair and issues root-key signatures."""

    def __init__(self) -> None:
        self._privkey = ec.generate_private_key(ec.SECP256R1(), default_backend())

    @property
    def public_key(self) -> ec.EllipticCurvePublicKey:
        return self._privkey.public_key()

    def _build_signed_msg(self, kroot: bytes, epoch: int, params: dict[str, int]) -> bytes:
        return (
            kroot
            + struct.pack(">I", epoch)
            + struct.pack("B", params.get("key_size_bits", KEY_SIZE_BITS) // 8)
            + struct.pack("B", params.get("mac_size_bits", MAC_SIZE_BITS) // 8)
            + struct.pack("B", params.get("delay", DISCLOSURE_DELAY))
        )

    def sign_root(self, kroot: bytes, epoch: int, params: dict[str, int]) -> bytes:
        """Sign K_root; returns raw (r || s) signature, 64 bytes."""
        msg = self._build_signed_msg(kroot, epoch, params)
        der = self._privkey.sign(msg, ec.ECDSA(hashes.SHA256()))
        r, s = decode_dss_signature(der)
        return r.to_bytes(32, "big") + s.to_bytes(32, "big")

    def verify_root_sig(self, kroot: bytes, epoch: int, params: dict[str, int], sig: bytes) -> bool:
        msg = self._build_signed_msg(kroot, epoch, params)
        r = int.from_bytes(sig[:32], "big")
        s = int.from_bytes(sig[32:], "big")
        der = encode_dss_signature(r, s)
        try:
            self.public_key.verify(der, msg, ec.ECDSA(hashes.SHA256()))
            return True
        except InvalidSignature:
            return False


# ---------------------------------------------------------------------------
# Transmitter
# ---------------------------------------------------------------------------


class OSNMATransmitter:
    """Per-satellite OSNMA transmitter."""

    def __init__(self, svid: int, chain: TESLAKeyChain) -> None:
        self.svid = svid
        self._chain = chain

    def broadcast(self, epoch: int, eph_data: bytes, gst: int) -> NavMessage:
        tesla_key = (
            self._chain.get_key(epoch - DISCLOSURE_DELAY) if epoch >= DISCLOSURE_DELAY else None
        )
        msg = NavMessage(
            svid=self.svid,
            epoch=epoch,
            gst=gst,
            eph_data=eph_data,
            tesla_key=tesla_key,
        )
        key_for_mac = self._chain.get_key(epoch)
        raw = hmac.new(key_for_mac, msg.auth_payload(), hashlib.sha256).digest()
        msg.mac_tag = raw[: MAC_SIZE_BITS // 8]
        return msg


# ---------------------------------------------------------------------------
# Receiver / verifier
# ---------------------------------------------------------------------------


class _AuthorityProtocol(Protocol):
    """Structural type for OSNMA/RLWE authority objects."""

    def verify_root_sig(
        self, kroot: bytes, epoch: int, params: dict[str, int], sig: bytes
    ) -> bool: ...


class OSNMAReceiver:
    """OSNMA receiver — TESLA chain + receipt-safety + MAC + quantum fidelity checks.

    Optional eph_oracle enables the quantum fidelity layer:
        eph_oracle(svid, epoch) → expected ephemeris bytes

    When provided, each verified message is also checked with QuantumFidelityDetector.
    This catches key_compromise attacks that bypass all three TESLA checks.
    """

    def __init__(
        self,
        public_key: object,
        chain_params: dict[str, int],
        root_sig: bytes,
        chain_root: bytes,
        root_epoch: int,
        authority: _AuthorityProtocol,
        eph_oracle: Callable[[int, int], bytes] | None = None,
        fidelity_threshold: float = 0.85,  # mirrors pqc.QUANTUM_FIDELITY_THRESHOLD
    ) -> None:
        self._pubkey = public_key
        self._params = chain_params
        self._delay = chain_params.get("delay", DISCLOSURE_DELAY)
        self._buf: dict[tuple[int, int], tuple[NavMessage, float]] = {}
        self._verified_keys: dict[int, bytes] = {}
        self._verified_buf_epochs: set[tuple[int, int]] = set()
        self._eph_oracle = eph_oracle
        if eph_oracle:
            from gnss.pqc import QuantumFidelityDetector  # lazy: research-only module

            self._fidelity: object = QuantumFidelityDetector(fidelity_threshold)
        else:
            self._fidelity = None
        if authority.verify_root_sig(chain_root, root_epoch, chain_params, root_sig):
            self._verified_keys[root_epoch] = chain_root

    def receive(self, msg: NavMessage, receive_time_epoch: float) -> VerificationResult | None:
        """Process one message; returns result only when a TESLA key is disclosed."""
        self._buf[(msg.svid, msg.epoch)] = (msg, receive_time_epoch)
        if msg.tesla_key is None:
            return None

        disclosed_epoch = msg.epoch - self._delay

        # 1. TESLA key chain verification
        key_valid = self._verify_key(msg.tesla_key, disclosed_epoch)

        # 2. Receipt safety: buffer message must have arrived before key disclosure
        #    t_disclose(K_i) = (i + delay)   [in epoch units]
        buf_entry = self._buf.get((msg.svid, disclosed_epoch))
        buffered, buf_recv_time = buf_entry if buf_entry else (None, None)
        key_disclose_time = disclosed_epoch + self._delay
        receipt_safe = (
            buffered is not None
            and buf_recv_time is not None
            and buf_recv_time < key_disclose_time - 0.1
        )

        # 3. MAC verification
        mac_valid = False
        if key_valid and buffered is not None:
            expected = hmac.new(msg.tesla_key, buffered.auth_payload(), hashlib.sha256).digest()[
                : MAC_SIZE_BITS // 8
            ]
            mac_valid = buffered.mac_tag == expected

        # 4. Quantum fidelity check (when eph_oracle is configured)
        quantum_anomaly = False
        if self._fidelity is not None and self._eph_oracle is not None and buffered is not None:
            expected_eph = self._eph_oracle(msg.svid, disclosed_epoch)
            quantum_anomaly = self._fidelity.is_anomaly(buffered.eph_data, expected_eph)

        detected = not (key_valid and mac_valid and receipt_safe) or quantum_anomaly
        if key_valid:
            self._verified_keys[disclosed_epoch] = msg.tesla_key
        self._verified_buf_epochs.add((msg.svid, disclosed_epoch))

        buf_spoofed = buffered.is_spoofed if buffered is not None else False
        return VerificationResult(
            epoch=disclosed_epoch,
            disclosure_epoch=msg.epoch,
            svid=msg.svid,
            key_valid=key_valid,
            mac_valid=mac_valid,
            receipt_safe=receipt_safe,
            is_spoofed=msg.is_spoofed or buf_spoofed,
            detected=detected,
            quantum_anomaly=quantum_anomaly,
        )

    def flush_expired(self, final_epoch: int) -> list[tuple[int, int, NavMessage, float]]:
        """Return unverified buffer entries whose key disclosure epoch <= final_epoch.

        Used after the main simulation loop to surface spoofed messages that were
        buffered at epochs where the key was never disclosed (boundary epochs).

        Returns list of (svid, buf_epoch, msg, recv_time).
        """
        out: list[tuple[int, int, NavMessage, float]] = []
        for (svid, buf_epoch), (msg, recv_time) in self._buf.items():
            if (svid, buf_epoch) in self._verified_buf_epochs:
                continue
            if buf_epoch + self._delay <= final_epoch:
                out.append((svid, buf_epoch, msg, recv_time))
        return out

    def _verify_key(self, key: bytes, epoch: int) -> bool:
        if epoch < 0:
            return False
        anchor_epoch: int | None = None
        anchor_key: bytes | None = None
        for ae, ak in self._verified_keys.items():
            if ae > epoch and (anchor_epoch is None or ae < anchor_epoch):
                anchor_epoch, anchor_key = ae, ak
        if anchor_epoch is None or anchor_key is None:
            return False
        current = anchor_key
        for i in range(anchor_epoch - 1, epoch - 1, -1):
            current = hashlib.sha256(current + struct.pack("<I", i)).digest()[
                : TESLAKeyChain.KEY_BYTES
            ]
        return current == key


# ---------------------------------------------------------------------------
# Spoofing attacker models
# ---------------------------------------------------------------------------


class SpoofingAttacker:
    """Four spoofing attack models used in simulation."""

    def naive_replay(self, original: NavMessage, ep: int) -> NavMessage:
        """Replay an old message at a new epoch — key chain mismatch."""
        return NavMessage(
            svid=original.svid,
            epoch=ep,
            gst=ep * SUBFRAME_DURATION,
            eph_data=original.eph_data,
            mac_tag=original.mac_tag,
            tesla_key=original.tesla_key,
            is_spoofed=True,
        )

    def modified_replay(self, original: NavMessage, fake_eph: bytes) -> NavMessage:
        """Replace ephemeris, forge a random MAC tag — MAC mismatch."""
        return NavMessage(
            svid=original.svid,
            epoch=original.epoch,
            gst=original.gst,
            eph_data=fake_eph,
            mac_tag=os.urandom(MAC_SIZE_BITS // 8),
            tesla_key=original.tesla_key,
            is_spoofed=True,
        )

    def key_disclosure(
        self, original: NavMessage, disclosed_key: bytes, fake_eph: bytes
    ) -> NavMessage:
        """Compute a valid MAC with an already-disclosed key — receipt safety violation."""
        fake = NavMessage(
            svid=original.svid,
            epoch=original.epoch,
            gst=original.gst,
            eph_data=fake_eph,
            tesla_key=original.tesla_key,
            is_spoofed=True,
        )
        raw = hmac.new(disclosed_key, fake.auth_payload(), hashlib.sha256).digest()
        fake.mac_tag = raw[: MAC_SIZE_BITS // 8]
        return fake

    def late_injection(
        self, svid: int, ep: int, chain: TESLAKeyChain, fake_eph: bytes
    ) -> NavMessage:
        """Inject a back-dated message using the just-disclosed key — receipt safety fail."""
        target_epoch = ep - DISCLOSURE_DELAY
        disc_key = chain.get_key(target_epoch)
        fake = NavMessage(
            svid=svid,
            epoch=target_epoch,
            gst=target_epoch * SUBFRAME_DURATION,
            eph_data=fake_eph,
            tesla_key=(
                chain.get_key(target_epoch - DISCLOSURE_DELAY)
                if target_epoch >= DISCLOSURE_DELAY
                else None
            ),
            is_spoofed=True,
        )
        raw = hmac.new(disc_key, fake.auth_payload(), hashlib.sha256).digest()
        fake.mac_tag = raw[: MAC_SIZE_BITS // 8]
        return fake

    def key_compromise(
        self, svid: int, ep: int, gst: int, chain: TESLAKeyChain, fake_eph: bytes
    ) -> NavMessage:
        """Attack with compromised TESLA key K_ep: valid MAC + fake eph.

        Attacker knows the current epoch's key K_ep (not yet public) and uses it
        to forge a valid MAC over fake ephemeris data.  All three TESLA checks pass:
            key_valid=True  (real chain key)
            mac_valid=True  (HMAC computed with real K_ep)
            receipt_safe=True (message delivered on time)

        Only the quantum fidelity layer detects this attack:
            F(fake_eph, make_eph(svid, ep)) ≈ 0.25  < τ=0.85
        """
        real_key = chain.get_key(ep)
        disc_key = chain.get_key(ep - DISCLOSURE_DELAY) if ep >= DISCLOSURE_DELAY else None
        fake = NavMessage(
            svid=svid,
            epoch=ep,
            gst=gst,
            eph_data=fake_eph,
            tesla_key=disc_key,
            is_spoofed=True,
        )
        raw = hmac.new(real_key, fake.auth_payload(), hashlib.sha256).digest()
        fake.mac_tag = raw[: MAC_SIZE_BITS // 8]
        return fake


# ---------------------------------------------------------------------------
# Ephemeris helper
# ---------------------------------------------------------------------------


def make_eph(svid: int, epoch: int) -> bytes:
    """Deterministic dummy ephemeris."""
    return hashlib.sha256(struct.pack(">II", svid, epoch)).digest()
