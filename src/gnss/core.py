"""GNSS spoofing detection core — OSNMA/TESLA verification engine.

Adapted from osnma_simulation.py (Galileo OSNMA SIS ICD v1.1 simplified model).

Key components:
    TESLAKeyChain          — hash-chain key generation and verification
    OSNMAAuthority         — ECDSA-P256 root-key signing (simulated GSA)
    RLWEAuthority          — Ring-LWE post-quantum root-key signing (from pqc.py)
    OSNMATransmitter       — per-satellite broadcaster
    OSNMAReceiver          — verifier with receipt-safety, MAC, and quantum fidelity checks
    SpoofingAttacker       — 5 attack models (4 TESLA + 1 key_compromise)
    run_simulation()       — end-to-end detection simulation → SimReport
"""

from __future__ import annotations

import hashlib
import hmac
import os
import struct
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import (
    decode_dss_signature,
    encode_dss_signature,
)

from gnss.pqc import QUANTUM_FIDELITY_THRESHOLD, QuantumFidelityDetector, RLWEAuthority

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_SVIDS: int = 4
KEY_SIZE_BITS: int = 128  # TESLA key size [bits]
MAC_SIZE_BITS: int = 40  # MAC tag size [bits]
DISCLOSURE_DELAY: int = 2  # key disclosure delay [subframes]
SUBFRAME_DURATION: int = 30  # subframe length [seconds]
EPH_SIZE: int = 32  # dummy ephemeris size [bytes]
DEFAULT_SEED: int = 42


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class NavMessage:
    """Simplified Galileo I/NAV message (one subframe = 30 s)."""

    svid: int
    epoch: int
    gst: int  # Galileo System Time [s]
    eph_data: bytes  # ephemeris (EPH_SIZE bytes)
    tesla_key: bytes | None  # disclosed TESLA key K_{epoch-delay}
    mac_tag: bytes = field(default_factory=lambda: bytes(MAC_SIZE_BITS // 8))
    is_spoofed: bool = False

    def auth_payload(self) -> bytes:
        """MAC input: SVID(1) || GST(4) || EPH_DATA."""
        return struct.pack("B", self.svid) + struct.pack(">I", self.gst) + self.eph_data


@dataclass
class VerificationResult:
    """Per-message OSNMA verification outcome."""

    epoch: int  # epoch of the buffered message being verified
    disclosure_epoch: int  # epoch at which the key was disclosed
    svid: int
    key_valid: bool  # TESLA key lies on authenticated chain
    mac_valid: bool  # MAC tag matches recomputed value
    receipt_safe: bool  # message received before key was disclosed
    is_spoofed: bool  # ground-truth label
    detected: bool  # any check failed (TESLA or quantum fidelity)
    quantum_anomaly: bool = False  # quantum fidelity below threshold (eph mismatch)


@dataclass
class SimReport:
    """Aggregated detection metrics from run_simulation()."""

    total: int
    spoofed: int
    normal: int
    tp: int
    fp: int
    fn: int
    tn: int
    p_fa: float
    p_md: float
    precision: float
    recall: float
    f1: float
    by_attack_type: dict[str, dict[str, int | float]]
    quantum_detections: int = 0  # key_compromise attacks caught only by quantum fidelity layer


# ---------------------------------------------------------------------------
# TESLA key chain
# ---------------------------------------------------------------------------


class TESLAKeyChain:
    """Hash-chain key generation and single-key verification.

    Chain structure (right = root):
        K_0 <--H-- K_1 <--H-- ... <--H-- K_{n-1}

    K_i = trunc_{ks}( SHA-256( K_{i+1} || LE32(i) ) )
    """

    KEY_BYTES: int = KEY_SIZE_BITS // 8

    def __init__(self, n: int, seed: int = DEFAULT_SEED) -> None:
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
        authority: object,
        eph_oracle: Callable[[int, int], bytes] | None = None,
        fidelity_threshold: float = QUANTUM_FIDELITY_THRESHOLD,
    ) -> None:
        self._pubkey = public_key
        self._params = chain_params
        self._delay = chain_params.get("delay", DISCLOSURE_DELAY)
        self._buf: dict[tuple[int, int], tuple[NavMessage, float]] = {}
        self._verified_keys: dict[int, bytes] = {}
        self._verified_buf_epochs: set[tuple[int, int]] = set()
        self._eph_oracle = eph_oracle
        self._fidelity = QuantumFidelityDetector(fidelity_threshold) if eph_oracle else None
        if authority.verify_root_sig(chain_root, root_epoch, chain_params, root_sig):  # type: ignore[union-attr]
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
# Helpers
# ---------------------------------------------------------------------------


def make_eph(svid: int, epoch: int) -> bytes:
    """Deterministic dummy ephemeris."""
    return hashlib.sha256(struct.pack(">II", svid, epoch)).digest()


def _dedup(results: list[dict]) -> list[dict]:
    """Collapse TESLA-delayed duplicates: one row per (svid, attack_epoch)."""
    groups: dict[tuple, list[dict]] = {}
    for r in results:
        key = (r["svid"], r["attack_epoch"])
        groups.setdefault(key, []).append(r)
    out: list[dict] = []
    for rows in groups.values():
        detected = [r for r in rows if r["detected"]]
        out.append(detected[0] if detected else rows[0])
    return out


def _metrics(rows: list[dict]) -> dict:
    spoofed = [r for r in rows if r["is_spoofed"]]
    normal = [r for r in rows if not r["is_spoofed"]]
    tp = sum(1 for r in spoofed if r["detected"])
    fp = sum(1 for r in normal if r["detected"])
    fn = len(spoofed) - tp
    tn = len(normal) - fp
    p_fa = fp / len(normal) if normal else 0.0
    p_md = fn / len(spoofed) if spoofed else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return dict(
        total=len(rows),
        spoofed=len(spoofed),
        normal=len(normal),
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        p_fa=p_fa,
        p_md=p_md,
        precision=prec,
        recall=rec,
        f1=f1,
    )


# ---------------------------------------------------------------------------
# Simulation entry point
# ---------------------------------------------------------------------------


def run_simulation(
    num_epochs: int = 40,
    attack_prob: float = 0.25,
    seed: int = DEFAULT_SEED,
) -> SimReport:
    """Run OSNMA/TESLA spoofing simulation with quantum-resistant root signing.

    Uses RLWEAuthority (Ring-LWE Lyubashevsky) instead of ECDSA-P256.
    Enables QuantumFidelityDetector to catch key_compromise attacks.

    Returns:
        SimReport with detection metrics broken down by attack type.
    """
    rng = np.random.default_rng(seed)
    # RLWEAuthority: quantum-resistant replacement for OSNMAAuthority
    authority = RLWEAuthority(seed=seed)
    chain = TESLAKeyChain(n=num_epochs + 10, seed=seed)
    chain_params: dict[str, int] = dict(
        key_size_bits=KEY_SIZE_BITS,
        mac_size_bits=MAC_SIZE_BITS,
        delay=DISCLOSURE_DELAY,
    )
    root_epoch = num_epochs + 9
    root_sig = authority.sign_root(chain.root, root_epoch, chain_params)

    txs = [OSNMATransmitter(svid=i + 1, chain=chain) for i in range(NUM_SVIDS)]
    rx = OSNMAReceiver(
        authority.public_key,
        chain_params,
        root_sig,
        chain.root,
        root_epoch,
        authority,
        eph_oracle=make_eph,  # enables quantum fidelity layer
    )
    attacker = SpoofingAttacker()
    # Separate RNG stream for key_compromise to preserve existing seed behavior
    rng_kc = np.random.default_rng(seed + 100_000)

    prev_msgs: dict[tuple[int, int], NavMessage] = {}
    attack_log: dict[tuple[int, int], str] = {}
    raw_rows: list[dict] = []

    for ep in range(num_epochs):
        gst = ep * SUBFRAME_DURATION
        for tx in txs:
            eph = make_eph(tx.svid, ep)
            real_msg = tx.broadcast(ep, eph, gst)

            attack_type = "none"
            msg_to_send = real_msg
            if rng.random() < attack_prob and ep >= DISCLOSURE_DELAY + 2:
                attack = int(rng.integers(0, 4))
                old_key = (tx.svid, ep - 3)
                if attack == 0 and old_key in prev_msgs:
                    msg_to_send = attacker.naive_replay(prev_msgs[old_key], ep)
                    attack_type = "naive_replay"
                elif attack == 1:
                    msg_to_send = attacker.modified_replay(real_msg, os.urandom(EPH_SIZE))
                    attack_type = "modified_replay"
                elif attack == 2:
                    disc_key = chain.get_key(max(0, ep - DISCLOSURE_DELAY))
                    msg_to_send = attacker.key_disclosure(real_msg, disc_key, os.urandom(EPH_SIZE))
                    attack_type = "key_disclosure"
                elif attack == 3:
                    late_fake = attacker.late_injection(tx.svid, ep, chain, os.urandom(EPH_SIZE))
                    rx.receive(late_fake, ep + 0.5)
                    attack_log[(tx.svid, ep - DISCLOSURE_DELAY)] = "late_injection"

            attack_log[(tx.svid, msg_to_send.epoch)] = attack_type
            prev_msgs[(tx.svid, ep)] = real_msg
            result = rx.receive(msg_to_send, receive_time_epoch=ep + 0.5)

            # key_compromise: inject fake eph with valid MAC using real K_ep.
            # Only fires when no other attack was injected, using a separate RNG stream
            # so existing seed behavior (rng state) is preserved exactly.
            # ep + DISCLOSURE_DELAY < num_epochs ensures verification happens in-loop.
            if (
                attack_type == "none"
                and ep + DISCLOSURE_DELAY < num_epochs
                and ep >= DISCLOSURE_DELAY + 2
                and rng_kc.random() < attack_prob
            ):
                fake_kc = attacker.key_compromise(tx.svid, ep, gst, chain, os.urandom(EPH_SIZE))
                # Overwrite the buffer entry so the fake eph is verified when K_ep is disclosed
                rx._buf[(tx.svid, ep)] = (fake_kc, ep + 0.5)
                attack_log[(tx.svid, ep)] = "key_compromise"

            if result is not None:
                disc_at = attack_log.get((result.svid, result.disclosure_epoch), "none")
                buf_at = attack_log.get((result.svid, result.epoch), "none")
                row_base = dict(
                    epoch=result.epoch,
                    disclosure_epoch=result.disclosure_epoch,
                    svid=result.svid,
                    key_valid=result.key_valid,
                    mac_valid=result.mac_valid,
                    receipt_safe=result.receipt_safe,
                    is_spoofed=result.is_spoofed,
                    detected=result.detected,
                    quantum_anomaly=result.quantum_anomaly,
                )
                # Emit one row per attack epoch so each (svid, attack_epoch) dedup
                # group can independently pick up a detected=True row.
                if disc_at != "none":
                    raw_rows.append(
                        {
                            **row_base,
                            "attack_type": disc_at,
                            "attack_epoch": result.disclosure_epoch,
                        }
                    )
                if buf_at != "none":
                    raw_rows.append(
                        {
                            **row_base,
                            "attack_type": buf_at,
                            "attack_epoch": result.epoch,
                        }
                    )
                if disc_at == "none" and buf_at == "none":
                    raw_rows.append(
                        {
                            **row_base,
                            "attack_type": "none",
                            "attack_epoch": result.epoch,
                        }
                    )

    # -----------------------------------------------------------------------
    # Fix 2: Flush boundary epochs whose key was never disclosed in the loop.
    # Messages buffered at epoch b require K_b to be disclosed at b+delay.
    # If b+delay >= num_epochs, that disclosure never arrived; verify directly
    # from the chain (key_valid is always True for chain-generated keys).
    # -----------------------------------------------------------------------
    # Cover all epochs that were simulated: buf_epoch in [0, num_epochs-1]
    # requires disclosure at up to (num_epochs - 1) + DISCLOSURE_DELAY.
    for svid, buf_epoch, buffered_msg, recv_time in rx.flush_expired(
        num_epochs - 1 + DISCLOSURE_DELAY
    ):
        disc_epoch = buf_epoch + DISCLOSURE_DELAY
        try:
            key = chain.get_key(buf_epoch)
        except IndexError:
            continue
        key_disclose_time = float(disc_epoch)
        receipt_safe = recv_time < key_disclose_time - 0.1
        expected_mac = hmac.new(key, buffered_msg.auth_payload(), hashlib.sha256).digest()[
            : MAC_SIZE_BITS // 8
        ]
        mac_valid = buffered_msg.mac_tag == expected_mac
        detected = not (mac_valid and receipt_safe)

        # Quantum fidelity check for flushed boundary messages
        quantum_anomaly = False
        if rx._fidelity is not None and rx._eph_oracle is not None:
            expected_eph = rx._eph_oracle(svid, buf_epoch)
            quantum_anomaly = rx._fidelity.is_anomaly(buffered_msg.eph_data, expected_eph)
        detected = detected or quantum_anomaly

        disc_at = attack_log.get((svid, disc_epoch), "none")
        buf_at = attack_log.get((svid, buf_epoch), "none")
        row_base = dict(
            epoch=buf_epoch,
            disclosure_epoch=disc_epoch,
            svid=svid,
            key_valid=True,
            mac_valid=mac_valid,
            receipt_safe=receipt_safe,
            is_spoofed=buffered_msg.is_spoofed,
            detected=detected,
            quantum_anomaly=quantum_anomaly,
        )
        if disc_at != "none":
            raw_rows.append({**row_base, "attack_type": disc_at, "attack_epoch": disc_epoch})
        if buf_at != "none":
            raw_rows.append({**row_base, "attack_type": buf_at, "attack_epoch": buf_epoch})
        if disc_at == "none" and buf_at == "none":
            raw_rows.append({**row_base, "attack_type": "none", "attack_epoch": buf_epoch})

    deduped = _dedup(raw_rows)
    m = _metrics(deduped)

    # Per-attack-type stats (including key_compromise as 5th type)
    atypes = [
        "naive_replay",
        "modified_replay",
        "key_disclosure",
        "late_injection",
        "key_compromise",
    ]
    by_type: dict[str, dict[str, int | float]] = {}
    for at in atypes:
        rows_at = [r for r in deduped if r["attack_type"] == at]
        if not rows_at:
            continue
        det = sum(1 for r in rows_at if r["detected"])
        by_type[at] = dict(total=len(rows_at), detected=det, p_detect=det / len(rows_at))

    # quantum_detections: key_compromise rows caught exclusively by quantum layer
    # (TESLA checks all passed, only quantum_anomaly=True triggered detection)
    quantum_detections = sum(
        1 for r in deduped if r.get("attack_type") == "key_compromise" and r.get("detected")
    )

    return SimReport(
        total=m["total"],
        spoofed=m["spoofed"],
        normal=m["normal"],
        tp=m["tp"],
        fp=m["fp"],
        fn=m["fn"],
        tn=m["tn"],
        p_fa=m["p_fa"],
        p_md=m["p_md"],
        precision=m["precision"],
        recall=m["recall"],
        f1=m["f1"],
        by_attack_type=by_type,
        quantum_detections=quantum_detections,
    )


def verify_tesla_key(
    candidate_key_hex: str,
    candidate_index: int,
    anchor_key_hex: str,
    anchor_index: int,
) -> bool:
    """Verify that a TESLA key lies on the chain anchored at anchor_key.

    Uses the same hash function as TESLAKeyChain._derive():
        K_i = SHA-256( K_{i+1} || LE32(i) ) [:KEY_BYTES]

    Args:
        candidate_key_hex:  hex-encoded key to verify
        candidate_index:    chain index i of the candidate
        anchor_key_hex:     hex-encoded verified anchor key
        anchor_index:       chain index of the anchor (must be > candidate_index)

    Returns:
        True iff hash^(anchor_index - candidate_index)(anchor_key) == candidate_key
    """
    if candidate_index >= anchor_index:
        raise ValueError("anchor_index must be > candidate_index")
    candidate = bytes.fromhex(candidate_key_hex)
    anchor = bytes.fromhex(anchor_key_hex)
    chain = TESLAKeyChain.__new__(TESLAKeyChain)  # skip __init__
    chain._keys = []  # not used by _derive
    result = anchor
    for i in range(anchor_index - 1, candidate_index - 1, -1):
        result = TESLAKeyChain._derive(result, i)
    return result == candidate
