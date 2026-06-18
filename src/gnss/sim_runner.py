"""GNSS spoofing simulation runner and TESLA key verifier (T1500).

Entry points:
    run_simulation()    — end-to-end OSNMA/TESLA detection simulation → SimReport
    verify_tesla_key()  — standalone TESLA chain membership check
"""

from __future__ import annotations

import hashlib
import hmac
import os

import numpy as np

from gnss.core import (
    DEFAULT_SEED,
    DISCLOSURE_DELAY,
    EPH_SIZE,
    KEY_SIZE_BITS,
    MAC_SIZE_BITS,
    NUM_SVIDS,
    SimReport,
)
from gnss.osnma_simulation import (
    OSNMAReceiver,
    OSNMATransmitter,
    SpoofingAttacker,
    TESLAKeyChain,
    make_eph,
)

# pqc is a research-only module — import lazily inside run_simulation()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


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


def _emit_rows(
    row_base: dict,
    disc_at: str,
    buf_at: str,
    disc_epoch: int,
    buf_epoch: int,
) -> list[dict]:
    """Build raw_rows entries for one verification event.

    Emits one row per attacked epoch so each (svid, attack_epoch) dedup
    group can independently pick up a detected=True row.
    """
    rows: list[dict] = []
    if disc_at != "none":
        rows.append({**row_base, "attack_type": disc_at, "attack_epoch": disc_epoch})
    if buf_at != "none":
        rows.append({**row_base, "attack_type": buf_at, "attack_epoch": buf_epoch})
    if disc_at == "none" and buf_at == "none":
        rows.append({**row_base, "attack_type": "none", "attack_epoch": buf_epoch})
    return rows


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
    from gnss.pqc import RLWEAuthority  # lazy: research-only module

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

    prev_msgs: dict[tuple[int, int], object] = {}
    attack_log: dict[tuple[int, int], str] = {}
    raw_rows: list[dict] = []

    for ep in range(num_epochs):
        gst = ep * 30  # SUBFRAME_DURATION = 30 s
        for tx in txs:
            eph = make_eph(tx.svid, ep)
            real_msg = tx.broadcast(ep, eph, gst)

            attack_type = "none"
            msg_to_send = real_msg
            if rng.random() < attack_prob and ep >= DISCLOSURE_DELAY + 2:
                attack = int(rng.integers(0, 4))
                old_key = (tx.svid, ep - 3)
                if attack == 0 and old_key in prev_msgs:
                    msg_to_send = attacker.naive_replay(prev_msgs[old_key], ep)  # type: ignore[arg-type]
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
                raw_rows.extend(
                    _emit_rows(row_base, disc_at, buf_at, result.disclosure_epoch, result.epoch)
                )

    # -----------------------------------------------------------------------
    # Flush boundary epochs whose key was never disclosed in the loop.
    # Messages buffered at epoch b require K_b to be disclosed at b+delay.
    # If b+delay >= num_epochs, that disclosure never arrived; verify directly
    # from the chain (key_valid is always True for chain-generated keys).
    # -----------------------------------------------------------------------
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
        raw_rows.extend(_emit_rows(row_base, disc_at, buf_at, disc_epoch, buf_epoch))

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


# ---------------------------------------------------------------------------
# Standalone TESLA key verifier
# ---------------------------------------------------------------------------


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
    result = anchor
    for i in range(anchor_index - 1, candidate_index - 1, -1):
        result = TESLAKeyChain._derive(result, i)
    return result == candidate
