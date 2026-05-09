"""Smoke tests for src/gnss/core.py — OSNMA/TESLA verification engine (T1500)."""

from __future__ import annotations

import os

import pytest

from gnss.core import (
    DISCLOSURE_DELAY,
    EPH_SIZE,
    MAC_SIZE_BITS,
    NUM_SVIDS,
    OSNMAAuthority,
    OSNMAReceiver,
    OSNMATransmitter,
    SimReport,
    SpoofingAttacker,
    TESLAKeyChain,
    make_eph,
    run_simulation,
)

# ---------------------------------------------------------------------------
# TESLAKeyChain
# ---------------------------------------------------------------------------


class TestTESLAKeyChain:
    def test_construction(self) -> None:
        chain = TESLAKeyChain(n=10, seed=0)
        assert chain.root == chain.get_key(9)

    def test_key_length(self) -> None:
        chain = TESLAKeyChain(n=8, seed=1)
        key = chain.get_key(0)
        assert len(key) == TESLAKeyChain.KEY_BYTES

    def test_keys_differ(self) -> None:
        chain = TESLAKeyChain(n=8, seed=0)
        keys = [chain.get_key(i) for i in range(8)]
        assert len(set(keys)) == 8

    def test_index_out_of_range(self) -> None:
        chain = TESLAKeyChain(n=5, seed=0)
        with pytest.raises(IndexError):
            chain.get_key(5)
        with pytest.raises(IndexError):
            chain.get_key(-1)

    def test_verify_valid(self) -> None:
        chain = TESLAKeyChain(n=10, seed=0)
        # K_3 should verify against anchor K_7
        assert chain.verify(chain.get_key(3), index=3, anchor_index=7, anchor_key=chain.get_key(7))

    def test_verify_invalid_wrong_key(self) -> None:
        chain = TESLAKeyChain(n=10, seed=0)
        wrong = os.urandom(TESLAKeyChain.KEY_BYTES)
        assert not chain.verify(wrong, index=3, anchor_index=7, anchor_key=chain.get_key(7))

    def test_verify_index_not_before_anchor(self) -> None:
        chain = TESLAKeyChain(n=10, seed=0)
        # index >= anchor_index must return False
        k5 = chain.get_key(5)
        assert not chain.verify(k5, index=5, anchor_index=5, anchor_key=k5)

    def test_deterministic(self) -> None:
        c1 = TESLAKeyChain(n=8, seed=42)
        c2 = TESLAKeyChain(n=8, seed=42)
        assert c1.get_key(0) == c2.get_key(0)

    def test_different_seeds_differ(self) -> None:
        c1 = TESLAKeyChain(n=8, seed=0)
        c2 = TESLAKeyChain(n=8, seed=1)
        assert c1.get_key(0) != c2.get_key(0)


# ---------------------------------------------------------------------------
# OSNMAAuthority (ECDSA-P256)
# ---------------------------------------------------------------------------


class TestOSNMAAuthority:
    def test_sign_verify_round_trip(self) -> None:
        auth = OSNMAAuthority()
        kroot = os.urandom(16)
        params = {"key_size_bits": 128, "mac_size_bits": 40, "delay": 2}
        sig = auth.sign_root(kroot, epoch=10, params=params)
        assert len(sig) == 64
        assert auth.verify_root_sig(kroot, epoch=10, params=params, sig=sig)

    def test_verify_wrong_epoch_fails(self) -> None:
        auth = OSNMAAuthority()
        kroot = os.urandom(16)
        params = {"key_size_bits": 128, "mac_size_bits": 40, "delay": 2}
        sig = auth.sign_root(kroot, epoch=10, params=params)
        assert not auth.verify_root_sig(kroot, epoch=99, params=params, sig=sig)

    def test_verify_tampered_kroot_fails(self) -> None:
        auth = OSNMAAuthority()
        kroot = os.urandom(16)
        params = {"key_size_bits": 128, "mac_size_bits": 40, "delay": 2}
        sig = auth.sign_root(kroot, epoch=10, params=params)
        tampered = bytes(b ^ 0xFF for b in kroot)
        assert not auth.verify_root_sig(tampered, epoch=10, params=params, sig=sig)


# ---------------------------------------------------------------------------
# OSNMATransmitter + OSNMAReceiver (genuine scenario)
# ---------------------------------------------------------------------------


def _build_system(
    num_epochs: int = 20,
    seed: int = 0,
) -> tuple[TESLAKeyChain, OSNMAAuthority, list[OSNMATransmitter], OSNMAReceiver]:
    from gnss.core import KEY_SIZE_BITS  # re-import for clarity

    chain = TESLAKeyChain(n=num_epochs + 10, seed=seed)
    authority = OSNMAAuthority()
    params: dict[str, int] = {
        "key_size_bits": KEY_SIZE_BITS,
        "mac_size_bits": MAC_SIZE_BITS,
        "delay": DISCLOSURE_DELAY,
    }
    root_epoch = num_epochs + 9
    sig = authority.sign_root(chain.root, root_epoch, params)
    txs = [OSNMATransmitter(svid=i + 1, chain=chain) for i in range(NUM_SVIDS)]
    rx = OSNMAReceiver(
        authority.public_key,
        params,
        sig,
        chain.root,
        root_epoch,
        authority,
        eph_oracle=make_eph,
    )
    return chain, authority, txs, rx


class TestGenuineScenario:
    def test_genuine_messages_not_detected(self) -> None:
        num_epochs = 20
        chain, _, txs, rx = _build_system(num_epochs=num_epochs)
        detected_count = 0
        for ep in range(num_epochs):
            gst = ep * 30
            for tx in txs:
                eph = make_eph(tx.svid, ep)
                msg = tx.broadcast(ep, eph, gst)
                result = rx.receive(msg, receive_time_epoch=ep + 0.5)
                if result is not None and result.detected:
                    detected_count += 1
        # Genuine messages must not trigger false alarms
        assert detected_count == 0

    def test_receiver_returns_results_after_delay(self) -> None:
        num_epochs = 10
        _, _, txs, rx = _build_system(num_epochs=num_epochs)
        results = []
        for ep in range(num_epochs):
            gst = ep * 30
            msg = txs[0].broadcast(ep, make_eph(1, ep), gst)
            r = rx.receive(msg, receive_time_epoch=ep + 0.5)
            if r is not None:
                results.append(r)
        # Results should appear after DISCLOSURE_DELAY epochs
        assert len(results) > 0


# ---------------------------------------------------------------------------
# SpoofingAttacker
# ---------------------------------------------------------------------------


class TestSpoofingAttacker:
    def setup_method(self) -> None:
        self.num_epochs = 20
        self.chain, _, self.txs, self.rx = _build_system(num_epochs=self.num_epochs)
        self.attacker = SpoofingAttacker()

    def _genuine_msg(self, svid_idx: int = 0, ep: int = 5) -> tuple:
        tx = self.txs[svid_idx]
        eph = make_eph(tx.svid, ep)
        gst = ep * 30
        msg = tx.broadcast(ep, eph, gst)
        return tx, msg

    def test_naive_replay_is_spoofed(self) -> None:
        tx, orig = self._genuine_msg(ep=4)
        replayed = self.attacker.naive_replay(orig, ep=8)
        assert replayed.is_spoofed

    def test_modified_replay_is_spoofed(self) -> None:
        _, orig = self._genuine_msg(ep=5)
        fake = self.attacker.modified_replay(orig, os.urandom(EPH_SIZE))
        assert fake.is_spoofed

    def test_key_disclosure_is_spoofed(self) -> None:
        tx, orig = self._genuine_msg(ep=5)
        disc_key = self.chain.get_key(max(0, 5 - DISCLOSURE_DELAY))
        fake = self.attacker.key_disclosure(orig, disc_key, os.urandom(EPH_SIZE))
        assert fake.is_spoofed

    def test_late_injection_is_spoofed(self) -> None:
        tx = self.txs[0]
        fake = self.attacker.late_injection(
            tx.svid, ep=8, chain=self.chain, fake_eph=os.urandom(EPH_SIZE)
        )
        assert fake.is_spoofed

    def test_key_compromise_is_spoofed(self) -> None:
        tx = self.txs[0]
        ep = 8
        fake = self.attacker.key_compromise(
            tx.svid, ep=ep, gst=ep * 30, chain=self.chain, fake_eph=os.urandom(EPH_SIZE)
        )
        assert fake.is_spoofed


# ---------------------------------------------------------------------------
# make_eph
# ---------------------------------------------------------------------------


class TestMakeEph:
    def test_deterministic(self) -> None:
        assert make_eph(1, 5) == make_eph(1, 5)

    def test_different_svid(self) -> None:
        assert make_eph(1, 5) != make_eph(2, 5)

    def test_different_epoch(self) -> None:
        assert make_eph(1, 5) != make_eph(1, 6)

    def test_length(self) -> None:
        assert len(make_eph(1, 0)) == 32


# ---------------------------------------------------------------------------
# run_simulation — end-to-end smoke test
# ---------------------------------------------------------------------------


class TestRunSimulation:
    def test_returns_sim_report(self) -> None:
        report = run_simulation(num_epochs=15, attack_prob=0.3, seed=0)
        assert isinstance(report, SimReport)

    def test_counts_consistent(self) -> None:
        report = run_simulation(num_epochs=15, attack_prob=0.3, seed=0)
        assert report.spoofed + report.normal == report.total
        assert report.tp + report.fn == report.spoofed
        assert report.fp + report.tn == report.normal

    def test_metrics_in_range(self) -> None:
        report = run_simulation(num_epochs=15, attack_prob=0.3, seed=0)
        assert 0.0 <= report.p_fa <= 1.0
        assert 0.0 <= report.p_md <= 1.0
        assert 0.0 <= report.precision <= 1.0
        assert 0.0 <= report.recall <= 1.0
        assert 0.0 <= report.f1 <= 1.0

    def test_by_attack_type_keys(self) -> None:
        report = run_simulation(num_epochs=20, attack_prob=0.4, seed=1)
        expected_keys = {
            "naive_replay", "modified_replay", "key_disclosure",
            "late_injection", "key_compromise", "none",
        }
        assert set(report.by_attack_type.keys()).issubset(expected_keys)

    def test_deterministic(self) -> None:
        r1 = run_simulation(num_epochs=15, attack_prob=0.3, seed=7)
        r2 = run_simulation(num_epochs=15, attack_prob=0.3, seed=7)
        assert r1.tp == r2.tp
        assert r1.fp == r2.fp

    def test_no_attack_no_false_alarms(self) -> None:
        report = run_simulation(num_epochs=15, attack_prob=0.0, seed=0)
        assert report.fp == 0
        assert report.p_fa == 0.0

    def test_quantum_detections_field(self) -> None:
        report = run_simulation(num_epochs=20, attack_prob=0.5, seed=3)
        assert isinstance(report.quantum_detections, int)
        assert report.quantum_detections >= 0
