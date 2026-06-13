"""Tests for src/gnss/osnma_inav.py — Galileo I/NAV OSNMA verification engine."""

from __future__ import annotations

import os

import pytest

from gnss.osnma_inav import (
    ADKD_INAV_CED,
    ALPHA_BYTES,
    KEY_SIZE_BYTES,
    MAC_TAG_BYTES,
    NAV_DATA_BYTES,
    NMA_STATUS_DONT_USE,
    NMA_STATUS_OPERATIONAL,
    NMA_STATUS_TEST,
    SUBFRAME_DURATION_S,
    TESLA_DELAY,
    GSTTESLAChain,
    INavOSNMAEngine,
    INavOSNMASimulator,
    SubframeVerifyResult,
    compute_mac_tag,
    make_inav_nav_data,
    run_inav_simulation,
)

# ---------------------------------------------------------------------------
# GSTTESLAChain
# ---------------------------------------------------------------------------


class TestGSTTESLAChain:
    _ALPHA = bytes(range(ALPHA_BYTES))

    def _chain(self, n: int = 12, seed: int = 0) -> GSTTESLAChain:
        return GSTTESLAChain(n=n, gst_start=0, alpha=self._ALPHA, seed=seed)

    def test_root_is_last_key(self) -> None:
        chain = self._chain()
        assert chain.root == chain.get_key(chain._n - 1)

    def test_key_length(self) -> None:
        chain = self._chain()
        for i in range(chain._n):
            assert len(chain.get_key(i)) == KEY_SIZE_BYTES

    def test_keys_differ(self) -> None:
        chain = self._chain(n=8)
        keys = [chain.get_key(i) for i in range(8)]
        assert len(set(keys)) == 8

    def test_index_out_of_range_negative(self) -> None:
        chain = self._chain(n=5)
        with pytest.raises(IndexError):
            chain.get_key(-1)

    def test_index_out_of_range_upper(self) -> None:
        chain = self._chain(n=5)
        with pytest.raises(IndexError):
            chain.get_key(5)

    def test_gst_of(self) -> None:
        chain = GSTTESLAChain(n=5, gst_start=3000, alpha=self._ALPHA, seed=0)
        assert chain.gst_of(0) == 3000
        assert chain.gst_of(3) == 3000 + 3 * SUBFRAME_DURATION_S

    def test_verify_valid(self) -> None:
        chain = self._chain(n=12)
        # K_3 should verify against anchor K_8
        assert chain.verify_key(chain.get_key(3), idx=3, anchor_idx=8, anchor_key=chain.get_key(8))

    def test_verify_adjacent(self) -> None:
        chain = self._chain(n=12)
        # K_5 should verify against K_6
        assert chain.verify_key(chain.get_key(5), idx=5, anchor_idx=6, anchor_key=chain.get_key(6))

    def test_verify_wrong_key(self) -> None:
        chain = self._chain(n=12)
        wrong = os.urandom(KEY_SIZE_BYTES)
        assert not chain.verify_key(wrong, idx=3, anchor_idx=8, anchor_key=chain.get_key(8))

    def test_verify_index_ge_anchor_false(self) -> None:
        chain = self._chain(n=12)
        k5 = chain.get_key(5)
        assert not chain.verify_key(k5, idx=5, anchor_idx=5, anchor_key=k5)

    def test_deterministic(self) -> None:
        c1 = GSTTESLAChain(n=8, gst_start=0, alpha=self._ALPHA, seed=7)
        c2 = GSTTESLAChain(n=8, gst_start=0, alpha=self._ALPHA, seed=7)
        assert c1.get_key(0) == c2.get_key(0)

    def test_different_seeds_differ(self) -> None:
        c1 = GSTTESLAChain(n=8, gst_start=0, alpha=self._ALPHA, seed=0)
        c2 = GSTTESLAChain(n=8, gst_start=0, alpha=self._ALPHA, seed=1)
        assert c1.get_key(0) != c2.get_key(0)

    def test_different_alpha_differs(self) -> None:
        alpha1 = bytes(range(ALPHA_BYTES))
        alpha2 = bytes([x ^ 0xFF for x in range(ALPHA_BYTES)])
        c1 = GSTTESLAChain(n=8, gst_start=0, alpha=alpha1, seed=0)
        c2 = GSTTESLAChain(n=8, gst_start=0, alpha=alpha2, seed=0)
        # Same root seed but different alpha → different derived keys
        assert c1.get_key(0) != c2.get_key(0)

    def test_different_gst_differs(self) -> None:
        c1 = GSTTESLAChain(n=8, gst_start=0, alpha=self._ALPHA, seed=0)
        c2 = GSTTESLAChain(n=8, gst_start=1000, alpha=self._ALPHA, seed=0)
        assert c1.get_key(0) != c2.get_key(0)

    def test_alpha_wrong_length_raises(self) -> None:
        with pytest.raises(ValueError):
            GSTTESLAChain(n=5, gst_start=0, alpha=b"\x00" * 4, seed=0)


# ---------------------------------------------------------------------------
# compute_mac_tag
# ---------------------------------------------------------------------------


class TestComputeMacTag:
    _KEY = os.urandom(KEY_SIZE_BYTES)
    _NAV = os.urandom(NAV_DATA_BYTES)

    def test_length(self) -> None:
        tag = compute_mac_tag(
            self._KEY,
            svid=1,
            gst_sf=0,
            adkd=0,
            cop=0,
            nma_status=NMA_STATUS_OPERATIONAL,
            nav_data=self._NAV,
        )
        assert len(tag) == MAC_TAG_BYTES

    def test_deterministic(self) -> None:
        args = dict(
            key=self._KEY,
            svid=2,
            gst_sf=60,
            adkd=0,
            cop=0,
            nma_status=NMA_STATUS_OPERATIONAL,
            nav_data=self._NAV,
        )
        assert compute_mac_tag(**args) == compute_mac_tag(**args)

    def test_different_key(self) -> None:
        other_key = os.urandom(KEY_SIZE_BYTES)
        t1 = compute_mac_tag(self._KEY, 1, 0, 0, 0, NMA_STATUS_OPERATIONAL, self._NAV)
        t2 = compute_mac_tag(other_key, 1, 0, 0, 0, NMA_STATUS_OPERATIONAL, self._NAV)
        assert t1 != t2

    def test_different_svid(self) -> None:
        t1 = compute_mac_tag(self._KEY, 1, 0, 0, 0, NMA_STATUS_OPERATIONAL, self._NAV)
        t2 = compute_mac_tag(self._KEY, 2, 0, 0, 0, NMA_STATUS_OPERATIONAL, self._NAV)
        assert t1 != t2

    def test_different_gst(self) -> None:
        t1 = compute_mac_tag(self._KEY, 1, 0, 0, 0, NMA_STATUS_OPERATIONAL, self._NAV)
        t2 = compute_mac_tag(self._KEY, 1, 30, 0, 0, NMA_STATUS_OPERATIONAL, self._NAV)
        assert t1 != t2

    def test_different_nav_data(self) -> None:
        other_nav = os.urandom(NAV_DATA_BYTES)
        t1 = compute_mac_tag(self._KEY, 1, 0, 0, 0, NMA_STATUS_OPERATIONAL, self._NAV)
        t2 = compute_mac_tag(self._KEY, 1, 0, 0, 0, NMA_STATUS_OPERATIONAL, other_nav)
        assert t1 != t2

    def test_different_nma_status(self) -> None:
        t1 = compute_mac_tag(self._KEY, 1, 0, 0, 0, NMA_STATUS_OPERATIONAL, self._NAV)
        t2 = compute_mac_tag(self._KEY, 1, 0, 0, 0, NMA_STATUS_TEST, self._NAV)
        assert t1 != t2

    def test_different_adkd(self) -> None:
        t1 = compute_mac_tag(self._KEY, 1, 0, ADKD_INAV_CED, 0, NMA_STATUS_OPERATIONAL, self._NAV)
        t2 = compute_mac_tag(self._KEY, 1, 0, 4, 0, NMA_STATUS_OPERATIONAL, self._NAV)
        assert t1 != t2


# ---------------------------------------------------------------------------
# make_inav_nav_data
# ---------------------------------------------------------------------------


class TestMakeINavNavData:
    def test_length(self) -> None:
        assert len(make_inav_nav_data(1, 0)) == NAV_DATA_BYTES

    def test_deterministic(self) -> None:
        assert make_inav_nav_data(3, 5) == make_inav_nav_data(3, 5)

    def test_different_svid(self) -> None:
        assert make_inav_nav_data(1, 0) != make_inav_nav_data(2, 0)

    def test_different_epoch(self) -> None:
        assert make_inav_nav_data(1, 0) != make_inav_nav_data(1, 1)


# ---------------------------------------------------------------------------
# INavOSNMASimulator
# ---------------------------------------------------------------------------


class TestINavOSNMASimulator:
    def test_engine_params_keys(self) -> None:
        sim = INavOSNMASimulator(svids=[1, 2], n_subframes=5)
        params = sim.engine_params
        assert set(params.keys()) == {"kroot", "kroot_idx", "gst_start", "alpha"}

    def test_engine_params_kroot_length(self) -> None:
        sim = INavOSNMASimulator(svids=[1], n_subframes=5)
        assert len(sim.engine_params["kroot"]) == KEY_SIZE_BYTES

    def test_engine_params_alpha_length(self) -> None:
        sim = INavOSNMASimulator(svids=[1], n_subframes=5)
        assert len(sim.engine_params["alpha"]) == ALPHA_BYTES

    def test_svids_property(self) -> None:
        sim = INavOSNMASimulator(svids=[3, 7, 11], n_subframes=5)
        assert sim.svids == [3, 7, 11]

    def test_make_subframe_fields(self) -> None:
        sim = INavOSNMASimulator(svids=[1], n_subframes=5)
        sf = sim.make_subframe(svid=1, sf_idx=2)
        assert sf.svid == 1
        assert sf.subframe_idx == 2
        assert sf.gst_sf == 2 * SUBFRAME_DURATION_S
        assert len(sf.nav_data) == NAV_DATA_BYTES
        assert len(sf.mack.tag0) == MAC_TAG_BYTES

    def test_make_subframe_no_tesla_key_before_delay(self) -> None:
        sim = INavOSNMASimulator(svids=[1], n_subframes=5)
        for sf_idx in range(TESLA_DELAY):
            sf = sim.make_subframe(svid=1, sf_idx=sf_idx)
            assert sf.mack.tesla_key is None
            assert sf.mack.key_id < 0

    def test_make_subframe_tesla_key_after_delay(self) -> None:
        sim = INavOSNMASimulator(svids=[1], n_subframes=5)
        sf = sim.make_subframe(svid=1, sf_idx=TESLA_DELAY)
        assert sf.mack.tesla_key is not None
        assert sf.mack.key_id == 0

    def test_tamper_tag0_changes_tag(self) -> None:
        sim = INavOSNMASimulator(svids=[1], n_subframes=5, seed=0)
        genuine = sim.make_subframe(1, 2)
        tampered = sim.make_subframe(1, 2, tamper_tag0=True)
        # Both tamper=True calls produce fresh random bytes, so they won't match
        # the genuine tag and (with overwhelming probability) each other
        assert genuine.mack.tag0 != tampered.mack.tag0

    def test_tamper_tesla_key_sets_zeros(self) -> None:
        sim = INavOSNMASimulator(svids=[1], n_subframes=5)
        sf = sim.make_subframe(1, TESLA_DELAY, tamper_tesla_key=True)
        assert sf.mack.tesla_key == bytes(KEY_SIZE_BYTES)

    def test_late_recv_override(self) -> None:
        sim = INavOSNMASimulator(svids=[1], n_subframes=5)
        sf = sim.make_subframe(1, 0, late_recv_delay_s=SUBFRAME_DURATION_S + 5.0)
        assert sf.recv_time_gst > SUBFRAME_DURATION_S


# ---------------------------------------------------------------------------
# INavOSNMAEngine — core verification logic
# ---------------------------------------------------------------------------


def _build_engine_and_sim(
    svids: list[int] = None,
    n_subframes: int = 8,
    seed: int = 0,
) -> tuple[INavOSNMAEngine, INavOSNMASimulator]:
    if svids is None:
        svids = [1, 2, 3]
    sim = INavOSNMASimulator(svids=svids, n_subframes=n_subframes, seed=seed)
    engine = INavOSNMAEngine(**sim.engine_params)
    return engine, sim


class TestINavOSNMAEngine:
    def test_genuine_subframes_authenticated(self) -> None:
        """All genuine subframes must be authenticated after TESLA_DELAY."""
        engine, sim = _build_engine_and_sim(svids=[1], n_subframes=6)
        results: list[SubframeVerifyResult] = []
        for sf_idx in range(6):
            sf = sim.make_subframe(1, sf_idx)
            results.append(engine.verify_subframe(sf))
        # sf_idx 0 cannot be verified (no key disclosed yet)
        assert not results[0].authenticated
        # sf_idx TESLA_DELAY onward should be authenticated
        for r in results[TESLA_DELAY:]:
            assert r.authenticated, f"sf_idx={r.subframe_idx} not authenticated"

    def test_no_auth_before_tesla_delay(self) -> None:
        engine, sim = _build_engine_and_sim(svids=[1])
        for sf_idx in range(TESLA_DELAY):
            sf = sim.make_subframe(1, sf_idx)
            r = engine.verify_subframe(sf)
            assert not r.key_valid
            assert not r.mac_valid
            assert not r.receipt_safe
            assert not r.authenticated

    def test_tampered_tag0_mac_invalid(self) -> None:
        engine, sim = _build_engine_and_sim(svids=[1])
        # sf_0 with tampered tag-0 buffered, then sf_1 discloses K_0
        sf0 = sim.make_subframe(1, 0, tamper_tag0=True)
        sf1 = sim.make_subframe(1, 1)
        engine.verify_subframe(sf0)
        r = engine.verify_subframe(sf1)
        assert r.key_valid  # TESLA key itself is correct
        assert not r.mac_valid
        assert not r.authenticated

    def test_tampered_nav_data_mac_invalid(self) -> None:
        engine, sim = _build_engine_and_sim(svids=[1])
        sf0 = sim.make_subframe(1, 0, tamper_nav_data=True)
        sf1 = sim.make_subframe(1, 1)
        engine.verify_subframe(sf0)
        r = engine.verify_subframe(sf1)
        # tag-0 was computed over the real nav_data, but buffer holds tampered nav_data
        assert r.key_valid
        assert not r.mac_valid
        assert not r.authenticated

    def test_tampered_tesla_key_key_invalid(self) -> None:
        engine, sim = _build_engine_and_sim(svids=[1])
        sf0 = sim.make_subframe(1, 0)
        sf1 = sim.make_subframe(1, 1, tamper_tesla_key=True)
        engine.verify_subframe(sf0)
        r = engine.verify_subframe(sf1)
        assert not r.key_valid
        assert not r.authenticated

    def test_receipt_safety_violation(self) -> None:
        """Subframe received after key disclosure is rejected."""
        engine, sim = _build_engine_and_sim(svids=[1])
        # sf_0 arrives late: GST_sf_0 + 31 > GST_sf_1 = GST_sf_0 + 30
        late_delay = float(SUBFRAME_DURATION_S) + 1.0
        sf0 = sim.make_subframe(1, 0, late_recv_delay_s=late_delay)
        sf1 = sim.make_subframe(1, 1)
        engine.verify_subframe(sf0)
        r = engine.verify_subframe(sf1)
        assert r.key_valid
        assert not r.receipt_safe
        assert not r.authenticated

    def test_dont_use_nma_status_rejected(self) -> None:
        sim = INavOSNMASimulator(svids=[1], n_subframes=5, nma_status=NMA_STATUS_DONT_USE)
        engine = INavOSNMAEngine(**sim.engine_params)
        for sf_idx in range(5):
            sf = sim.make_subframe(1, sf_idx)
            r = engine.verify_subframe(sf)
            assert not r.nma_ok
            assert not r.authenticated

    def test_test_nma_status_accepted_by_default(self) -> None:
        sim = INavOSNMASimulator(svids=[1], n_subframes=4, nma_status=NMA_STATUS_TEST)
        engine = INavOSNMAEngine(**sim.engine_params)
        results = []
        for sf_idx in range(4):
            sf = sim.make_subframe(1, sf_idx)
            results.append(engine.verify_subframe(sf))
        # TEST status should be accepted; subframes from TESLA_DELAY onward verified
        for r in results[TESLA_DELAY:]:
            assert r.nma_ok
            assert r.authenticated

    def test_authenticated_svids_multiple(self) -> None:
        engine, sim = _build_engine_and_sim(svids=[1, 2, 3], n_subframes=5)
        for svid in [1, 2, 3]:
            for sf_idx in range(5):
                engine.verify_subframe(sim.make_subframe(svid, sf_idx))
        flags = engine.authenticated_svids([1, 2, 3])
        assert flags == [True, True, True]

    def test_authenticated_svids_absent_svid_false(self) -> None:
        engine, sim = _build_engine_and_sim(svids=[1], n_subframes=5)
        for sf_idx in range(5):
            engine.verify_subframe(sim.make_subframe(1, sf_idx))
        # SVID 99 was never seen
        flags = engine.authenticated_svids([1, 99])
        assert flags[0] is True
        assert flags[1] is False

    def test_authenticated_svids_empty_list(self) -> None:
        engine, sim = _build_engine_and_sim(svids=[1], n_subframes=3)
        assert engine.authenticated_svids([]) == []

    def test_reset_clears_state(self) -> None:
        engine, sim = _build_engine_and_sim(svids=[1], n_subframes=5)
        for sf_idx in range(5):
            engine.verify_subframe(sim.make_subframe(1, sf_idx))
        assert engine.authenticated_svids([1]) == [True]

        engine.reset()
        assert engine.authenticated_svids([1]) == [False]
        assert len(engine._buffer) == 0
        assert len(engine._verified_keys) == 1  # only K_ROOT remains

    def test_wrong_kroot_rejects_all(self) -> None:
        """Engine initialised with wrong K_ROOT must reject every subframe."""
        sim = INavOSNMASimulator(svids=[1], n_subframes=5, seed=0)
        params = sim.engine_params
        params["kroot"] = os.urandom(KEY_SIZE_BYTES)  # wrong root
        engine = INavOSNMAEngine(**params)
        for sf_idx in range(5):
            sf = sim.make_subframe(1, sf_idx)
            r = engine.verify_subframe(sf)
            assert not r.key_valid
            assert not r.authenticated

    def test_verify_result_fields_present(self) -> None:
        engine, sim = _build_engine_and_sim(svids=[1], n_subframes=4)
        sf = sim.make_subframe(1, 2)
        engine.verify_subframe(sim.make_subframe(1, 0))
        engine.verify_subframe(sim.make_subframe(1, 1))
        r = engine.verify_subframe(sf)
        assert isinstance(r, SubframeVerifyResult)
        assert r.svid == 1
        assert r.subframe_idx == 2
        assert r.gst_sf == 2 * SUBFRAME_DURATION_S
        assert isinstance(r.key_valid, bool)
        assert isinstance(r.mac_valid, bool)
        assert isinstance(r.receipt_safe, bool)
        assert isinstance(r.nma_ok, bool)
        assert isinstance(r.authenticated, bool)

    def test_partial_tamper_partial_rejection(self) -> None:
        """Only consistently tampered SVIDs are rejected; genuine SVID authenticates."""
        engine, sim = _build_engine_and_sim(svids=[1, 2], n_subframes=5)
        for sf_idx in range(5):
            # SVID 1: all genuine — must authenticate from TESLA_DELAY onward
            engine.verify_subframe(sim.make_subframe(1, sf_idx))
            # SVID 2: every subframe has a tampered tag-0 → mac_valid always False
            engine.verify_subframe(sim.make_subframe(2, sf_idx, tamper_tag0=True))
        flags = engine.authenticated_svids([1, 2])
        assert flags[0] is True  # SVID 1: all genuine
        assert flags[1] is False  # SVID 2: every subframe tampered

    def test_custom_nma_accept_set(self) -> None:
        """Only NMA_STATUS_OPERATIONAL accepted when custom set is supplied."""
        sim = INavOSNMASimulator(svids=[1], n_subframes=5, nma_status=NMA_STATUS_TEST)
        engine = INavOSNMAEngine(
            **sim.engine_params,
            nma_status_accept=frozenset({NMA_STATUS_OPERATIONAL}),
        )
        for sf_idx in range(5):
            r = engine.verify_subframe(sim.make_subframe(1, sf_idx))
            assert not r.nma_ok
            assert not r.authenticated

    def test_multi_subframe_key_reuse(self) -> None:
        """Each subframe uses a distinct key; later subframes are also authenticated."""
        engine, sim = _build_engine_and_sim(svids=[1], n_subframes=10)
        auth_count = 0
        for sf_idx in range(10):
            r = engine.verify_subframe(sim.make_subframe(1, sf_idx))
            if r.authenticated:
                auth_count += 1
        # All subframes from TESLA_DELAY onward should be authenticated
        assert auth_count == 10 - TESLA_DELAY


# ---------------------------------------------------------------------------
# run_inav_simulation
# ---------------------------------------------------------------------------


class TestRunINavSimulation:
    def test_genuine_all_authenticated_after_delay(self) -> None:
        results = run_inav_simulation(svids=[1, 2], n_subframes=8, attack_prob=0.0, seed=0)
        for svid, sf_results in results.items():
            for r in sf_results[TESLA_DELAY:]:
                assert r.authenticated, f"SVID={svid} sf_idx={r.subframe_idx} not auth"

    def test_full_attack_none_authenticated(self) -> None:
        """attack_prob=1.0 means every tag-0 is replaced → nothing authenticates."""
        results = run_inav_simulation(svids=[1], n_subframes=8, attack_prob=1.0, seed=0)
        for r in results[1]:
            assert not r.authenticated

    def test_returns_dict_per_svid(self) -> None:
        results = run_inav_simulation(svids=[5, 7, 11], n_subframes=5, seed=1)
        assert set(results.keys()) == {5, 7, 11}
        for sf_list in results.values():
            assert len(sf_list) == 5

    def test_deterministic(self) -> None:
        r1 = run_inav_simulation(svids=[1, 2], n_subframes=6, attack_prob=0.3, seed=42)
        r2 = run_inav_simulation(svids=[1, 2], n_subframes=6, attack_prob=0.3, seed=42)
        for svid in [1, 2]:
            for a, b in zip(r1[svid], r2[svid]):
                assert a.authenticated == b.authenticated
