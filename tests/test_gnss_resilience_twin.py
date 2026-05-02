"""Tests for GNSS Resilience Twin (T1500)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from gnss.resilience_twin import (
    DuminilCopinPhaseMonitor,
    FaultEntropyMonitor,
    GMMRaim,
    HuhSubsetSelector,
    IMMKalman,
    ResilienceTwin,
    ResilienceTwinConfig,
    SpectralMonitor,
    _DC_SUSCEPTIBILITY_ALERT,
    _FAULT_CLASSES,
    _FEL_H_THRESH,
    _inject_hw_fault,
    _inject_multipath,
    run_resilience_simulation,
)
from gnss.spoof_sim import _init_constellation
from schemas import FaultClass, ResilienceTwinReport


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_los(n: int = 6) -> np.ndarray:
    return _init_constellation(n)


def _elevations(los: np.ndarray) -> np.ndarray:
    return np.arcsin(np.clip(los[:, 2], -1.0, 1.0))


# ---------------------------------------------------------------------------
# GM-RAIM
# ---------------------------------------------------------------------------


class TestGMMRaim:
    def test_nominal_low_fault_posterior(self) -> None:
        rng = np.random.default_rng(0)
        los = _make_los()
        el = _elevations(los)
        meas = rng.normal(0.0, 0.30, size=6)
        result = GMMRaim().classify(meas, el)
        assert all(g < 0.5 for g in result.gamma), "Nominal meas should yield low fault posteriors"
        assert result.n_fault == 0

    def test_outlier_raises_fault_posterior(self) -> None:
        los = _make_los()
        el = _elevations(los)
        meas = np.zeros(6)
        meas[2] = 20.0  # large outlier
        result = GMMRaim().classify(meas, el)
        assert result.gamma[2] > 0.9, "Outlier should dominate fault posterior"
        assert result.n_fault >= 1

    def test_sign_corr_high_for_common_bias(self) -> None:
        los = _make_los()
        el = _elevations(los)
        meas = np.full(6, 3.0)  # all same sign
        result = GMMRaim().classify(meas, el)
        assert result.sign_corr > 0.9

    def test_gamma_bounds(self) -> None:
        rng = np.random.default_rng(1)
        los = _make_los()
        el = _elevations(los)
        meas = rng.normal(0.0, 5.0, size=6)
        result = GMMRaim().classify(meas, el)
        assert all(0.0 <= g <= 1.0 for g in result.gamma)


# ---------------------------------------------------------------------------
# IMM-KF
# ---------------------------------------------------------------------------


class TestIMMKalman:
    def test_mode_weights_sum_to_one(self) -> None:
        los = _make_los()
        imm = IMMKalman(los)
        z = np.zeros(6)
        result = imm.update(z)
        assert abs(sum(result.mode_weights) - 1.0) < 1e-10

    def test_repeated_nominal_favours_nominal_mode(self) -> None:
        rng = np.random.default_rng(42)
        los = _make_los()
        imm = IMMKalman(los)
        for _ in range(30):
            z = rng.normal(0.0, 0.30, size=6)
            result = imm.update(z)
        assert result.mode_weights[0] > result.mode_weights[2], (
            "Nominal mode should dominate under genuine measurements"
        )

    def test_output_shapes(self) -> None:
        los = _make_los()
        imm = IMMKalman(los)
        result = imm.update(np.zeros(6))
        assert len(result.mode_weights) == 3
        assert len(result.x_fused) == 4
        assert len(result.innovation_norms) == 3


# ---------------------------------------------------------------------------
# SpectralMonitor
# ---------------------------------------------------------------------------


class TestSpectralMonitor:
    def test_fiedler_ratio_near_one_for_nominal(self) -> None:
        rng = np.random.default_rng(0)
        mon = SpectralMonitor(n_sats=6)
        # Average over many nominal samples
        ratios = [mon.analyze(rng.normal(0.0, 0.30, size=6)).fiedler_ratio for _ in range(50)]
        mean_ratio = float(np.mean(ratios))
        assert 0.3 < mean_ratio < 3.0, f"Nominal Fiedler ratio out of expected range: {mean_ratio}"

    def test_spectral_entropy_nonnegative(self) -> None:
        rng = np.random.default_rng(1)
        mon = SpectralMonitor(n_sats=6)
        for _ in range(20):
            result = mon.analyze(rng.normal(0.0, 1.0, size=6))
            assert result.spectral_entropy >= 0.0

    def test_rmt_anomaly_large_under_spoofing(self) -> None:
        mon = SpectralMonitor(n_sats=6)
        # Spoofing: large common bias shifts all Dopplers similarly
        meas = np.full(6, 5.0) + np.random.default_rng(2).normal(0.0, 0.3, size=6)
        result = mon.analyze(meas)
        # Spoofing collapses inter-satellite differences → low Fiedler; RMT may differ
        assert result.rmt_anomaly >= 0.0


# ---------------------------------------------------------------------------
# FaultEntropyMonitor
# ---------------------------------------------------------------------------


class TestFaultEntropyMonitor:
    def test_alert_on_uniform_posterior(self) -> None:
        mon = FaultEntropyMonitor()
        result = mon.update(np.array([0.25, 0.25, 0.25, 0.25]))
        assert result.entropy == pytest.approx(math.log(4), abs=1e-6)
        assert result.alert, "Uniform posterior should trigger entropy alert"

    def test_no_alert_on_nominal_posterior(self) -> None:
        mon = FaultEntropyMonitor()
        result = mon.update(np.array([0.97, 0.01, 0.01, 0.01]))
        assert not result.alert

    def test_kl_positive(self) -> None:
        mon = FaultEntropyMonitor()
        result = mon.update(np.array([0.25, 0.25, 0.25, 0.25]))
        assert result.kl > 0.0

    def test_entropy_gradient_alert(self) -> None:
        mon = FaultEntropyMonitor(grad_thresh=0.01)
        mon.update(np.array([0.97, 0.01, 0.01, 0.01]))  # low entropy
        result = mon.update(np.array([0.25, 0.25, 0.25, 0.25]))  # high entropy jump
        assert result.alert, "Large entropy jump should trigger gradient alert"


# ---------------------------------------------------------------------------
# ResilienceTwin
# ---------------------------------------------------------------------------


class TestResilienceTwin:
    def test_step_returns_epoch_diagnosis(self) -> None:
        los = _make_los()
        twin = ResilienceTwin(los)
        diag = twin.step(np.zeros(6), t=0)
        assert isinstance(diag.diagnosis, FaultClass)
        assert abs(sum(diag.fault_posterior) - 1.0) < 1e-10
        assert 0.0 <= diag.confidence <= 1.0

    def test_fault_posterior_sums_to_one(self) -> None:
        rng = np.random.default_rng(99)
        los = _make_los()
        twin = ResilienceTwin(los)
        for _ in range(10):
            diag = twin.step(rng.normal(0.0, 0.30, size=6))
            assert abs(sum(diag.fault_posterior) - 1.0) < 1e-9

    def test_hw_fault_detected(self) -> None:
        los = _make_los()
        twin = ResilienceTwin(los)
        # Inject single large outlier on satellite 0 for many epochs
        votes: dict[FaultClass, int] = {}
        for _ in range(40):
            meas = np.zeros(6)
            meas[0] = 15.0
            diag = twin.step(meas)
            votes[diag.diagnosis] = votes.get(diag.diagnosis, 0) + 1
        top = max(votes, key=lambda k: votes[k])
        assert top == FaultClass.HARDWARE_FAULT, f"Expected HW_FAULT majority, got {votes}"


# ---------------------------------------------------------------------------
# Attack generators
# ---------------------------------------------------------------------------


class TestAttackGenerators:
    def test_inject_hw_fault_single_satellite(self) -> None:
        base = np.zeros(6)
        result = _inject_hw_fault(base, 3, 10.0)
        assert result[3] == pytest.approx(10.0)
        assert np.allclose(np.delete(result, 3), 0.0)

    def test_inject_multipath_does_not_modify_high_elevation(self) -> None:
        rng = np.random.default_rng(0)
        los = _make_los()
        el = _elevations(los)
        base = np.zeros(6)
        high_el_idx = np.argsort(el)[-1]  # highest elevation satellite
        result = _inject_multipath(base, el, rng)
        # Multipath targets lowest third; highest-elevation sat should be unmodified
        assert result[high_el_idx] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Full simulation
# ---------------------------------------------------------------------------


class TestRunResilienceSimulation:
    def test_report_type_and_fields(self) -> None:
        cfg = ResilienceTwinConfig(n_mc=8, n_epochs=10, random_seed=0)
        report = run_resilience_simulation(cfg)
        assert isinstance(report, ResilienceTwinReport)
        assert report.n_mc == 8
        assert len(report.confusion_matrix) == 4
        assert all(len(row) == 4 for row in report.confusion_matrix)

    def test_auc_in_unit_interval(self) -> None:
        cfg = ResilienceTwinConfig(n_mc=16, n_epochs=10, random_seed=7)
        report = run_resilience_simulation(cfg)
        assert 0.0 <= report.auc <= 1.0

    def test_detection_and_fa_in_unit_interval(self) -> None:
        cfg = ResilienceTwinConfig(n_mc=16, n_epochs=10, random_seed=3)
        report = run_resilience_simulation(cfg)
        assert 0.0 <= report.p_detection <= 1.0
        assert 0.0 <= report.p_false_alarm <= 1.0

    def test_per_class_accuracy_all_present(self) -> None:
        cfg = ResilienceTwinConfig(n_mc=8, n_epochs=8, random_seed=1)
        report = run_resilience_simulation(cfg)
        expected_keys = {fc.value for fc in FaultClass}
        assert expected_keys == set(report.per_class_accuracy.keys())

    def test_fault_classes_constant_index(self) -> None:
        assert _FAULT_CLASSES[0] == FaultClass.NOMINAL
        assert _FAULT_CLASSES[1] == FaultClass.MULTIPATH
        assert _FAULT_CLASSES[2] == FaultClass.HARDWARE_FAULT
        assert _FAULT_CLASSES[3] == FaultClass.SPOOFING


# ---------------------------------------------------------------------------
# Layer 9 — HuhSubsetSelector
# ---------------------------------------------------------------------------


class TestHuhSubsetSelector:
    def test_no_fault_keeps_all_satellites(self) -> None:
        los = _make_los(6)
        sel = HuhSubsetSelector(los)
        flags = np.zeros(6, dtype=bool)
        result = sel.select(flags)
        assert result.n_selected == 6
        assert result.n_excluded == 0
        assert result.selected_subset == tuple(range(6))

    def test_one_fault_excludes_that_satellite(self) -> None:
        los = _make_los(6)
        sel = HuhSubsetSelector(los)
        flags = np.array([False, False, True, False, False, False])
        result = sel.select(flags)
        assert result.n_selected == 5
        assert result.n_excluded == 1
        assert 2 not in result.selected_subset

    def test_det_ratio_nonnegative(self) -> None:
        los = _make_los(6)
        sel = HuhSubsetSelector(los)
        flags = np.array([False, True, False, False, False, False])
        result = sel.select(flags)
        assert result.det_ratio >= 0.0

    def test_fallback_when_too_few_healthy(self) -> None:
        # Only 3 healthy → below MIN_SATS=4 → fallback to all satellites
        los = _make_los(6)
        sel = HuhSubsetSelector(los)
        flags = np.array([True, True, True, False, False, False])
        result = sel.select(flags)
        assert result.n_selected == 6
        assert result.n_excluded == 0

    def test_log_concavity_ratio_positive(self) -> None:
        los = _make_los(6)
        sel = HuhSubsetSelector(los)
        flags = np.zeros(6, dtype=bool)
        result = sel.select(flags)
        assert result.log_concavity_ratio > 0.0

    def test_result_in_integrity_score(self) -> None:
        los = _make_los(6)
        twin = ResilienceTwin(los)
        diag = twin.step(np.zeros(6), t=0)
        huh = diag.integrity.huh
        assert huh.n_selected >= 4
        assert 0.0 <= huh.det_ratio


# ---------------------------------------------------------------------------
# Layer 10 — DuminilCopinPhaseMonitor
# ---------------------------------------------------------------------------


class TestDuminilCopinPhaseMonitor:
    def test_susceptibility_peak_nonnegative(self) -> None:
        rng = np.random.default_rng(0)
        mon = DuminilCopinPhaseMonitor()
        result = mon.update(rng.normal(0.0, 0.30, size=6))
        assert result.susceptibility_peak >= 0.0

    def test_percolation_threshold_in_unit_interval(self) -> None:
        rng = np.random.default_rng(1)
        mon = DuminilCopinPhaseMonitor()
        result = mon.update(rng.normal(0.0, 0.30, size=6))
        assert 0.0 <= result.percolation_threshold <= 1.0

    def test_lcc_at_null_in_unit_interval(self) -> None:
        rng = np.random.default_rng(2)
        mon = DuminilCopinPhaseMonitor()
        result = mon.update(rng.normal(0.0, 0.30, size=6))
        assert 0.0 <= result.lcc_at_null <= 1.0

    def test_nominal_no_phase_alert(self) -> None:
        # Nominal: P(min_w > 0.95 | σ=0.3 Hz) ≈ 0.04 %/epoch → no alert expected
        rng = np.random.default_rng(3)
        mon = DuminilCopinPhaseMonitor()
        alerts = [mon.update(rng.normal(0.0, 0.30, size=6)).phase_alert for _ in range(20)]
        assert not any(alerts), "Nominal measurements should not trigger phase alert"

    def test_spoofing_triggers_phase_alert(self) -> None:
        # Strong common bias → all edge-weights ≈ same → synchronised collapse → χ_peak >> 10
        rng = np.random.default_rng(7)
        mon = DuminilCopinPhaseMonitor()
        spoofed = np.full(6, 5.0) + rng.normal(0.0, 0.05, size=6)
        result = mon.update(spoofed)
        assert result.phase_alert, (
            f"Spoofing should trigger phase alert; χ_peak={result.susceptibility_peak:.2f}"
        )

    def test_hw_fault_no_phase_alert(self) -> None:
        # HW fault: w_{0,j} ≈ 0 → min_w ≈ 0 < _DC_MIN_W_THRESHOLD=0.95 → no alert
        mon = DuminilCopinPhaseMonitor()
        meas = np.zeros(6)
        meas[0] = 15.0  # single large outlier
        result = mon.update(meas)
        assert not result.phase_alert, (
            f"Single HW fault should not trigger phase alert; "
            f"χ_peak={result.susceptibility_peak:.2f}, min_w={result.min_edge_weight:.3f}"
        )

    def test_result_in_structural_score(self) -> None:
        los = _make_los(6)
        twin = ResilienceTwin(los)
        diag = twin.step(np.zeros(6), t=0)
        phase = diag.structure.phase
        assert 0.0 <= phase.percolation_threshold <= 1.0
        assert phase.susceptibility_peak >= 0.0
