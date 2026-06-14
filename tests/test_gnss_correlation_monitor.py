"""Tests for src/gnss/correlation_monitor.py.

Acceptance criteria:
    E-L ratio sub-monitor:
        - Symmetric P_early == P_late → el_rms ≈ 0, no alarm
        - Large asymmetry → el_rms > EL_RMS_THRESH → alarm
        - Per-satellite flags match individual deviations

    GLRT sub-monitor:
        - Stationary window → glrt_stat low, no alarm
        - Step-change in C/N₀ → glrt_stat high → alarm after window fills

    Integration:
        - alarm = el_alarm OR glrt_alarm
        - quality_score ∈ [0, 1]
        - reset() clears GLRT history
"""

from __future__ import annotations

import numpy as np

from gnss.correlation_monitor import (
    EL_RMS_THRESH,
    GLRT_THRESH,
    GLRT_WINDOW,
    CorrelationMonitor,
    CorrelationMonitorResult,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_N_SATS = 8


def _symmetric_powers(n: int = _N_SATS, power: float = 1.0) -> tuple[list[float], list[float]]:
    return [power] * n, [power] * n


def _asymmetric_powers(n: int = _N_SATS) -> tuple[list[float], list[float]]:
    # Early ≈ 2× Late → ratio ≈ 2 → deviation ≈ 1 >> EL_RMS_THRESH
    p_early = [2.0] * n
    p_late = [1.0] * n
    return p_early, p_late


# ---------------------------------------------------------------------------
# E-L ratio sub-monitor
# ---------------------------------------------------------------------------


class TestELRatio:
    def test_symmetric_no_alarm(self) -> None:
        monitor = CorrelationMonitor()
        p_e, p_l = _symmetric_powers()
        res = monitor.assess(epoch=0, p_early=p_e, p_late=p_l, cn0_db=40.0)
        assert abs(res.el_rms) < 1e-9
        assert res.el_alarm is False

    def test_asymmetric_alarm(self) -> None:
        monitor = CorrelationMonitor()
        p_e, p_l = _asymmetric_powers()
        res = monitor.assess(epoch=0, p_early=p_e, p_late=p_l, cn0_db=40.0)
        assert res.el_rms > EL_RMS_THRESH
        assert res.el_alarm is True

    def test_per_satellite_flags_false_when_symmetric(self) -> None:
        monitor = CorrelationMonitor()
        p_e, p_l = _symmetric_powers()
        res = monitor.assess(epoch=0, p_early=p_e, p_late=p_l, cn0_db=40.0)
        assert all(not f for f in res.el_sat_flags)

    def test_per_satellite_flags_true_when_asymmetric(self) -> None:
        monitor = CorrelationMonitor(el_sat_thresh=0.2)
        p_e, p_l = _asymmetric_powers()
        res = monitor.assess(epoch=0, p_early=p_e, p_late=p_l, cn0_db=40.0)
        assert all(res.el_sat_flags)

    def test_el_rms_is_nonnegative(self) -> None:
        monitor = CorrelationMonitor()
        rng = np.random.default_rng(0)
        p_e = rng.uniform(0.5, 1.5, _N_SATS).tolist()
        p_l = rng.uniform(0.5, 1.5, _N_SATS).tolist()
        res = monitor.assess(0, p_e, p_l, cn0_db=38.0)
        assert res.el_rms >= 0.0

    def test_returns_correct_epoch(self) -> None:
        monitor = CorrelationMonitor()
        p_e, p_l = _symmetric_powers()
        res = monitor.assess(epoch=42, p_early=p_e, p_late=p_l, cn0_db=40.0)
        assert res.epoch == 42


# ---------------------------------------------------------------------------
# GLRT sub-monitor
# ---------------------------------------------------------------------------


class TestGLRT:
    def test_stationary_no_alarm(self) -> None:
        monitor = CorrelationMonitor(glrt_window=GLRT_WINDOW, glrt_thresh=GLRT_THRESH)
        p_e, p_l = _symmetric_powers()
        rng = np.random.default_rng(1)
        res = None
        for i in range(GLRT_WINDOW):
            cn0 = 40.0 + float(rng.normal(0, 0.1))
            res = monitor.assess(i, p_e, p_l, cn0_db=cn0)
        assert res is not None
        assert res.glrt_alarm is False

    def test_step_change_triggers_alarm(self) -> None:
        monitor = CorrelationMonitor(glrt_window=20, glrt_thresh=6.635)
        p_e, p_l = _symmetric_powers()
        # First 15 epochs: stable at 40 dB-Hz
        for i in range(15):
            monitor.assess(i, p_e, p_l, cn0_db=40.0)
        # Next 5 epochs: large drop to 20 dB-Hz (spoofing / interference)
        res = None
        for i in range(15, 20):
            res = monitor.assess(i, p_e, p_l, cn0_db=20.0)
        assert res is not None
        assert res.glrt_stat > GLRT_THRESH
        assert res.glrt_alarm is True

    def test_reset_clears_history(self) -> None:
        monitor = CorrelationMonitor(glrt_window=20, glrt_thresh=6.635)
        p_e, p_l = _symmetric_powers()
        # Fill with a step change to trigger alarm
        for i in range(15):
            monitor.assess(i, p_e, p_l, cn0_db=40.0)
        for i in range(15, 20):
            monitor.assess(i, p_e, p_l, cn0_db=10.0)
        monitor.reset()
        # After reset, first epoch should not alarm
        res = monitor.assess(0, p_e, p_l, cn0_db=40.0)
        assert res.glrt_alarm is False

    def test_insufficient_history_no_alarm(self) -> None:
        monitor = CorrelationMonitor(glrt_window=GLRT_WINDOW)
        p_e, p_l = _symmetric_powers()
        # Only 3 epochs — too few for GLRT
        for i in range(3):
            res = monitor.assess(i, p_e, p_l, cn0_db=40.0)
        assert res.glrt_stat == 0.0
        assert res.glrt_alarm is False


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


class TestCorrelationMonitorIntegration:
    def test_quality_score_in_range(self) -> None:
        monitor = CorrelationMonitor()
        p_e, p_l = _asymmetric_powers()
        res = monitor.assess(0, p_e, p_l, cn0_db=40.0)
        assert 0.0 <= res.quality_score <= 1.0

    def test_alarm_is_union_of_sub_monitors(self) -> None:
        monitor = CorrelationMonitor()
        p_e, p_l = _asymmetric_powers()
        res = monitor.assess(0, p_e, p_l, cn0_db=40.0)
        expected = res.el_alarm or res.glrt_alarm
        assert res.alarm == expected

    def test_reasons_populated_when_alarmed(self) -> None:
        monitor = CorrelationMonitor()
        p_e, p_l = _asymmetric_powers()
        res = monitor.assess(0, p_e, p_l, cn0_db=40.0)
        if res.el_alarm:
            assert any("el_rms" in r for r in res.reasons)

    def test_no_reasons_when_no_alarm(self) -> None:
        monitor = CorrelationMonitor()
        p_e, p_l = _symmetric_powers()
        res = monitor.assess(0, p_e, p_l, cn0_db=40.0)
        if not res.alarm:
            assert res.reasons == []

    def test_result_type(self) -> None:
        monitor = CorrelationMonitor()
        p_e, p_l = _symmetric_powers()
        res = monitor.assess(0, p_e, p_l, cn0_db=40.0)
        assert isinstance(res, CorrelationMonitorResult)
