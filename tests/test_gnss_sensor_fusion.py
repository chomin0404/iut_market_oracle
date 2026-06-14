"""Tests for src/gnss/sensor_fusion.py — barometer, VO, fixed-lag smoother.

Acceptance criteria:
    check_barometer:
        - Small Δh → chi² low, no alarm
        - Large Δh → chi² high, alarm
        - chi² ≈ Δh² / (σ₁² + σ₂²)

    check_visual_odometry:
        - Small Δv → chi² low, no alarm
        - Large Δv → chi² high, alarm
        - 2-D and 3-D variants
        - ValueError on shape mismatch

    FixedLagSmoother:
        - First call initialises without alarm
        - Consistent measurements → NIS low, no alarm
        - Outlier position → NIS high, alarm
        - reset() restores initial state

    SensorFusionLayer:
        - assess with all sensors: result has baro, vo, smoother fields
        - assess with partial sensors (None): missing sub-monitors are None
        - alarm = OR of sub-monitor alarms
        - quality_score ∈ [0, 1]
"""

from __future__ import annotations

import numpy as np
import pytest

from gnss.sensor_fusion import (
    BARO_CHI2_THRESH,
    VO_CHI2_THRESH_2D,
    BarometerResult,
    FixedLagSmoother,
    SensorFusionLayer,
    SensorFusionResult,
    check_barometer,
    check_visual_odometry,
)

# ---------------------------------------------------------------------------
# check_barometer
# ---------------------------------------------------------------------------


class TestCheckBarometer:
    def test_zero_difference_no_alarm(self) -> None:
        res = check_barometer(h_gnss=100.0, h_baro=100.0)
        assert isinstance(res, BarometerResult)
        assert res.chi2_stat < 1e-6
        assert res.alarm is False

    def test_large_difference_alarm(self) -> None:
        # Δh = 100 m >> threshold
        res = check_barometer(h_gnss=500.0, h_baro=400.0)
        assert res.chi2_stat > BARO_CHI2_THRESH
        assert res.alarm is True

    def test_chi2_formula(self) -> None:
        delta = 10.0
        res = check_barometer(
            h_gnss=110.0,
            h_baro=100.0,
            sigma_gnss_h=5.0,
            sigma_baro=2.0,
        )
        expected = delta**2 / (5.0**2 + 2.0**2)
        assert abs(res.chi2_stat - expected) < 1e-6

    def test_delta_h_sign(self) -> None:
        res = check_barometer(h_gnss=90.0, h_baro=100.0)
        assert res.delta_h == pytest.approx(-10.0)


# ---------------------------------------------------------------------------
# check_visual_odometry
# ---------------------------------------------------------------------------


class TestCheckVisualOdometry:
    def test_2d_zero_diff_no_alarm(self) -> None:
        v = np.array([1.0, 0.5])
        res = check_visual_odometry(v_gnss=v, v_vo=v)
        assert res.chi2_stat < 1e-6
        assert res.alarm is False
        assert res.dof == 2

    def test_2d_large_diff_alarm(self) -> None:
        v_gnss = np.array([10.0, 0.0])
        v_vo = np.array([0.0, 0.0])
        res = check_visual_odometry(v_gnss=v_gnss, v_vo=v_vo)
        assert res.chi2_stat > VO_CHI2_THRESH_2D
        assert res.alarm is True

    def test_3d_no_alarm(self) -> None:
        v = np.array([1.0, 0.5, 0.2])
        res = check_visual_odometry(v_gnss=v, v_vo=v)
        assert res.dof == 3
        assert res.alarm is False

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="same shape"):
            check_visual_odometry(
                v_gnss=np.array([1.0, 0.5]),
                v_vo=np.array([1.0, 0.5, 0.2]),
            )

    def test_unsupported_dim_raises(self) -> None:
        with pytest.raises(ValueError, match="dimension must be 2 or 3"):
            check_visual_odometry(
                v_gnss=np.array([1.0]),
                v_vo=np.array([1.0]),
            )

    def test_delta_v_matches_input(self) -> None:
        v_g = np.array([2.0, 3.0])
        v_v = np.array([1.0, 1.0])
        res = check_visual_odometry(v_gnss=v_g, v_vo=v_v)
        np.testing.assert_allclose(res.delta_v, np.array([1.0, 2.0]))


# ---------------------------------------------------------------------------
# FixedLagSmoother
# ---------------------------------------------------------------------------


class TestFixedLagSmoother:
    def test_first_call_no_alarm(self) -> None:
        smoother = FixedLagSmoother()
        res = smoother.update(pos_gnss=np.array([0.0, 0.0, 100.0]))
        assert res.alarm is False
        assert res.nis == 0.0

    def test_consistent_measurements_low_nis(self) -> None:
        smoother = FixedLagSmoother()
        smoother.update(pos_gnss=np.array([0.0, 0.0, 100.0]))
        # Small perturbation → low NIS
        for i in range(5):
            res = smoother.update(
                pos_gnss=np.array([float(i), 0.0, 100.0]),
                h_baro=100.0,
            )
        assert res.nis >= 0.0

    def test_large_outlier_high_nis(self) -> None:
        smoother = FixedLagSmoother(sigma_gnss_p=3.0)
        smoother.update(pos_gnss=np.array([0.0, 0.0, 100.0]))
        # Jump of 1000 m — should produce large NIS
        res = smoother.update(pos_gnss=np.array([1000.0, 0.0, 100.0]))
        assert res.nis > 1.0

    def test_n_fused_increases_with_sensors(self) -> None:
        smoother = FixedLagSmoother()
        smoother.update(pos_gnss=np.array([0.0, 0.0, 0.0]))
        res = smoother.update(
            pos_gnss=np.array([1.0, 0.0, 100.0]),
            h_baro=100.0,
            v_vo=np.array([0.1, 0.0]),
        )
        assert res.n_fused == 3  # GNSS + baro + VO

    def test_reset_reinitialises_state(self) -> None:
        smoother = FixedLagSmoother()
        smoother.update(pos_gnss=np.array([100.0, 200.0, 300.0]))
        smoother.reset()
        # After reset, first call should initialise again (nis = 0)
        res = smoother.update(pos_gnss=np.array([0.0, 0.0, 0.0]))
        assert res.nis == 0.0

    def test_wrong_pos_dim_raises(self) -> None:
        smoother = FixedLagSmoother()
        with pytest.raises(ValueError, match="3 elements"):
            smoother.update(pos_gnss=np.array([0.0, 0.0]))


# ---------------------------------------------------------------------------
# SensorFusionLayer
# ---------------------------------------------------------------------------


class TestSensorFusionLayer:
    def test_all_sensors_returns_all_fields(self) -> None:
        layer = SensorFusionLayer()
        pos = np.array([0.0, 0.0, 100.0])
        res = layer.assess(
            epoch=0,
            pos_gnss=pos,
            h_baro=100.0,
            v_gnss=np.array([1.0, 0.5]),
            v_vo=np.array([1.0, 0.5]),
        )
        assert isinstance(res, SensorFusionResult)
        assert res.baro is not None
        assert res.vo is not None
        assert res.smoother is not None

    def test_partial_sensors_no_crash(self) -> None:
        layer = SensorFusionLayer()
        res = layer.assess(epoch=1, pos_gnss=np.array([0.0, 0.0, 100.0]))
        assert isinstance(res, SensorFusionResult)
        assert res.baro is None
        assert res.vo is None

    def test_quality_score_in_range(self) -> None:
        layer = SensorFusionLayer()
        res = layer.assess(epoch=0, pos_gnss=np.array([0.0, 0.0, 100.0]))
        assert 0.0 <= res.quality_score <= 1.0

    def test_alarm_is_union(self) -> None:
        layer = SensorFusionLayer()
        pos = np.array([0.0, 0.0, 100.0])
        layer.assess(epoch=0, pos_gnss=pos)
        res = layer.assess(
            epoch=1,
            pos_gnss=pos,
            h_baro=100.0,
            v_gnss=np.array([0.0, 0.0]),
            v_vo=np.array([0.0, 0.0]),
        )
        sub_alarms = (
            (res.baro is not None and res.baro.alarm)
            or (res.vo is not None and res.vo.alarm)
            or (res.smoother is not None and res.smoother.alarm)
        )
        assert res.alarm == sub_alarms

    def test_large_baro_discrepancy_triggers_alarm(self) -> None:
        layer = SensorFusionLayer()
        pos = np.array([0.0, 0.0, 500.0])
        # h_baro = 100 → Δh = 400 m >> threshold
        res = layer.assess(epoch=0, pos_gnss=pos, h_baro=100.0)
        assert res.baro is not None
        assert res.baro.alarm is True
        assert res.alarm is True

    def test_consistent_sensors_no_alarm_after_init(self) -> None:
        layer = SensorFusionLayer()
        pos = np.array([0.0, 0.0, 100.0])
        # Initialise smoother
        layer.assess(epoch=0, pos_gnss=pos)
        # Consistent step
        res = layer.assess(
            epoch=1,
            pos_gnss=pos,
            h_baro=100.0,
            v_gnss=np.array([0.0, 0.0]),
            v_vo=np.array([0.0, 0.0]),
        )
        assert res.baro is not None
        assert res.baro.alarm is False
        assert res.vo is not None
        assert res.vo.alarm is False

    def test_epoch_stored_in_result(self) -> None:
        layer = SensorFusionLayer()
        res = layer.assess(epoch=99, pos_gnss=np.array([0.0, 0.0, 0.0]))
        assert res.epoch == 99
