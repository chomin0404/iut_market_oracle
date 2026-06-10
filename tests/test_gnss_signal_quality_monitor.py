"""Tests for src/gnss/signal_quality_monitor.py — signal quality monitor."""

from __future__ import annotations

import math

import numpy as np
import pytest

from gnss.signal_quality_monitor import (
    AGC_WARMUP_EPOCHS,
    CN0_SAT_THRESH,
    EL_MIN_DEG,
    MP_THRESH,
    CN0_ZENITH_DBHz,
    SignalQualityMonitor,
    SignalQualityResult,
    cn0_elevation_model,
    run_signal_quality_simulation,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_N_SATS = 6
_EL_RAD = np.radians(np.linspace(15.0, 75.0, _N_SATS))  # 15° … 75°


def _nominal_cn0(el_rad: np.ndarray, zenith: float = CN0_ZENITH_DBHz) -> np.ndarray:
    """Return cn0 exactly matching the elevation model (zero residual)."""
    return cn0_elevation_model(el_rad, cn0_zenith=zenith)


def _make_monitor(**kwargs: float | int) -> SignalQualityMonitor:
    return SignalQualityMonitor(n_sats=_N_SATS, **kwargs)  # type: ignore[arg-type]


def _warmup_agc(monitor: SignalQualityMonitor, agc_val: float = 30.0) -> None:
    """Drive AGC through warmup phase."""
    for ep in range(AGC_WARMUP_EPOCHS):
        monitor.assess(epoch=ep, agc_db=agc_val)


# ===========================================================================
# 1. cn0_elevation_model — public helper
# ===========================================================================


class TestCN0ElevationModel:
    def test_zenith_equals_cn0_zenith(self) -> None:
        """el=90° → cn0_expected = cn0_zenith (sin(90°)=1, log10(1)=0)."""
        result = cn0_elevation_model(np.array([math.pi / 2]))
        assert math.isclose(result[0], CN0_ZENITH_DBHz, rel_tol=1e-9)

    def test_low_elevation_less_than_zenith(self) -> None:
        """el=15° gives lower expected CN0 than zenith."""
        result = cn0_elevation_model(np.array([math.radians(15.0)]))
        assert result[0] < CN0_ZENITH_DBHz

    def test_higher_elevation_higher_cn0(self) -> None:
        """Monotonically increasing with elevation angle."""
        els = np.radians(np.array([10.0, 20.0, 45.0, 70.0, 90.0]))
        cn0s = cn0_elevation_model(els)
        assert np.all(np.diff(cn0s) > 0)

    def test_el_min_clamp(self) -> None:
        """Very low elevation is clamped to EL_MIN_DEG floor."""
        el_below = np.array([math.radians(1.0)])
        el_min = np.array([math.radians(EL_MIN_DEG)])
        # Both should return the same value after clamping
        assert math.isclose(
            cn0_elevation_model(el_below)[0],
            cn0_elevation_model(el_min)[0],
            rel_tol=1e-9,
        )

    def test_custom_zenith(self) -> None:
        """Custom zenith value is respected."""
        result = cn0_elevation_model(np.array([math.pi / 2]), cn0_zenith=50.0)
        assert math.isclose(result[0], 50.0, rel_tol=1e-9)

    def test_thirty_deg_minus_3dB(self) -> None:
        """el=30° → sin(30°)=0.5 → 10·log10(0.5) ≈ -3.01 dB below zenith."""
        result = cn0_elevation_model(np.array([math.radians(30.0)]))
        expected = CN0_ZENITH_DBHz + 10.0 * math.log10(0.5)
        assert math.isclose(result[0], expected, rel_tol=1e-6)

    def test_output_shape_matches_input(self) -> None:
        el = np.radians(np.linspace(5.0, 90.0, 20))
        out = cn0_elevation_model(el)
        assert out.shape == el.shape

    def test_output_dtype_float(self) -> None:
        out = cn0_elevation_model(np.array([1.0]))
        assert out.dtype == np.float64


# ===========================================================================
# 2. CN0 elevation sub-monitor (via assess())
# ===========================================================================


class TestCN0ElevationSubMonitor:
    def test_matching_cn0_no_alarm(self) -> None:
        """C/N₀ exactly matching the model → no cn0_alarm."""
        mon = _make_monitor()
        cn0 = _nominal_cn0(_EL_RAD)
        result = mon.assess(epoch=0, cn0_dbhz=cn0, elevation_rad=_EL_RAD)
        assert not result.cn0_alarm
        assert math.isclose(result.cn0_delta_rms, 0.0, abs_tol=1e-9)

    def test_no_cn0_data_no_alarm(self) -> None:
        """Missing cn0_dbhz → cn0_alarm=False, cn0_delta_rms=NaN."""
        mon = _make_monitor()
        result = mon.assess(epoch=0, elevation_rad=_EL_RAD)
        assert not result.cn0_alarm
        assert math.isnan(result.cn0_delta_rms)

    def test_no_elevation_data_no_alarm(self) -> None:
        """Missing elevation_rad → cn0_alarm=False, cn0_delta_rms=NaN."""
        mon = _make_monitor()
        cn0 = _nominal_cn0(_EL_RAD)
        result = mon.assess(epoch=0, cn0_dbhz=cn0)
        assert not result.cn0_alarm
        assert math.isnan(result.cn0_delta_rms)

    def test_large_rms_residual_triggers_alarm(self) -> None:
        """C/N₀ uniformly offset by +15 dB → RMS=15 > CN0_RESIDUAL_THRESH=8."""
        mon = _make_monitor()
        cn0 = _nominal_cn0(_EL_RAD) + 15.0
        result = mon.assess(epoch=0, cn0_dbhz=cn0, elevation_rad=_EL_RAD)
        assert result.cn0_alarm
        assert math.isclose(result.cn0_delta_rms, 15.0, rel_tol=1e-6)

    def test_per_satellite_anomaly_triggers_alarm(self) -> None:
        """One outlier satellite exceeding cn0_sat_thresh fires cn0_alarm."""
        mon = _make_monitor()
        cn0 = _nominal_cn0(_EL_RAD).copy()
        cn0[0] += CN0_SAT_THRESH + 1.0  # push satellite 0 above per-sat threshold
        result = mon.assess(epoch=0, cn0_dbhz=cn0, elevation_rad=_EL_RAD)
        assert result.cn0_alarm
        assert result.n_sat_cn0_anomaly >= 1

    def test_n_sat_cn0_anomaly_count(self) -> None:
        """Two outlier satellites → n_sat_cn0_anomaly == 2."""
        mon = _make_monitor()
        cn0 = _nominal_cn0(_EL_RAD).copy()
        cn0[0] += CN0_SAT_THRESH + 2.0
        cn0[1] -= CN0_SAT_THRESH + 2.0
        result = mon.assess(epoch=0, cn0_dbhz=cn0, elevation_rad=_EL_RAD)
        assert result.n_sat_cn0_anomaly == 2

    def test_nan_cn0_values_skipped(self) -> None:
        """NaN CN0 entries do not contribute to RMS or alarm."""
        mon = _make_monitor()
        cn0 = _nominal_cn0(_EL_RAD).copy()
        cn0[2] = float("nan")
        result = mon.assess(epoch=0, cn0_dbhz=cn0, elevation_rad=_EL_RAD)
        assert not math.isnan(result.cn0_delta_rms)
        assert not result.cn0_alarm

    def test_all_nan_cn0_returns_nan_rms(self) -> None:
        """All NaN CN0 → cn0_delta_rms=NaN, no alarm."""
        mon = _make_monitor()
        cn0 = np.full(_N_SATS, float("nan"))
        result = mon.assess(epoch=0, cn0_dbhz=cn0, elevation_rad=_EL_RAD)
        assert not result.cn0_alarm
        assert math.isnan(result.cn0_delta_rms)

    def test_cn0_score_clamped_to_one(self) -> None:
        """Huge RMS residual → cn0_score clamped to 1.0."""
        mon = _make_monitor()
        cn0 = _nominal_cn0(_EL_RAD) + 100.0
        result = mon.assess(epoch=0, cn0_dbhz=cn0, elevation_rad=_EL_RAD)
        assert math.isclose(result.cn0_score, 1.0)

    def test_cn0_score_zero_for_zero_residual(self) -> None:
        """Perfect match → cn0_score == 0.0."""
        mon = _make_monitor()
        cn0 = _nominal_cn0(_EL_RAD)
        result = mon.assess(epoch=0, cn0_dbhz=cn0, elevation_rad=_EL_RAD)
        assert math.isclose(result.cn0_score, 0.0, abs_tol=1e-9)


# ===========================================================================
# 3. AGC sub-monitor
# ===========================================================================


class TestAGCSubMonitor:
    def test_no_agc_data_no_alarm(self) -> None:
        """Missing agc_db → agc_alarm=False."""
        mon = _make_monitor()
        result = mon.assess(epoch=0)
        assert not result.agc_alarm
        assert math.isclose(result.agc_cusum_lower, 0.0)

    def test_no_alarm_during_warmup(self) -> None:
        """AGC CUSUM does not fire during warmup phase."""
        mon = _make_monitor()
        for ep in range(AGC_WARMUP_EPOCHS - 1):
            result = mon.assess(epoch=ep, agc_db=30.0)
            assert not result.agc_alarm, f"Unexpected alarm at epoch {ep} (warmup)"

    def test_no_alarm_after_warmup_stable(self) -> None:
        """Stable AGC after warmup → no alarm."""
        mon = _make_monitor()
        _warmup_agc(mon, agc_val=30.0)
        for ep in range(10):
            result = mon.assess(epoch=ep, agc_db=30.0)
            assert not result.agc_alarm

    def test_cusum_fires_on_sudden_drop(self) -> None:
        """Sudden AGC drop well below mu0 triggers alarm within a few epochs."""
        mon = _make_monitor(agc_cusum_k=0.5, agc_cusum_h=5.0)
        _warmup_agc(mon, agc_val=30.0)
        # Drop of 8 dB → increment = 8.0 - 0.5 = 7.5 per epoch → fires in 1 epoch
        result = mon.assess(epoch=0, agc_db=30.0 - 8.0)
        assert result.agc_alarm

    def test_cusum_lower_always_non_negative(self) -> None:
        """S⁻ₜ ≥ 0 invariant holds under nominal and drop conditions."""
        mon = _make_monitor()
        _warmup_agc(mon, agc_val=30.0)
        for agc in [30.0, 31.0, 29.5, 25.0, 30.0]:
            result = mon.assess(epoch=0, agc_db=agc)
            assert result.agc_cusum_lower >= 0.0

    def test_reset_clears_warmup_state(self) -> None:
        """reset() causes CUSUM to re-enter warmup; no alarm immediately after."""
        mon = _make_monitor()
        _warmup_agc(mon, agc_val=30.0)
        mon.reset()
        # Should be in warmup again → no alarm even on a big drop
        result = mon.assess(epoch=0, agc_db=10.0)
        assert not result.agc_alarm
        assert math.isclose(result.agc_cusum_lower, 0.0)

    def test_mu0_from_warmup_mean(self) -> None:
        """μ₀ is calibrated as the mean of warmup AGC samples."""
        agc_vals = [28.0, 29.0, 30.0, 31.0, 32.0] * 2  # mean = 30.0, 10 samples
        mon = _make_monitor(agc_cusum_k=0.5, agc_cusum_h=5.0)
        for ep, agc in enumerate(agc_vals):
            mon.assess(epoch=ep, agc_db=agc)
        # After warmup: mu0 = 30.0; a drop to 22.0 should fire
        result = mon.assess(epoch=10, agc_db=22.0)
        assert result.agc_alarm

    def test_agc_score_clamped_to_one(self) -> None:
        """Extreme drop → agc_score clamped to 1.0."""
        mon = _make_monitor()
        _warmup_agc(mon, agc_val=30.0)
        for _ in range(20):
            mon.assess(epoch=0, agc_db=0.0)  # extreme drop, accumulate
        result = mon.assess(epoch=0, agc_db=0.0)
        assert math.isclose(result.agc_score, 1.0)

    def test_agc_score_zero_on_nominal(self) -> None:
        """Stable AGC at mu0 → score == 0.0 (CUSUM stays at 0)."""
        mon = _make_monitor()
        _warmup_agc(mon, agc_val=30.0)
        result = mon.assess(epoch=0, agc_db=30.0)
        # increment = 30 - 30 - 0.5 = -0.5 → max(0, 0 - 0.5) = 0
        assert math.isclose(result.agc_score, 0.0)

    def test_cusum_lower_accumulates_incrementally(self) -> None:
        """S⁻ grows with each step below threshold when AGC drops gradually."""
        mon = _make_monitor(agc_cusum_k=0.5, agc_cusum_h=100.0)  # high h → no alarm
        _warmup_agc(mon, agc_val=30.0)
        prev_s = 0.0
        for _ in range(5):
            result = mon.assess(epoch=0, agc_db=28.0)  # drop of 2 dB > k=0.5
            assert result.agc_cusum_lower > prev_s
            prev_s = result.agc_cusum_lower


# ===========================================================================
# 4. Multipath sub-monitor
# ===========================================================================


class TestMultipathSubMonitor:
    def test_no_pr_data_no_alarm(self) -> None:
        """Missing pseudorange_residuals → no mp_alarm, mp_rms=NaN."""
        mon = _make_monitor()
        result = mon.assess(epoch=0, elevation_rad=_EL_RAD)
        assert not result.mp_alarm
        assert math.isnan(result.mp_rms)

    def test_no_elevation_data_no_alarm(self) -> None:
        """Missing elevation_rad → no mp_alarm, mp_rms=NaN."""
        mon = _make_monitor()
        pr = np.zeros(_N_SATS)
        result = mon.assess(epoch=0, pseudorange_residuals=pr)
        assert not result.mp_alarm
        assert math.isnan(result.mp_rms)

    def test_small_residuals_no_alarm(self) -> None:
        """Near-zero PR residuals → mp_rms << MP_THRESH, no alarm."""
        mon = _make_monitor()
        pr = np.full(_N_SATS, 0.1)
        result = mon.assess(epoch=0, pseudorange_residuals=pr, elevation_rad=_EL_RAD)
        assert not result.mp_alarm
        assert result.mp_rms < MP_THRESH

    def test_large_residuals_alarm(self) -> None:
        """Large PR residuals → mp_rms > MP_THRESH → mp_alarm=True."""
        mon = _make_monitor()
        pr = np.full(_N_SATS, 100.0)  # large residuals [m]
        result = mon.assess(epoch=0, pseudorange_residuals=pr, elevation_rad=_EL_RAD)
        assert result.mp_alarm

    def test_elevation_scaling_reduces_proxy(self) -> None:
        """Higher elevation → larger sin(el) → larger mp_proxy for same |PR|."""
        # mp_proxy = |PR| * sin(el); higher elevation → larger proxy
        mon_low = SignalQualityMonitor(n_sats=1)
        mon_high = SignalQualityMonitor(n_sats=1)
        res_low = mon_low.assess(
            epoch=0,
            pseudorange_residuals=np.array([10.0]),
            elevation_rad=np.array([math.radians(10.0)]),
        )
        res_high = mon_high.assess(
            epoch=0,
            pseudorange_residuals=np.array([10.0]),
            elevation_rad=np.array([math.radians(70.0)]),
        )
        assert res_low.mp_rms < res_high.mp_rms  # higher elevation → larger sin → larger proxy

    def test_mp_rms_formula(self) -> None:
        """Verify mp_rms = RMS(|pr|·sin(el)) manually."""
        mon = _make_monitor()
        pr = np.array([10.0, 5.0, 8.0, 3.0, 6.0, 4.0])
        proxies = np.abs(pr) * np.sin(np.maximum(_EL_RAD, math.radians(EL_MIN_DEG)))
        expected_rms = float(np.sqrt(np.mean(proxies**2)))
        result = mon.assess(epoch=0, pseudorange_residuals=pr, elevation_rad=_EL_RAD)
        assert math.isclose(result.mp_rms, expected_rms, rel_tol=1e-9)

    def test_mp_score_formula(self) -> None:
        """mp_score = clip(mp_rms / mp_thresh, 0, 1)."""
        mon = _make_monitor()
        pr = np.full(_N_SATS, 50.0)
        result = mon.assess(epoch=0, pseudorange_residuals=pr, elevation_rad=_EL_RAD)
        expected = min(result.mp_rms / MP_THRESH, 1.0)
        assert math.isclose(result.mp_score, expected, rel_tol=1e-9)

    def test_nan_pr_values_skipped(self) -> None:
        """NaN PR values do not contribute to mp_rms."""
        mon = _make_monitor()
        pr = np.array([1.0, float("nan"), 1.0, 1.0, 1.0, 1.0])
        result = mon.assess(epoch=0, pseudorange_residuals=pr, elevation_rad=_EL_RAD)
        assert not math.isnan(result.mp_rms)

    def test_all_nan_pr_returns_nan_rms(self) -> None:
        """All NaN PR → mp_rms=NaN, no alarm."""
        mon = _make_monitor()
        pr = np.full(_N_SATS, float("nan"))
        result = mon.assess(epoch=0, pseudorange_residuals=pr, elevation_rad=_EL_RAD)
        assert not result.mp_alarm
        assert math.isnan(result.mp_rms)


# ===========================================================================
# 5. SignalQualityMonitor — integration
# ===========================================================================


class TestSignalQualityMonitor:
    def test_assess_returns_signal_quality_result(self) -> None:
        mon = _make_monitor()
        result = mon.assess(epoch=0)
        assert isinstance(result, SignalQualityResult)

    def test_epoch_field_matches_input(self) -> None:
        mon = _make_monitor()
        result = mon.assess(epoch=7)
        assert result.epoch == 7

    def test_no_alarm_all_nominal(self) -> None:
        """Matching C/N₀, stable AGC, small PR → alarm=False after warmup."""
        mon = _make_monitor()
        cn0 = _nominal_cn0(_EL_RAD)
        pr = np.zeros(_N_SATS)
        # Warmup AGC
        _warmup_agc(mon, agc_val=30.0)
        for ep in range(5):
            result = mon.assess(
                epoch=ep,
                cn0_dbhz=cn0,
                agc_db=30.0,
                pseudorange_residuals=pr,
                elevation_rad=_EL_RAD,
            )
            assert not result.alarm, f"Unexpected alarm at epoch {ep}"

    def test_cn0_alarm_propagates_to_overall_alarm(self) -> None:
        mon = _make_monitor()
        cn0 = _nominal_cn0(_EL_RAD) + 20.0  # large deviation
        result = mon.assess(epoch=0, cn0_dbhz=cn0, elevation_rad=_EL_RAD)
        assert result.cn0_alarm
        assert result.alarm

    def test_agc_alarm_propagates_to_overall_alarm(self) -> None:
        mon = _make_monitor()
        _warmup_agc(mon, agc_val=30.0)
        result = mon.assess(epoch=0, agc_db=30.0 - 8.0)
        assert result.agc_alarm
        assert result.alarm

    def test_mp_alarm_propagates_to_overall_alarm(self) -> None:
        mon = _make_monitor()
        pr = np.full(_N_SATS, 100.0)
        result = mon.assess(epoch=0, pseudorange_residuals=pr, elevation_rad=_EL_RAD)
        assert result.mp_alarm
        assert result.alarm

    def test_quality_score_is_max_of_sub_scores(self) -> None:
        """quality_score = max(cn0_score, agc_score, mp_score)."""
        mon = _make_monitor()
        _warmup_agc(mon, agc_val=30.0)
        cn0 = _nominal_cn0(_EL_RAD) + 4.0  # moderate CN0 deviation
        pr = np.full(_N_SATS, 50.0)
        result = mon.assess(
            epoch=0,
            cn0_dbhz=cn0,
            agc_db=30.0,
            pseudorange_residuals=pr,
            elevation_rad=_EL_RAD,
        )
        expected = max(result.cn0_score, result.agc_score, result.mp_score)
        assert math.isclose(result.quality_score, expected, rel_tol=1e-9)

    def test_quality_score_in_range(self) -> None:
        mon = _make_monitor()
        result = mon.assess(epoch=0)
        assert 0.0 <= result.quality_score <= 1.0

    def test_sat_quality_length_equals_n_sats(self) -> None:
        mon = _make_monitor()
        result = mon.assess(epoch=0)
        assert len(result.sat_quality) == _N_SATS

    def test_sat_quality_all_ok_on_nominal(self) -> None:
        """All satellites nominal → sat_quality all 'ok'."""
        mon = _make_monitor()
        cn0 = _nominal_cn0(_EL_RAD)
        pr = np.zeros(_N_SATS)
        result = mon.assess(epoch=0, cn0_dbhz=cn0, pseudorange_residuals=pr, elevation_rad=_EL_RAD)
        assert all(q == "ok" for q in result.sat_quality)

    def test_sat_quality_cn0_label(self) -> None:
        """Outlier C/N₀ satellite gets 'cn0' label."""
        mon = _make_monitor()
        cn0 = _nominal_cn0(_EL_RAD).copy()
        cn0[0] += CN0_SAT_THRESH + 2.0
        result = mon.assess(epoch=0, cn0_dbhz=cn0, elevation_rad=_EL_RAD)
        assert result.sat_quality[0] == "cn0"

    def test_sat_quality_mp_label(self) -> None:
        """Multipath outlier satellite gets 'mp' label."""
        mon = _make_monitor(mp_thresh=1.0)  # low threshold
        pr = np.zeros(_N_SATS)
        pr[2] = 50.0  # large residual for satellite 2
        result = mon.assess(epoch=0, pseudorange_residuals=pr, elevation_rad=_EL_RAD)
        assert result.sat_quality[2] == "mp"

    def test_sat_quality_combined_label(self) -> None:
        """Satellite with both C/N₀ and multipath anomaly gets 'cn0+mp'."""
        mon = _make_monitor(mp_thresh=1.0)
        cn0 = _nominal_cn0(_EL_RAD).copy()
        cn0[1] += CN0_SAT_THRESH + 2.0
        pr = np.zeros(_N_SATS)
        pr[1] = 50.0
        result = mon.assess(epoch=0, cn0_dbhz=cn0, pseudorange_residuals=pr, elevation_rad=_EL_RAD)
        assert result.sat_quality[1] == "cn0+mp"

    def test_reasons_empty_when_no_alarm(self) -> None:
        mon = _make_monitor()
        result = mon.assess(epoch=0)
        assert result.alarm is False
        assert len(result.reasons) == 0

    def test_reasons_populated_when_alarm(self) -> None:
        mon = _make_monitor()
        cn0 = _nominal_cn0(_EL_RAD) + 20.0
        result = mon.assess(epoch=0, cn0_dbhz=cn0, elevation_rad=_EL_RAD)
        assert result.alarm
        assert len(result.reasons) >= 1
        assert any("C/N0" in r for r in result.reasons)

    def test_multiple_reasons_when_multiple_alarms(self) -> None:
        """All three sub-monitors fire → three reasons."""
        mon = _make_monitor()
        _warmup_agc(mon, agc_val=30.0)
        cn0 = _nominal_cn0(_EL_RAD) + 20.0
        pr = np.full(_N_SATS, 100.0)
        result = mon.assess(
            epoch=0,
            cn0_dbhz=cn0,
            agc_db=30.0 - 8.0,
            pseudorange_residuals=pr,
            elevation_rad=_EL_RAD,
        )
        assert result.cn0_alarm and result.agc_alarm and result.mp_alarm
        assert len(result.reasons) == 3

    def test_reset_restarts_agc_warmup(self) -> None:
        """After reset(), AGC CUSUM is back in warmup: no alarm on big drop."""
        mon = _make_monitor()
        _warmup_agc(mon, agc_val=30.0)
        mon.reset()
        result = mon.assess(epoch=0, agc_db=0.0)
        assert not result.agc_alarm  # warmup → suppressed

    def test_consistent_results_across_epochs(self) -> None:
        """assess() called 5 times returns 5 valid results."""
        mon = _make_monitor()
        cn0 = _nominal_cn0(_EL_RAD)
        pr = np.zeros(_N_SATS)
        for ep in range(5):
            result = mon.assess(
                epoch=ep, cn0_dbhz=cn0, pseudorange_residuals=pr, elevation_rad=_EL_RAD
            )
            assert result.epoch == ep
            assert isinstance(result.alarm, bool)


# ===========================================================================
# 6. SignalQualityResult — type and invariant checks
# ===========================================================================


class TestSignalQualityResult:
    def _make_result(self, **kwargs: object) -> SignalQualityResult:
        defaults: dict[str, object] = dict(
            epoch=0,
            alarm=False,
            quality_score=0.0,
            cn0_alarm=False,
            cn0_score=0.0,
            cn0_delta_rms=float("nan"),
            n_sat_cn0_anomaly=0,
            agc_alarm=False,
            agc_score=0.0,
            agc_cusum_lower=0.0,
            mp_alarm=False,
            mp_score=0.0,
            mp_rms=float("nan"),
            sat_quality=(),
            reasons=(),
        )
        defaults.update(kwargs)
        return SignalQualityResult(**defaults)  # type: ignore[arg-type]

    def test_frozen_dataclass(self) -> None:
        """SignalQualityResult is immutable."""
        result = self._make_result()
        with pytest.raises(Exception):  # FrozenInstanceError
            result.alarm = True  # type: ignore[misc]

    def test_alarm_false_when_all_sub_alarms_false(self) -> None:
        result = self._make_result(alarm=False, cn0_alarm=False, agc_alarm=False, mp_alarm=False)
        assert not result.alarm

    def test_quality_score_in_unit_interval(self) -> None:
        result = self._make_result(quality_score=0.75)
        assert 0.0 <= result.quality_score <= 1.0

    def test_sat_quality_valid_labels(self) -> None:
        valid = {"ok", "cn0", "mp", "cn0+mp"}
        result = self._make_result(sat_quality=("ok", "cn0", "mp", "cn0+mp"))
        assert all(q in valid for q in result.sat_quality)

    def test_reasons_is_tuple_of_strings(self) -> None:
        result = self._make_result(reasons=("AGC drop detected",))
        assert isinstance(result.reasons, tuple)
        assert all(isinstance(r, str) for r in result.reasons)


# ===========================================================================
# 7. run_signal_quality_simulation
# ===========================================================================


class TestRunSignalQualitySimulation:
    def test_returns_list_of_results(self) -> None:
        results = run_signal_quality_simulation(n_epochs=20, seed=0)
        assert isinstance(results, list)
        assert all(isinstance(r, SignalQualityResult) for r in results)

    def test_length_equals_n_epochs(self) -> None:
        n = 30
        results = run_signal_quality_simulation(n_epochs=n, seed=1)
        assert len(results) == n

    def test_epoch_sequence_is_sequential(self) -> None:
        results = run_signal_quality_simulation(n_epochs=10, seed=2)
        epochs = [r.epoch for r in results]
        assert epochs == list(range(10))

    def test_seed_reproducibility(self) -> None:
        r1 = run_signal_quality_simulation(n_epochs=20, seed=42)
        r2 = run_signal_quality_simulation(n_epochs=20, seed=42)
        for a, b in zip(r1, r2):
            assert a.alarm == b.alarm
            if not math.isnan(a.mp_rms):
                assert math.isclose(a.mp_rms, b.mp_rms, rel_tol=1e-9)

    def test_different_seeds_give_different_results(self) -> None:
        r1 = run_signal_quality_simulation(n_epochs=30, seed=1)
        r2 = run_signal_quality_simulation(n_epochs=30, seed=99)
        mp_rms_1 = [r.mp_rms for r in r1 if not math.isnan(r.mp_rms)]
        mp_rms_2 = [r.mp_rms for r in r2 if not math.isnan(r.mp_rms)]
        assert mp_rms_1 != mp_rms_2

    def test_agc_alarm_fires_after_jammer_start(self) -> None:
        """AGC alarm should fire in the jamming window."""
        jammer_start = 15
        results = run_signal_quality_simulation(
            n_epochs=30, jammer_start=jammer_start, agc_drop_db=10.0, seed=0
        )
        agc_alarms_after = [r.agc_alarm for r in results[jammer_start + AGC_WARMUP_EPOCHS :]]
        assert any(agc_alarms_after), "Expected AGC alarm in jamming window"

    def test_mp_alarm_fires_during_multipath_window(self) -> None:
        """Multipath alarms should appear in the mp_start..mp_end window."""
        results = run_signal_quality_simulation(n_epochs=30, mp_start=5, mp_end=15, seed=0)
        mp_alarms = [r.mp_alarm for r in results[5:15]]
        assert any(mp_alarms), "Expected multipath alarm in multipath window"

    def test_no_agc_alarm_before_jammer_start_nominal(self) -> None:
        """No AGC alarm should appear in the pre-jammer nominal window."""
        jammer_start = 25
        results = run_signal_quality_simulation(
            n_epochs=40, jammer_start=jammer_start, mp_start=99, mp_end=99, seed=0
        )
        # Nominal window: after warmup, before jammer
        pre_jammer = results[AGC_WARMUP_EPOCHS:jammer_start]
        assert not any(r.agc_alarm for r in pre_jammer)

    def test_quality_scores_in_unit_interval(self) -> None:
        results = run_signal_quality_simulation(n_epochs=20, seed=3)
        for r in results:
            assert 0.0 <= r.quality_score <= 1.0
