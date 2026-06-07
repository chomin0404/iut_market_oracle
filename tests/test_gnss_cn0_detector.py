"""Tests for CN0AnomalyDetector.

Coverage targets
----------------
- Genuine scenario: no alerts
- Spread collapse (meaconing): spread_alert fires
- CUSUM step change: cusum_alert fires after sufficient epochs
- Correlation burst (spoofing takeover): corr_alert fires
- NaN handling: no crash, graceful degradation
- Reset: CUSUM state cleared
- Satellite-count change: state adapts without crash
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from gnss.cn0_detector import (
    _CN0_CORR_MIN_EPOCHS,
    _CN0_WARMUP_EPOCHS,
    _CN0_WINDOW,
    CN0AnomalyDetector,
    CN0AnomalyResult,
    _CN0_NOMINAL_DBHz,
)

RANDOM_SEED: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_genuine_cn0(
    n_sats: int = 6,
    n_epochs: int = 30,
    spread_std: float = 5.0,
    noise_std: float = 0.5,
    seed: int = RANDOM_SEED,
) -> np.ndarray:
    """Generate realistic genuine C/N0 matrix (n_epochs, n_sats).

    Each satellite has a fixed offset drawn from N(40, spread_std²),
    plus small independent per-epoch noise N(0, noise_std²).
    """
    rng = np.random.default_rng(seed)
    offsets = rng.normal(_CN0_NOMINAL_DBHz, spread_std, size=n_sats)
    noise = rng.normal(0.0, noise_std, size=(n_epochs, n_sats))
    return np.clip(offsets[np.newaxis, :] + noise, 20.0, 60.0)


def _make_spoofed_cn0(
    n_sats: int = 6,
    n_epochs: int = 30,
    common_level: float = 40.0,
    noise_std: float = 0.1,
    seed: int = RANDOM_SEED,
) -> np.ndarray:
    """Generate C/N0 matrix simulating meaconing (all sats same level).

    All satellites share the same C/N0 source → spread collapses.
    Pairwise correlation approaches 1.
    """
    rng = np.random.default_rng(seed)
    common_noise = rng.normal(0.0, noise_std, size=(n_epochs, 1))
    return np.full((n_epochs, n_sats), common_level) + common_noise


# ---------------------------------------------------------------------------
# Genuine scenario — no alerts expected
# ---------------------------------------------------------------------------


def test_genuine_no_spread_alert() -> None:
    """Genuine constellation: spread does not collapse.

    Uses spread_std=8.0 to guarantee sigma >> spread_min=2.5 across all seeds.
    Checks post-warmup epochs only.
    """
    det = CN0AnomalyDetector(n_sats=6)
    # spread_std=8 → expected per-epoch sigma ~ 8 >> 2.5
    cn0_mat = _make_genuine_cn0(n_sats=6, n_epochs=30, spread_std=8.0, noise_std=0.3)
    results: list[CN0AnomalyResult] = [det.assess(row) for row in cn0_mat]
    for r in results[_CN0_WARMUP_EPOCHS:]:
        assert not r.spread_alert, f"Unexpected spread alert: sigma={r.sigma_t:.3f}"


def test_genuine_no_alarm() -> None:
    """Genuine constellation: no alarm after warmup completes.

    spread_std=8.0 guarantees spread > spread_min.
    Per-satellite mu0 is estimated during warmup → CUSUM stable post-warmup.
    """
    n_epochs = 40
    det = CN0AnomalyDetector(n_sats=6, corr_thresh=0.99)
    cn0_mat = _make_genuine_cn0(n_sats=6, n_epochs=n_epochs, spread_std=8.0, noise_std=0.3)
    results = [det.assess(row) for row in cn0_mat]
    # Evaluate post-warmup only
    post = results[_CN0_WARMUP_EPOCHS:]
    alarm_count = sum(1 for r in post if r.alarm)
    assert alarm_count == 0, f"Unexpected alarms in genuine scenario: {alarm_count}/{len(post)}"


# ---------------------------------------------------------------------------
# Spread collapse (meaconing)
# ---------------------------------------------------------------------------


def test_spread_collapse_fires() -> None:
    """Meaconing scenario: spread_alert must fire on every epoch."""
    det = CN0AnomalyDetector(n_sats=6)
    cn0_mat = _make_spoofed_cn0(n_sats=6, n_epochs=10, noise_std=0.05)
    results = [det.assess(row) for row in cn0_mat]
    for i, r in enumerate(results):
        assert r.spread_alert, f"Epoch {i}: spread_alert not fired (sigma={r.sigma_t:.3f})"


def test_spread_score_near_one_for_collapse() -> None:
    """Spread score should approach 1 when spread ~ 0."""
    det = CN0AnomalyDetector(n_sats=6)
    # Perfectly identical C/N0 → sigma = 0 → score = 1.0
    cn0 = np.full(6, 40.0)
    r = det.assess(cn0)
    assert r.spread_score == pytest.approx(1.0)


def test_spread_score_zero_for_wide_spread() -> None:
    """Spread score should be 0 when spread >> spread_min."""
    det = CN0AnomalyDetector(n_sats=6, spread_min=2.5)
    # Wide spread: 6 satellites from 30 to 55 dB-Hz → sigma >> 2.5
    cn0 = np.array([30.0, 35.0, 40.0, 45.0, 50.0, 55.0])
    r = det.assess(cn0)
    assert r.spread_score == pytest.approx(0.0)
    assert not r.spread_alert


# ---------------------------------------------------------------------------
# CUSUM step change
# ---------------------------------------------------------------------------


def test_cusum_fires_after_step_change() -> None:
    """CUSUM must alert within a few epochs after a large positive step.

    Warmup establishes baseline at 40 dB-Hz; after warmup a +15 dB-Hz step
    should trigger the alert promptly.
    """
    warmup = _CN0_WARMUP_EPOCHS
    det = CN0AnomalyDetector(n_sats=4, cusum_k=0.5, cusum_h=5.0, warmup_epochs=warmup)
    rng = np.random.default_rng(RANDOM_SEED)

    # warmup_epochs epochs at stable nominal level to calibrate mu0
    for _ in range(warmup):
        det.assess(rng.normal(40.0, 0.3, size=4))

    # 20 epochs with C/N0 shifted up by +15 dB-Hz
    alerted = False
    for _ in range(20):
        cn0 = rng.normal(55.0, 0.5, size=4)
        r = det.assess(cn0)
        if r.cusum_alert:
            alerted = True
            break

    assert alerted, "CUSUM did not fire after 20 epochs of +15 dB-Hz step"


def test_cusum_does_not_fire_at_nominal() -> None:
    """CUSUM should not fire during stable nominal reception after warmup."""
    warmup = _CN0_WARMUP_EPOCHS
    det = CN0AnomalyDetector(n_sats=4, cusum_k=0.5, cusum_h=5.0, warmup_epochs=warmup)
    rng = np.random.default_rng(RANDOM_SEED)
    results = [det.assess(rng.normal(40.0, 0.3, size=4)) for _ in range(50)]
    # Post-warmup only
    assert not any(r.cusum_alert for r in results[warmup:]), "False CUSUM alarm at nominal C/N0"


def test_cusum_reset_clears_state() -> None:
    """After reset(), CUSUM state, warmup buffer, and mu0 are cleared."""
    warmup = _CN0_WARMUP_EPOCHS
    det = CN0AnomalyDetector(n_sats=4, cusum_k=0.5, cusum_h=5.0, warmup_epochs=warmup)
    rng = np.random.default_rng(RANDOM_SEED)

    # Complete warmup, then drive CUSUM high
    for _ in range(warmup):
        det.assess(rng.normal(40.0, 0.3, size=4))
    for _ in range(15):
        det.assess(rng.normal(55.0, 0.3, size=4))

    det.reset()

    # After reset: no alarm, CUSUM zeroed, warmup cleared
    r = det.assess(np.full(4, 40.0))
    assert not r.cusum_alert, "CUSUM alarm immediately after reset (still in warmup)"
    assert np.all(det._cusum_s == 0.0), "CUSUM state not zeroed after reset"
    assert det._mu0_per_sat is None, "mu0_per_sat not cleared after reset"


# ---------------------------------------------------------------------------
# Pairwise correlation burst
# ---------------------------------------------------------------------------


def test_corr_alert_fires_for_spoofed() -> None:
    """Fully coherent C/N0 (meaconing) must trigger corr_alert."""
    n_sats = 6
    det = CN0AnomalyDetector(
        n_sats=n_sats,
        corr_thresh=0.85,
        window=_CN0_WINDOW,
        corr_min_epochs=_CN0_CORR_MIN_EPOCHS,
    )
    cn0_mat = _make_spoofed_cn0(n_sats=n_sats, n_epochs=30, noise_std=0.05)
    results = [det.assess(row) for row in cn0_mat]

    # After min_epochs, correlation test is active; should fire
    late_results = results[_CN0_CORR_MIN_EPOCHS:]
    assert any(r.corr_alert for r in late_results), (
        "Correlation alert never fired for fully coherent spoofed signal"
    )


def test_corr_nan_before_min_epochs() -> None:
    """corr_score and mean_corr should be NaN until min_epochs collected."""
    det = CN0AnomalyDetector(n_sats=4, corr_min_epochs=4)
    rng = np.random.default_rng(RANDOM_SEED)
    for ep in range(3):
        r = det.assess(rng.normal(40.0, 1.0, size=4))
        assert math.isnan(r.corr_score), f"epoch {ep}: corr_score should be NaN"
        assert math.isnan(r.mean_corr), f"epoch {ep}: mean_corr should be NaN"


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------


def test_nan_satellite_does_not_crash() -> None:
    """NaN in one satellite slot must not raise."""
    det = CN0AnomalyDetector(n_sats=6)
    cn0 = np.array([40.0, 38.0, np.nan, 42.0, 39.0, 41.0])
    r = det.assess(cn0)  # must not raise
    assert isinstance(r, CN0AnomalyResult)


def test_all_nan_returns_no_spread_alert() -> None:
    """All-NaN epoch: spread test cannot run → no spread alert."""
    det = CN0AnomalyDetector(n_sats=4)
    cn0 = np.full(4, np.nan)
    r = det.assess(cn0)
    assert not r.spread_alert
    assert math.isnan(r.sigma_t)


# ---------------------------------------------------------------------------
# Satellite count change
# ---------------------------------------------------------------------------


def test_satellite_count_change_adapts() -> None:
    """Changing n_sats between epochs should not raise."""
    det = CN0AnomalyDetector(n_sats=6)
    rng = np.random.default_rng(RANDOM_SEED)

    det.assess(rng.normal(40.0, 1.0, size=6))
    # Simulate dropout to 4 satellites
    r = det.assess(rng.normal(40.0, 1.0, size=4))
    assert isinstance(r, CN0AnomalyResult)
    assert det._n_sats == 4


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------


def test_p_spoof_cn0_range() -> None:
    """p_spoof_cn0 must always be in [0, 1]."""
    det = CN0AnomalyDetector(n_sats=6)
    rng = np.random.default_rng(RANDOM_SEED)
    for _ in range(50):
        cn0 = rng.normal(40.0, 5.0, size=6)
        r = det.assess(cn0)
        assert 0.0 <= r.p_spoof_cn0 <= 1.0


def test_alarm_iff_any_subalert() -> None:
    """alarm must equal cusum_alert | spread_alert | corr_alert."""
    det = CN0AnomalyDetector(n_sats=6)
    cn0_mat = _make_genuine_cn0(n_sats=6, n_epochs=30)
    for row in cn0_mat:
        r = det.assess(row)
        expected = r.cusum_alert or r.spread_alert or r.corr_alert
        assert r.alarm == expected


def test_reasons_nonempty_on_alarm() -> None:
    """When alarm is True, reasons tuple should be non-empty."""
    det = CN0AnomalyDetector(n_sats=6)
    cn0_mat = _make_spoofed_cn0(n_sats=6, n_epochs=10)
    for row in cn0_mat:
        r = det.assess(row)
        if r.alarm:
            assert len(r.reasons) > 0


def test_reasons_empty_no_alarm() -> None:
    """When alarm is False, reasons should be empty."""
    det = CN0AnomalyDetector(n_sats=6, corr_thresh=0.99)
    rng = np.random.default_rng(42)
    for _ in range(30):
        cn0 = rng.normal(40.0, 5.0, size=6)
        r = det.assess(cn0)
        if not r.alarm:
            assert r.reasons == ()
