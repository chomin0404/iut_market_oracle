"""Signal Quality Monitor — real-time C/N₀, AGC, and multipath analysis.

Three independent sub-monitors run per epoch and fuse into a single result:

  Sub-monitor 1 — C/N₀ elevation model residual
      Compares measured C/N₀ against an elevation-dependent physical baseline.
      Model:  cn0_expected(el) = cn0_zenith + 10 · log₁₀(sin(el))
              Received power scales with sin(el) because the effective antenna
              cross-section projected onto the satellite direction is proportional
              to sin(el); hence CN0_linear ∝ sin(el) → CN0_dBHz = zenith + 10log10(sin).
      Residual: δᵢ = CN0_measured_i − CN0_expected(elᵢ)   [dB-Hz]
      Ensemble RMS alert: RMS(δ) > cn0_residual_thresh
      Per-satellite alert: |δᵢ| > cn0_sat_thresh
      Ref: Langley (1997) GPS World §3.2; Groves (2013) §9.1.

  Sub-monitor 2 — AGC drop detector (lower-side CUSUM)
      An in-band jammer raises the RF noise floor; the receiver AGC reduces gain
      to maintain a constant IF output level.  A sudden AGC *drop* therefore
      signals wideband jamming or interference.
      Statistic (lower-side CUSUM, Page 1954):
          S⁻ₜ = max(0, S⁻ₜ₋₁ + (μ₀ − xₜ − k))
      Alert when S⁻ₜ > h.  μ₀ calibrated over the first agc_warmup_epochs epochs.
      Ref: Page (1954); Montgomery (2009) §9.1.

  Sub-monitor 3 — Multipath index (elevation-normalised PR residual)
      Code pseudorange residuals grow with multipath.  Elevation normalisation
      removes the 1/sin(el) geometric component, isolating the multipath effect.
          mp_proxyᵢ = |PR_residualᵢ| · sin(elᵢ)
          mp_rms    = RMS(mp_proxy)   [m]
      Alert when mp_rms > mp_thresh.
      Ref: Groves (2013) §9.3; Nee & Coenen (1993).

Fusion:
    alarm         = cn0_alarm OR agc_alarm OR mp_alarm
    quality_score = max(cn0_score, agc_score, mp_score)   ∈ [0, 1]

Per-satellite quality labels (sat_quality):
    "ok"     — no anomaly detected
    "cn0"    — C/N₀ residual exceeds per-satellite threshold
    "mp"     — elevation-normalised PR residual exceeds per-satellite threshold
    "cn0+mp" — both C/N₀ and multipath anomalies simultaneously present

SignalQualityResult is compatible with ResilienceTwin / MVPPipeline:
    alarm and reasons mirror CN0AnomalyResult conventions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

CN0_ZENITH_DBHz: float = 45.0  # GPS L1 C/A nominal zenith C/N₀ [dB-Hz]
CN0_RESIDUAL_THRESH: float = 8.0  # RMS(δCN0) ensemble alarm threshold [dB-Hz]
CN0_SAT_THRESH: float = 12.0  # per-satellite |δCN0| alarm threshold [dB-Hz]
EL_MIN_DEG: float = 5.0  # elevation floor for model evaluation [degrees]

AGC_WARMUP_EPOCHS: int = 10  # warmup length before AGC CUSUM fires
AGC_CUSUM_K: float = 0.5  # CUSUM allowance (slack) parameter k [dB]
AGC_CUSUM_H: float = 5.0  # CUSUM decision threshold h [dB]

MP_THRESH: float = 5.0  # mp_rms ensemble alarm threshold [m]
_MP_SAT_FACTOR: float = 1.5  # per-satellite mp threshold = MP_THRESH × factor

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SignalQualityResult:
    """Per-epoch signal quality assessment.

    Fields
    ------
    epoch           Epoch index passed to assess().
    alarm           True when any sub-monitor fires.
    quality_score   max(cn0_score, agc_score, mp_score) ∈ [0, 1].

    cn0_alarm           C/N₀ ensemble RMS OR per-satellite anomaly alert.
    cn0_score           clip(RMS(δCN0) / cn0_residual_thresh, 0, 1).
    cn0_delta_rms       RMS of per-satellite (measured − predicted) [dB-Hz].
                        NaN when cn0_dbhz or elevation_rad is absent.
    n_sat_cn0_anomaly   Number of satellites with |δ| > cn0_sat_thresh.

    agc_alarm       Lower-side CUSUM threshold crossed.
    agc_score       clip(S⁻ₜ / agc_h, 0, 1).
    agc_cusum_lower Current S⁻ₜ value.

    mp_alarm        mp_rms > mp_thresh.
    mp_score        clip(mp_rms / mp_thresh, 0, 1).
    mp_rms          Elevation-normalised PR residual RMS [m].
                    NaN when pseudorange_residuals or elevation_rad is absent.

    sat_quality     Per-satellite label tuple: "ok" | "cn0" | "mp" | "cn0+mp".
                    Length always equals n_sats supplied at construction.
    reasons         Human-readable descriptions of active alerts (empty when alarm=False).
    """

    epoch: int
    alarm: bool
    quality_score: float

    cn0_alarm: bool
    cn0_score: float
    cn0_delta_rms: float
    n_sat_cn0_anomaly: int

    agc_alarm: bool
    agc_score: float
    agc_cusum_lower: float

    mp_alarm: bool
    mp_score: float
    mp_rms: float

    sat_quality: tuple[str, ...]
    reasons: tuple[str, ...]


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------


class SignalQualityMonitor:
    """Stateful per-epoch signal quality monitor.

    Parameters
    ----------
    n_sats : int
        Number of tracked satellites.  Determines sat_quality tuple length.
    cn0_zenith : float
        Nominal zenith C/N₀ [dB-Hz] for the elevation model.
    cn0_residual_thresh : float
        Ensemble RMS(δCN0) alarm threshold [dB-Hz].
    cn0_sat_thresh : float
        Per-satellite |δCN0| alarm threshold [dB-Hz].
    agc_warmup_epochs : int
        Warmup length.  AGC CUSUM does not fire during warmup.
    agc_cusum_k : float
        AGC lower-side CUSUM slack k [dB].
    agc_cusum_h : float
        AGC lower-side CUSUM threshold h [dB].
    mp_thresh : float
        Elevation-normalised PR residual RMS alarm threshold [m].
    el_min_deg : float
        Elevation floor used in C/N₀ model and multipath scaling [degrees].

    Invariants
    ----------
    - agc_cusum_lower >= 0 always.
    - len(sat_quality) == n_sats always.
    - cn0_delta_rms and mp_rms are NaN when the corresponding inputs are absent.
    - assess() is the sole external mutating method (only self._* are modified).
    """

    def __init__(
        self,
        n_sats: int,
        cn0_zenith: float = CN0_ZENITH_DBHz,
        cn0_residual_thresh: float = CN0_RESIDUAL_THRESH,
        cn0_sat_thresh: float = CN0_SAT_THRESH,
        agc_warmup_epochs: int = AGC_WARMUP_EPOCHS,
        agc_cusum_k: float = AGC_CUSUM_K,
        agc_cusum_h: float = AGC_CUSUM_H,
        mp_thresh: float = MP_THRESH,
        el_min_deg: float = EL_MIN_DEG,
    ) -> None:
        self._n_sats = n_sats
        self._cn0_zenith = cn0_zenith
        self._cn0_residual_thresh = cn0_residual_thresh
        self._cn0_sat_thresh = cn0_sat_thresh
        self._agc_warmup_epochs = agc_warmup_epochs
        self._agc_k = agc_cusum_k
        self._agc_h = agc_cusum_h
        self._mp_thresh = mp_thresh
        self._el_min_rad: float = float(np.radians(el_min_deg))
        self._mp_sat_thresh: float = mp_thresh * _MP_SAT_FACTOR

        # AGC CUSUM state
        self._agc_warmup_buf: list[float] = []
        self._agc_mu0: float | None = None
        self._agc_cusum_lower: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset AGC warmup buffer and CUSUM state.

        Call after receiver reacquisition or signal interruption.
        """
        self._agc_warmup_buf = []
        self._agc_mu0 = None
        self._agc_cusum_lower = 0.0

    def assess(
        self,
        epoch: int,
        cn0_dbhz: np.ndarray | None = None,
        agc_db: float | None = None,
        pseudorange_residuals: np.ndarray | None = None,
        elevation_rad: np.ndarray | None = None,
    ) -> SignalQualityResult:
        """Assess signal quality for one epoch.

        Parameters
        ----------
        epoch : int
            Epoch index (forwarded to result; monotonicity not enforced here).
        cn0_dbhz : np.ndarray, shape (n_sats,), optional
            Carrier-to-noise ratios [dB-Hz].  NaN values are skipped.
        agc_db : float, optional
            Mean receiver AGC level [dB] — scalar per epoch.
        pseudorange_residuals : np.ndarray, shape (n_sats,), optional
            Code pseudorange residuals after model subtraction [m].
        elevation_rad : np.ndarray, shape (n_sats,), optional
            Satellite elevation angles [radians].  Required by sub-monitors 1
            and 3; both are skipped when absent.

        Returns
        -------
        SignalQualityResult
        """
        cn0_alarm, cn0_score, cn0_delta_rms, n_sat_cn0 = self._check_cn0_elevation(
            cn0_dbhz, elevation_rad
        )
        agc_alarm, agc_score, agc_cusum_lower = self._check_agc(agc_db)
        mp_alarm, mp_score, mp_rms = self._check_multipath(pseudorange_residuals, elevation_rad)

        sat_quality = self._label_satellites(cn0_dbhz, pseudorange_residuals, elevation_rad)

        quality_score = float(max(cn0_score, agc_score, mp_score))
        alarm = cn0_alarm or agc_alarm or mp_alarm

        reasons: list[str] = []
        if cn0_alarm:
            reasons.append(
                f"C/N0 model residual (RMS={cn0_delta_rms:.2f} dB-Hz, {n_sat_cn0} sat anomalous)"
            )
        if agc_alarm:
            reasons.append(f"AGC drop detected (S^-={agc_cusum_lower:.2f} > h={self._agc_h:.2f})")
        if mp_alarm:
            reasons.append(f"Multipath elevated (mp_rms={mp_rms:.2f} m > {self._mp_thresh:.2f} m)")

        return SignalQualityResult(
            epoch=epoch,
            alarm=alarm,
            quality_score=quality_score,
            cn0_alarm=cn0_alarm,
            cn0_score=cn0_score,
            cn0_delta_rms=cn0_delta_rms,
            n_sat_cn0_anomaly=n_sat_cn0,
            agc_alarm=agc_alarm,
            agc_score=agc_score,
            agc_cusum_lower=agc_cusum_lower,
            mp_alarm=mp_alarm,
            mp_score=mp_score,
            mp_rms=mp_rms,
            sat_quality=sat_quality,
            reasons=tuple(reasons),
        )

    # ------------------------------------------------------------------
    # Sub-monitor 1 — C/N₀ elevation model
    # ------------------------------------------------------------------

    def _check_cn0_elevation(
        self,
        cn0_dbhz: np.ndarray | None,
        elevation_rad: np.ndarray | None,
    ) -> tuple[bool, float, float, int]:
        """Compare measured C/N₀ against elevation-predicted baseline.

        Returns (alarm, score ∈ [0,1], rms_residual_dBHz, n_sat_anomaly).
        Returns (False, 0.0, nan, 0) when either input is absent.
        """
        if cn0_dbhz is None or elevation_rad is None:
            return False, 0.0, float("nan"), 0

        cn0 = np.asarray(cn0_dbhz, dtype=float)
        el = np.asarray(elevation_rad, dtype=float)
        n = min(len(cn0), len(el))
        if n == 0:
            return False, 0.0, float("nan"), 0

        cn0 = cn0[:n]
        el_clamped = np.maximum(el[:n], self._el_min_rad)

        # CN0 model: cn0_expected = zenith + 10·log10(sin(el))
        cn0_expected = self._cn0_zenith + 10.0 * np.log10(np.sin(el_clamped))

        valid = ~np.isnan(cn0)
        if not np.any(valid):
            return False, 0.0, float("nan"), 0

        delta = cn0 - cn0_expected  # per-satellite residuals [dB-Hz]
        delta_valid = delta[valid]

        cn0_delta_rms = float(np.sqrt(np.mean(delta_valid**2)))
        n_sat_anomaly = int(np.sum(np.abs(delta_valid) > self._cn0_sat_thresh))

        score = float(np.clip(cn0_delta_rms / self._cn0_residual_thresh, 0.0, 1.0))
        alarm = (cn0_delta_rms > self._cn0_residual_thresh) or (n_sat_anomaly > 0)

        return alarm, score, cn0_delta_rms, n_sat_anomaly

    # ------------------------------------------------------------------
    # Sub-monitor 2 — AGC drop detector
    # ------------------------------------------------------------------

    def _check_agc(self, agc_db: float | None) -> tuple[bool, float, float]:
        """Lower-side CUSUM on scalar AGC level.

        S⁻ₜ = max(0, S⁻ₜ₋₁ + (μ₀ − xₜ − k))

        Returns (alarm, score ∈ [0,1], cusum_lower).
        Returns (False, 0.0, current_state) when agc_db is None.
        """
        if agc_db is None:
            return False, 0.0, self._agc_cusum_lower

        # Warmup: accumulate samples to calibrate μ₀
        if self._agc_mu0 is None:
            self._agc_warmup_buf.append(float(agc_db))
            if len(self._agc_warmup_buf) >= self._agc_warmup_epochs:
                self._agc_mu0 = float(np.mean(self._agc_warmup_buf))
                self._agc_cusum_lower = 0.0
            return False, 0.0, self._agc_cusum_lower

        # Lower-side CUSUM update: fires when xₜ drops well below μ₀
        increment = self._agc_mu0 - float(agc_db) - self._agc_k
        self._agc_cusum_lower = max(0.0, self._agc_cusum_lower + increment)

        score = float(np.clip(self._agc_cusum_lower / self._agc_h, 0.0, 1.0))
        alarm = self._agc_cusum_lower > self._agc_h

        return alarm, score, self._agc_cusum_lower

    # ------------------------------------------------------------------
    # Sub-monitor 3 — Multipath index
    # ------------------------------------------------------------------

    def _check_multipath(
        self,
        pr_residuals: np.ndarray | None,
        elevation_rad: np.ndarray | None,
    ) -> tuple[bool, float, float]:
        """Elevation-normalised pseudorange residual RMS.

        mp_proxyᵢ = |PR_residualᵢ| · sin(elᵢ)
        mp_rms    = RMS(mp_proxy)

        Returns (alarm, score ∈ [0,1], mp_rms).
        Returns (False, 0.0, nan) when either input is absent.
        """
        if pr_residuals is None or elevation_rad is None:
            return False, 0.0, float("nan")

        pr = np.asarray(pr_residuals, dtype=float)
        el = np.asarray(elevation_rad, dtype=float)
        n = min(len(pr), len(el))
        if n == 0:
            return False, 0.0, float("nan")

        pr = pr[:n]
        el_clamped = np.maximum(el[:n], self._el_min_rad)

        valid = ~np.isnan(pr) & ~np.isnan(el[:n])
        if not np.any(valid):
            return False, 0.0, float("nan")

        mp_proxy = np.abs(pr[valid]) * np.sin(el_clamped[valid])
        mp_rms = float(np.sqrt(np.mean(mp_proxy**2)))

        score = float(np.clip(mp_rms / self._mp_thresh, 0.0, 1.0))
        alarm = mp_rms > self._mp_thresh

        return alarm, score, mp_rms

    # ------------------------------------------------------------------
    # Per-satellite labelling
    # ------------------------------------------------------------------

    def _label_satellites(
        self,
        cn0_dbhz: np.ndarray | None,
        pr_residuals: np.ndarray | None,
        elevation_rad: np.ndarray | None,
    ) -> tuple[str, ...]:
        """Assign per-satellite quality label based on C/N₀ and multipath checks."""
        labels: list[str] = ["ok"] * self._n_sats

        # C/N₀ per-satellite labelling
        if cn0_dbhz is not None and elevation_rad is not None:
            cn0 = np.asarray(cn0_dbhz, dtype=float)
            el = np.asarray(elevation_rad, dtype=float)
            k = min(len(cn0), len(el), self._n_sats)
            el_clamped = np.maximum(el[:k], self._el_min_rad)
            cn0_expected = self._cn0_zenith + 10.0 * np.log10(np.sin(el_clamped))
            for i in range(k):
                if not np.isnan(cn0[i]) and abs(cn0[i] - cn0_expected[i]) > self._cn0_sat_thresh:
                    labels[i] = "cn0"

        # Multipath per-satellite labelling
        if pr_residuals is not None and elevation_rad is not None:
            pr = np.asarray(pr_residuals, dtype=float)
            el = np.asarray(elevation_rad, dtype=float)
            k = min(len(pr), len(el), self._n_sats)
            el_clamped = np.maximum(el[:k], self._el_min_rad)
            for i in range(k):
                if not np.isnan(pr[i]) and (
                    abs(pr[i]) * float(np.sin(el_clamped[i])) > self._mp_sat_thresh
                ):
                    labels[i] = "cn0+mp" if labels[i] == "cn0" else "mp"

        return tuple(labels)


# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------


def cn0_elevation_model(
    elevation_rad: np.ndarray,
    cn0_zenith: float = CN0_ZENITH_DBHz,
    el_min_deg: float = EL_MIN_DEG,
) -> np.ndarray:
    """Return expected C/N₀ [dB-Hz] for each elevation angle.

    cn0_expected(el) = cn0_zenith + 10 · log₁₀(sin(max(el, el_min)))

    Parameters
    ----------
    elevation_rad : np.ndarray
        Satellite elevation angles [radians].
    cn0_zenith : float
        Zenith C/N₀ [dB-Hz].
    el_min_deg : float
        Elevation floor [degrees].

    Returns
    -------
    np.ndarray
        Expected C/N₀ values [dB-Hz], same shape as input.
    """
    el = np.asarray(elevation_rad, dtype=float)
    el_clamped = np.maximum(el, float(np.radians(el_min_deg)))
    return cn0_zenith + 10.0 * np.log10(np.sin(el_clamped))


# ---------------------------------------------------------------------------
# Simulation helper
# ---------------------------------------------------------------------------


def run_signal_quality_simulation(
    n_sats: int = 8,
    n_epochs: int = 40,
    jammer_start: int = 25,
    mp_start: int = 10,
    mp_end: int = 20,
    agc_drop_db: float = 8.0,
    seed: int = 42,
) -> list[SignalQualityResult]:
    """Simulate a mixed-scenario signal quality sequence.

    Scenario timeline:
        Epochs 0 … mp_start−1             : nominal
        Epochs mp_start … mp_end−1         : multipath environment
        Epochs jammer_start … n_epochs−1   : AGC drop (in-band jamming)

    Parameters
    ----------
    n_sats : int
        Number of satellites.
    n_epochs : int
        Total epochs to simulate.
    jammer_start : int
        Epoch at which AGC drops by agc_drop_db [dB].
    mp_start, mp_end : int
        Epoch range [mp_start, mp_end) with elevated multipath.
    agc_drop_db : float
        AGC decrease magnitude during jamming [dB].
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list[SignalQualityResult]
        One SignalQualityResult per epoch.
    """
    rng = np.random.default_rng(seed)
    monitor = SignalQualityMonitor(n_sats=n_sats)

    # Fixed constellation geometry: elevations distributed 10°–80°
    elevations_deg = np.linspace(10.0, 80.0, n_sats)
    elevations_rad = np.radians(elevations_deg)

    _AGC_NOMINAL: float = 30.0  # nominal AGC level [dB]
    _AGC_NOISE_STD: float = 0.3  # AGC measurement noise [dB]

    results: list[SignalQualityResult] = []
    for ep in range(n_epochs):
        # C/N0: model-predicted + small Gaussian noise
        cn0_nominal = cn0_elevation_model(elevations_rad)
        cn0 = cn0_nominal + rng.normal(0.0, 1.0, n_sats)

        # AGC: nominal until jammer_start
        agc_mean = _AGC_NOMINAL - agc_drop_db if ep >= jammer_start else _AGC_NOMINAL
        agc = float(agc_mean + rng.normal(0.0, _AGC_NOISE_STD))

        # PR residuals: small normally, elevated during multipath window
        if mp_start <= ep < mp_end:
            # Multipath: inverse-elevation scaling inflates residuals at low elevation
            pr = rng.normal(0.0, 20.0, n_sats) / np.sin(elevations_rad)
        else:
            pr = rng.normal(0.0, 1.0, n_sats)

        result = monitor.assess(
            epoch=ep,
            cn0_dbhz=cn0,
            agc_db=agc,
            pseudorange_residuals=pr,
            elevation_rad=elevations_rad,
        )
        results.append(result)

    return results
