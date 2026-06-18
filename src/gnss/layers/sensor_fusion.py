"""Multi-sensor fusion for GNSS integrity — barometer, visual odometry, factor graph.

Three complementary L4 consistency checks that detect GNSS position/velocity
anomalies by cross-referencing independent sensor streams:

Sub-monitor 1 — Barometer altitude chi² test
---------------------------------------------
GNSS altitude and barometric altitude are independent measurements of the same
physical quantity.  Under a spoofing attack the GNSS-reported altitude drifts
while the barometer continues to measure true altitude.

Test (1-DOF chi²):
    Δh = h_gnss − h_baro                        [m]
    T_baro = Δh² / (σ_gnss_h² + σ_baro²)       ~ χ²(1)

Alert threshold: χ²(1, 1 − α) = 3.841  (α = 0.05)

Ref: Ioannides et al. (2016) Inside GNSS §4; ESA GNSS Receiver Fault Detection.

Sub-monitor 2 — Visual Odometry chi² test
------------------------------------------
A camera-derived velocity estimate (visual odometry) provides an independent
horizontal velocity reference that is immune to GNSS signal manipulation.

Test (2-DOF chi²):
    Δv = v_gnss − v_vo                           [m/s, 2-D or 3-D]
    T_vo = Δv · Σ⁻¹ · Δv                        ~ χ²(d)
           where Σ = diag(σ_gnss_v², ...) + diag(σ_vo², ...)
           and d = dimension of v

Alert threshold:
    d=2: χ²(2, 0.95) = 5.991
    d=3: χ²(3, 0.95) = 7.815

Ref: Scaramuzza & Fraundorfer (2011) IEEE RAM §3; Psiaki et al. (2014) §VI.

Sub-monitor 3 — Fixed-lag smoother (simplified factor-graph)
--------------------------------------------------------------
A sliding-window Gaussian smoother fuses GNSS position, barometric altitude,
and VO velocity over a fixed lag L.  The consistency residual at each epoch
is used as an additional anomaly score.

For a linear-Gaussian model the factor-graph smoother is equivalent to
a fixed-lag RTS smoother.  Here we implement a simplified diagonal-Gaussian
version:

    State:  s_t = [x_t, y_t, z_t, vx_t, vy_t, vz_t]   ∈ ℝ⁶
    Motion model: s_t = F·s_{t-1} + w_t,  w_t ~ N(0, Q)
    Observations:
        z_gnss_t = H_gnss · s_t + η_gnss,    η_gnss ~ N(0, R_gnss)
        z_baro_t = H_baro · s_t + η_baro,    η_baro ~ N(0, R_baro)
        z_vo_t   = H_vo   · s_t + η_vo,      η_vo   ~ N(0, R_vo)

    Smoother residual: r_t = z_gnss_t − H_gnss · ŝ_{t|t-1}
    Consistency score: T_sm = r_t · S⁻¹ · r_t   ~ χ²(d_gnss)

Implementation: Sequential EKF over the fixed-lag window; we expose the
innovation covariance S and normalised innovation squared (NIS).

Ref: Thrun, Burgard, Fox (2005) Probabilistic Robotics §3.3 (EKF);
     Bar-Shalom, Li, Kirubarajan (2001) §7 (fixed-lag smoother).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EPS: float = 1e-12

# Barometer sub-monitor
BARO_SIGMA_GNSS_H: float = 5.0  # GNSS altitude 1-σ uncertainty [m]
BARO_SIGMA_BARO: float = 2.0  # barometer 1-σ uncertainty [m]
BARO_CHI2_THRESH: float = 3.841  # χ²(1, 0.95)

# Visual odometry sub-monitor
VO_SIGMA_GNSS_V: float = 0.3  # GNSS velocity 1-σ [m/s] per axis
VO_SIGMA_VO: float = 0.1  # VO velocity 1-σ [m/s] per axis
VO_CHI2_THRESH_2D: float = 5.991  # χ²(2, 0.95)
VO_CHI2_THRESH_3D: float = 7.815  # χ²(3, 0.95)

# Fixed-lag smoother
SMOOTHER_LAG: int = 10  # fixed-lag window length L [epochs]
SMOOTHER_DT: float = 1.0  # nominal epoch duration [s]
SMOOTHER_SIGMA_Q: float = 1.0  # motion noise 1-σ per axis per step [m/s²·dt]
SMOOTHER_NIS_THRESH: float = 9.488  # χ²(4, 0.95) — 4 GNSS obs (x,y,z, h_baro fused)

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BarometerResult:
    """Per-epoch barometer altitude check result."""

    delta_h: float  # GNSS − baro altitude difference [m]
    chi2_stat: float  # T_baro ~ χ²(1)
    alarm: bool


@dataclass(frozen=True)
class VisualOdometryResult:
    """Per-epoch visual odometry consistency result."""

    delta_v: np.ndarray  # GNSS − VO velocity residual [m/s]
    chi2_stat: float  # T_vo ~ χ²(d)
    dof: int  # degrees of freedom (2 or 3)
    alarm: bool


@dataclass(frozen=True)
class SmootherResult:
    """Per-epoch fixed-lag smoother normalised innovation squared."""

    nis: float  # normalised innovation squared (NIS) ~ χ²(d_gnss)
    alarm: bool
    n_fused: int  # number of sensor streams fused this epoch


@dataclass(frozen=True)
class SensorFusionResult:
    """Combined per-epoch result for all L4 sensor-fusion sub-monitors.

    Fields
    ------
    epoch          Epoch index.
    baro           Barometer altitude check.
    vo             Visual odometry velocity check.
    smoother       Fixed-lag smoother NIS.
    alarm          True when ANY sub-monitor fires.
    quality_score  max sub-score ∈ [0, 1].
    reasons        Active alert labels.
    """

    epoch: int
    baro: BarometerResult | None
    vo: VisualOdometryResult | None
    smoother: SmootherResult | None
    alarm: bool
    quality_score: float
    reasons: list[str]


# ---------------------------------------------------------------------------
# Sub-monitor 1: Barometer altitude
# ---------------------------------------------------------------------------


def check_barometer(
    h_gnss: float,
    h_baro: float,
    sigma_gnss_h: float = BARO_SIGMA_GNSS_H,
    sigma_baro: float = BARO_SIGMA_BARO,
    chi2_thresh: float = BARO_CHI2_THRESH,
) -> BarometerResult:
    """Chi² consistency check between GNSS and barometric altitude.

    Args:
        h_gnss:       GNSS-reported altitude [m].
        h_baro:       Barometric altitude [m].
        sigma_gnss_h: GNSS altitude 1-σ uncertainty [m].
        sigma_baro:   Barometer 1-σ uncertainty [m].
        chi2_thresh:  χ²(1) detection threshold.

    Returns:
        BarometerResult.
    """
    delta_h = h_gnss - h_baro
    sigma_sq = sigma_gnss_h**2 + sigma_baro**2
    chi2_stat = delta_h**2 / (sigma_sq + _EPS)
    return BarometerResult(
        delta_h=delta_h,
        chi2_stat=chi2_stat,
        alarm=chi2_stat > chi2_thresh,
    )


# ---------------------------------------------------------------------------
# Sub-monitor 2: Visual Odometry
# ---------------------------------------------------------------------------


def check_visual_odometry(
    v_gnss: np.ndarray,
    v_vo: np.ndarray,
    sigma_gnss_v: float = VO_SIGMA_GNSS_V,
    sigma_vo: float = VO_SIGMA_VO,
    chi2_thresh_2d: float = VO_CHI2_THRESH_2D,
    chi2_thresh_3d: float = VO_CHI2_THRESH_3D,
) -> VisualOdometryResult:
    """Chi² consistency check between GNSS and visual odometry velocity.

    Args:
        v_gnss:         GNSS velocity vector [m/s], shape (2,) or (3,).
        v_vo:           VO velocity vector [m/s], same shape as v_gnss.
        sigma_gnss_v:   GNSS velocity 1-σ per axis [m/s].
        sigma_vo:       VO velocity 1-σ per axis [m/s].
        chi2_thresh_2d: Detection threshold for 2-D velocity.
        chi2_thresh_3d: Detection threshold for 3-D velocity.

    Returns:
        VisualOdometryResult.

    Raises:
        ValueError: if v_gnss and v_vo have different shapes or unsupported dim.
    """
    v_g = np.asarray(v_gnss, dtype=float).ravel()
    v_v = np.asarray(v_vo, dtype=float).ravel()
    if v_g.shape != v_v.shape:
        raise ValueError(f"v_gnss and v_vo must have the same shape: {v_g.shape} vs {v_v.shape}")
    d = len(v_g)
    if d not in (2, 3):
        raise ValueError(f"Velocity dimension must be 2 or 3, got {d}.")

    delta_v = v_g - v_v
    sigma_sq = sigma_gnss_v**2 + sigma_vo**2
    # Σ = diag(σ²) → T_vo = Δv · Σ⁻¹ · Δv = ||Δv||² / σ²
    chi2_stat = float(np.dot(delta_v, delta_v) / (sigma_sq + _EPS))
    thresh = chi2_thresh_2d if d == 2 else chi2_thresh_3d
    return VisualOdometryResult(
        delta_v=delta_v,
        chi2_stat=chi2_stat,
        dof=d,
        alarm=chi2_stat > thresh,
    )


# ---------------------------------------------------------------------------
# Sub-monitor 3: Fixed-lag smoother (simplified EKF)
# ---------------------------------------------------------------------------


class FixedLagSmoother:
    """Simplified fixed-lag EKF smoother for multi-sensor GNSS fusion.

    State vector: [x, y, z, vx, vy, vz] ∈ ℝ⁶ (position + velocity).
    Observations accepted each epoch:
        - GNSS position: z_gnss = [x, y, z]   (3-DOF)
        - Barometric altitude: z_baro = z       (1-DOF, optional)
        - VO velocity: z_vo = [vx, vy, vz]     (2- or 3-DOF, optional)

    The lag window is maintained as a deque of (state, cov) pairs.  Only the
    most recent marginal (last element) is used here; the full smoother pass
    is skipped in favour of the simpler forward filter for computational
    efficiency.

    The NIS at the GNSS measurement step is reported as the consistency score.

    Usage::

        smoother = FixedLagSmoother()
        result   = smoother.update(
            pos_gnss=np.array([x, y, z]),
            h_baro=300.0,         # optional
            v_vo=np.array([...]), # optional
        )
    """

    def __init__(
        self,
        lag: int = SMOOTHER_LAG,
        dt: float = SMOOTHER_DT,
        sigma_q: float = SMOOTHER_SIGMA_Q,
        sigma_gnss_p: float = 3.0,  # GNSS position 1-σ [m]
        sigma_baro: float = BARO_SIGMA_BARO,
        sigma_vo: float = VO_SIGMA_VO,
        nis_thresh: float = SMOOTHER_NIS_THRESH,
    ) -> None:
        self._lag = lag
        self._dt = dt
        self._sigma_q = sigma_q
        self._sigma_gnss_p = sigma_gnss_p
        self._sigma_baro = sigma_baro
        self._sigma_vo = sigma_vo
        self._nis_thresh = nis_thresh

        # State transition (constant-velocity model)
        self._F = np.eye(6)
        self._F[0, 3] = dt
        self._F[1, 4] = dt
        self._F[2, 5] = dt

        # Process noise (piecewise constant acceleration noise)
        q = sigma_q**2
        self._Q = np.diag([q * dt**2, q * dt**2, q * dt**2, q, q, q])

        # Initial state and covariance
        self._x: np.ndarray = np.zeros(6)
        self._P: np.ndarray = np.eye(6) * 1000.0
        self._initialized: bool = False
        self._history: deque[tuple[np.ndarray, np.ndarray]] = deque(maxlen=lag)

    def reset(self) -> None:
        """Reset filter state between experiments."""
        self._x = np.zeros(6)
        self._P = np.eye(6) * 1000.0
        self._initialized = False
        self._history.clear()

    def update(
        self,
        pos_gnss: np.ndarray,
        h_baro: float | None = None,
        v_vo: np.ndarray | None = None,
    ) -> SmootherResult:
        """Run one EKF predict + update step.

        Args:
            pos_gnss: GNSS position [x, y, z] in metres (ENU or ECEF).
            h_baro:   Barometric altitude [m] (z-coordinate equivalent), or None.
            v_vo:     VO velocity [vx, vy] or [vx, vy, vz] [m/s], or None.

        Returns:
            SmootherResult with NIS and alarm flag.
        """
        pos = np.asarray(pos_gnss, dtype=float).ravel()
        if len(pos) != 3:
            raise ValueError(f"pos_gnss must have 3 elements, got {len(pos)}.")

        # Initialise state from first GNSS fix
        if not self._initialized:
            self._x[:3] = pos
            self._initialized = True
            self._history.append((self._x.copy(), self._P.copy()))
            return SmootherResult(nis=0.0, alarm=False, n_fused=1)

        # --- Predict ---
        x_pred = self._F @ self._x
        P_pred = self._F @ self._P @ self._F.T + self._Q

        # --- GNSS position update (3-DOF) ---
        H_gnss = np.zeros((3, 6))
        H_gnss[0, 0] = 1.0
        H_gnss[1, 1] = 1.0
        H_gnss[2, 2] = 1.0
        R_gnss = np.eye(3) * self._sigma_gnss_p**2

        innov = pos - H_gnss @ x_pred
        S = H_gnss @ P_pred @ H_gnss.T + R_gnss
        K = P_pred @ H_gnss.T @ np.linalg.inv(S)
        x_upd = x_pred + K @ innov
        P_upd = (np.eye(6) - K @ H_gnss) @ P_pred

        # NIS for GNSS measurement
        nis = float(innov @ np.linalg.inv(S) @ innov)
        n_fused = 1

        # --- Barometer altitude update (1-DOF) ---
        if h_baro is not None:
            H_baro = np.zeros((1, 6))
            H_baro[0, 2] = 1.0
            R_baro = np.array([[self._sigma_baro**2]])
            innov_b = np.array([h_baro]) - H_baro @ x_upd
            S_b = H_baro @ P_upd @ H_baro.T + R_baro
            K_b = P_upd @ H_baro.T @ np.linalg.inv(S_b)
            x_upd = x_upd + (K_b @ innov_b).ravel()
            P_upd = (np.eye(6) - K_b @ H_baro) @ P_upd
            n_fused += 1

        # --- VO velocity update ---
        if v_vo is not None:
            v = np.asarray(v_vo, dtype=float).ravel()
            d_vo = len(v)
            H_vo = np.zeros((d_vo, 6))
            for k in range(d_vo):
                H_vo[k, 3 + k] = 1.0
            R_vo = np.eye(d_vo) * self._sigma_vo**2
            innov_v = v - H_vo @ x_upd
            S_v = H_vo @ P_upd @ H_vo.T + R_vo
            K_v = P_upd @ H_vo.T @ np.linalg.inv(S_v)
            x_upd = x_upd + K_v @ innov_v
            P_upd = (np.eye(6) - K_v @ H_vo) @ P_upd
            n_fused += 1

        self._x = x_upd
        self._P = P_upd
        self._history.append((self._x.copy(), self._P.copy()))

        return SmootherResult(
            nis=nis,
            alarm=nis > self._nis_thresh,
            n_fused=n_fused,
        )


# ---------------------------------------------------------------------------
# Epoch-level fusion layer
# ---------------------------------------------------------------------------


class SensorFusionLayer:
    """Combined L4 sensor fusion layer.

    Runs all available sub-monitors each epoch and fuses results.

    Usage::

        layer = SensorFusionLayer()
        result = layer.assess(
            epoch=t,
            pos_gnss=np.array([x, y, z]),
            h_baro=300.0,
            v_gnss=np.array([vx, vy]),
            v_vo=np.array([vx_vo, vy_vo]),
        )
    """

    def __init__(
        self,
        baro_thresh: float = BARO_CHI2_THRESH,
        vo_thresh_2d: float = VO_CHI2_THRESH_2D,
        vo_thresh_3d: float = VO_CHI2_THRESH_3D,
        sigma_gnss_h: float = BARO_SIGMA_GNSS_H,
        sigma_baro: float = BARO_SIGMA_BARO,
        sigma_gnss_v: float = VO_SIGMA_GNSS_V,
        sigma_vo: float = VO_SIGMA_VO,
        smoother_lag: int = SMOOTHER_LAG,
        nis_thresh: float = SMOOTHER_NIS_THRESH,
    ) -> None:
        self._baro_thresh = baro_thresh
        self._vo_thresh_2d = vo_thresh_2d
        self._vo_thresh_3d = vo_thresh_3d
        self._sigma_gnss_h = sigma_gnss_h
        self._sigma_baro = sigma_baro
        self._sigma_gnss_v = sigma_gnss_v
        self._sigma_vo = sigma_vo
        self._smoother = FixedLagSmoother(
            lag=smoother_lag,
            sigma_baro=sigma_baro,
            sigma_vo=sigma_vo,
            nis_thresh=nis_thresh,
        )

    def reset(self) -> None:
        """Reset the smoother between independent experiments."""
        self._smoother.reset()

    def assess(
        self,
        epoch: int,
        pos_gnss: np.ndarray,
        h_baro: float | None = None,
        v_gnss: np.ndarray | None = None,
        v_vo: np.ndarray | None = None,
    ) -> SensorFusionResult:
        """Assess sensor consistency for the current epoch.

        Args:
            epoch:    Epoch index.
            pos_gnss: GNSS position [x, y, z] [m].
            h_baro:   Barometric altitude [m], or None.
            v_gnss:   GNSS velocity [vx, vy] or [vx, vy, vz] [m/s], or None.
            v_vo:     VO velocity (same shape as v_gnss), or None.

        Returns:
            SensorFusionResult.
        """
        baro_res: BarometerResult | None = None
        vo_res: VisualOdometryResult | None = None
        reasons: list[str] = []
        scores: list[float] = []

        # --- Barometer ---
        if h_baro is not None:
            h_gnss = float(pos_gnss[2]) if len(pos_gnss) >= 3 else 0.0
            baro_res = check_barometer(
                h_gnss=h_gnss,
                h_baro=h_baro,
                sigma_gnss_h=self._sigma_gnss_h,
                sigma_baro=self._sigma_baro,
                chi2_thresh=self._baro_thresh,
            )
            if baro_res.alarm:
                reasons.append(f"baro_chi2={baro_res.chi2_stat:.2f}>{self._baro_thresh}")
            scores.append(min(1.0, baro_res.chi2_stat / (self._baro_thresh + _EPS)))

        # --- Visual Odometry ---
        if v_gnss is not None and v_vo is not None:
            vo_res = check_visual_odometry(
                v_gnss=np.asarray(v_gnss),
                v_vo=np.asarray(v_vo),
                sigma_gnss_v=self._sigma_gnss_v,
                sigma_vo=self._sigma_vo,
                chi2_thresh_2d=self._vo_thresh_2d,
                chi2_thresh_3d=self._vo_thresh_3d,
            )
            thresh = self._vo_thresh_2d if vo_res.dof == 2 else self._vo_thresh_3d
            if vo_res.alarm:
                reasons.append(f"vo_chi2={vo_res.chi2_stat:.2f}>{thresh}")
            scores.append(min(1.0, vo_res.chi2_stat / (thresh + _EPS)))

        # --- Fixed-lag smoother ---
        sm_res = self._smoother.update(
            pos_gnss=np.asarray(pos_gnss, dtype=float),
            h_baro=h_baro,
            v_vo=np.asarray(v_vo, dtype=float) if v_vo is not None else None,
        )
        if sm_res.alarm:
            reasons.append(f"smoother_nis={sm_res.nis:.2f}>{self._smoother._nis_thresh}")
        scores.append(min(1.0, sm_res.nis / (self._smoother._nis_thresh + _EPS)))

        alarm = (
            (baro_res is not None and baro_res.alarm)
            or (vo_res is not None and vo_res.alarm)
            or sm_res.alarm
        )
        quality_score = max(scores) if scores else 0.0

        return SensorFusionResult(
            epoch=epoch,
            baro=baro_res,
            vo=vo_res,
            smoother=sm_res,
            alarm=alarm,
            quality_score=quality_score,
            reasons=reasons,
        )
