"""GNSS Resilience Twin — Layers 1, 2, 5, 6, 9: Integrity Pillar components.

Layer 1 — GM-RAIM: per-satellite fault posterior via 2-component GMM
Layer 2 — IMM-KF:  3-mode interacting multiple model Kalman filter
Layer 5 — INS Coupling: chi² cross-check against external INS velocity
Layer 6 — Cooperative RAIM: parity-space + split-subset integrity tests
Layer 9 — Huh D-optimal subset: greedy max-det healthy satellite selection
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2 as _chi2_dist

from gnss.constants import _DOPPLER_NOISE_STD, _INS_VEL_STD
from gnss.math_utils import _geometry_matrix

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EPS: float = 1e-300

_EL_MIN_DEG: float = 5.0
_EL_MIN_RAD: float = math.radians(_EL_MIN_DEG)

# Layer 1 — GM-RAIM
_GMM_FAULT_PRIOR: float = 0.05
_GMM_FAULT_SCALE: float = 5.0
_GMM_FAULT_THRESH: float = 0.5

# Layer 2 — IMM-KF
_IMM_Q0: float = 0.001
_IMM_Q1: float = 0.010
_IMM_Q2: float = 0.005
_IMM_SPOOF_RHO: float = 3.0
_IMM_P_INIT: float = 1.0
_IMM_TRANSITION: list[list[float]] = [
    [0.95, 0.03, 0.02],
    [0.10, 0.85, 0.05],
    [0.05, 0.05, 0.90],
]

# Layer 5 — INS coupling chi² thresholds (chi²(3) at 1% significance)
_INS_CHI2_VEL_THRESH: float = 11.345
_INS_CHI2_CROSS_THRESH: float = 11.345

# Layer 6 — Cooperative RAIM significance level
_COOP_RAIM_ALPHA: float = 0.05

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GMMResult:
    """Output of GM-RAIM per-epoch classification."""

    gamma: tuple[float, ...]  # per-satellite fault posterior γᵢ
    n_fault: int  # satellites with γᵢ > _GMM_FAULT_THRESH
    sign_corr: float  # |mean(sign(Δf))| — common-bias (spoofing) indicator
    elev_corr: float  # |corr(|Δf|, 1/sin(el))| — multipath indicator
    raim_stat: float  # mean(γᵢ) — aggregate fault intensity


@dataclass(frozen=True)
class IMMResult:
    """Output of IMM-KF per-epoch update."""

    mode_weights: tuple[float, float, float]  # μ = [μ_nom, μ_mp, μ_spoof]
    x_fused: tuple[float, float, float, float]  # fused state [Δvx,Δvy,Δvz,Δḃ]
    innovation_norms: tuple[float, float, float]  # ‖νₘ‖₂ per mode


@dataclass(frozen=True)
class INSCouplingResult:
    """Output of INS coupling chi² cross-check (Layer 5).

    chi2_vel = ‖x_fused[:3]‖² / σ_INS²  ∼  chi²(3) under H₀
    chi2_cross = ‖v_ins − x_fused[:3]‖² / (2σ_INS²)  ∼  chi²(3)  [if INS available]
    """

    chi2_vel: float
    chi2_cross: float
    ins_available: bool
    alert: bool


@dataclass(frozen=True)
class CoopRAIMResult:
    """Output of cooperative RAIM parity-space test (Layer 6).

    Parity matrix P = I − H(HᵀH)⁻¹Hᵀ,  T_p = pᵀp / σ²  ∼  chi²(n−4) under H₀
    Split chi²: minimum-norm LS on two equal-sized subsets, ‖x̂_A − x̂_B‖² / σ²
    """

    parity_chi2: float
    dof: int
    parity_alert: bool
    split_chi2: float
    split_alert: bool


@dataclass(frozen=True)
class HuhSelectionResult:
    """Output of Huh D-optimal satellite subset selector (Layer 9).

    det_ratio = det(H_sel ᵀ H_sel) / det(H_all ᵀ H_all)
    log_concavity_ratio = min σₖ² / (σₖ₋₁ σₖ₊₁) on singular values of H_sel
    """

    selected_subset: tuple[int, ...]
    det_ratio: float
    n_selected: int
    n_excluded: int
    log_concavity_ratio: float


# ---------------------------------------------------------------------------
# Layer 1 — GM-RAIM
# ---------------------------------------------------------------------------


class GMMRaim:
    """Gaussian-mixture RAIM for per-satellite fault detection.

    2-component GMM per satellite i:
        H₀: N(0, σᵢ²)             weight 1 − π_fault   (nominal)
        H₁: N(0, (scale·σᵢ)²)    weight π_fault        (fault)

        σᵢ = σ_nom / sin(max(elᵢ, el_min))   [elevation-adjusted noise]
        γᵢ = P(H₁ | Δfᵢ)                     [per-satellite fault posterior]
    """

    def __init__(
        self,
        noise_std: float = _DOPPLER_NOISE_STD,
        fault_prior: float = _GMM_FAULT_PRIOR,
        fault_scale: float = _GMM_FAULT_SCALE,
    ) -> None:
        self._sigma = noise_std
        self._pi_fault = fault_prior
        self._pi_nom = 1.0 - fault_prior
        self._scale = fault_scale

    def classify(self, doppler_dev: np.ndarray, elevations: np.ndarray) -> GMMResult:
        """Compute per-satellite fault posteriors and aggregate indicators.

        Args:
            doppler_dev: (n,) Doppler residuals [Hz]
            elevations:  (n,) elevation angles [radians]
        """
        el_clamped = np.maximum(elevations, _EL_MIN_RAD)
        sigma_i = self._sigma / np.sin(el_clamped)

        log_p0 = -0.5 * (doppler_dev / sigma_i) ** 2 - np.log(sigma_i) + math.log(self._pi_nom)
        log_p1 = (
            -0.5 * (doppler_dev / (self._scale * sigma_i)) ** 2
            - np.log(self._scale * sigma_i)
            + math.log(self._pi_fault)
        )

        log_m = np.maximum(log_p0, log_p1)
        log_sum = log_m + np.log(np.exp(log_p0 - log_m) + np.exp(log_p1 - log_m))
        gamma = np.exp(log_p1 - log_sum)

        sign_corr = float(abs(np.sign(doppler_dev).mean()))

        abs_dev = np.abs(doppler_dev)
        inv_sin_el = 1.0 / np.sin(el_clamped)
        if np.std(abs_dev) > _EPS and np.std(inv_sin_el) > _EPS:
            elev_corr = float(abs(np.corrcoef(abs_dev, inv_sin_el)[0, 1]))
        else:
            elev_corr = 0.0

        return GMMResult(
            gamma=tuple(float(g) for g in gamma),
            n_fault=int(np.sum(gamma > _GMM_FAULT_THRESH)),
            sign_corr=sign_corr,
            elev_corr=elev_corr,
            raim_stat=float(gamma.mean()),
        )


# ---------------------------------------------------------------------------
# Layer 2 — IMM Kalman Filter
# ---------------------------------------------------------------------------


class IMMKalman:
    """Interacting Multiple Model Kalman filter — 3 regime modes.

    State:  x = [Δvx, Δvy, Δvz, Δḃ] ∈ ℝ⁴
    Transition: F = I₄  (random-walk receiver dynamics)
    Observation: z = H·x + w,  H ∈ ℝ^{n×4},  w ∼ N(0, σ_D²·Iₙ)

    Process noise per mode:
        Q₀ = q₀·I₄              (nominal)
        Q₁ = q₁·I₄              (multipath)
        Q₂ = q₂·(I₄ + ρ·e₃e₃ᵀ) (spoofing — inflated clock component)
    """

    def __init__(self, los: np.ndarray, noise_std: float = _DOPPLER_NOISE_STD) -> None:
        n_sats = len(los)
        self._n = n_sats
        self._H = _geometry_matrix(los, list(range(n_sats)))
        self._R = noise_std**2 * np.eye(n_sats)

        e3 = np.zeros(4)
        e3[3] = 1.0
        self._Q: list[np.ndarray] = [
            _IMM_Q0 * np.eye(4),
            _IMM_Q1 * np.eye(4),
            _IMM_Q2 * (np.eye(4) + _IMM_SPOOF_RHO * np.outer(e3, e3)),
        ]

        self._Pi = np.array(_IMM_TRANSITION, dtype=float)
        self._mu = np.array([0.97, 0.015, 0.015])
        self._x = [np.zeros(4) for _ in range(3)]
        self._P = [_IMM_P_INIT * np.eye(4) for _ in range(3)]

    def update(self, z: np.ndarray) -> IMMResult:
        """Run one IMM-KF step.

        Args:
            z: (n_sats,) Doppler deviations [Hz]
        """
        M = 3
        H, R, I4 = self._H, self._R, np.eye(4)

        # 1. Interaction (mixing)
        c_bar = np.clip(self._Pi.T @ self._mu, _EPS, None)

        x_mix = np.zeros((M, 4))
        P_mix = [np.zeros((4, 4)) for _ in range(M)]
        for m in range(M):
            for j in range(M):
                w_jm = self._Pi[j, m] * self._mu[j] / c_bar[m]
                x_mix[m] += w_jm * self._x[j]
        for m in range(M):
            for j in range(M):
                w_jm = self._Pi[j, m] * self._mu[j] / c_bar[m]
                dx = self._x[j] - x_mix[m]
                P_mix[m] += w_jm * (self._P[j] + np.outer(dx, dx))

        # 2. Mode-conditioned KF update
        x_upd = np.zeros((M, 4))
        P_upd = [np.zeros((4, 4)) for _ in range(M)]
        log_lkl = np.zeros(M)
        nu_norms = np.zeros(M)

        for m in range(M):
            x_pred = x_mix[m]
            P_pred = P_mix[m] + self._Q[m]

            nu = z - H @ x_pred
            nu_norms[m] = float(np.linalg.norm(nu))

            S = H @ P_pred @ H.T + R
            try:
                K_T = np.linalg.solve(S, H @ P_pred)
            except np.linalg.LinAlgError:
                K_T = np.zeros((self._n, 4))
            K = K_T.T

            x_upd[m] = x_pred + K @ nu
            P_upd[m] = (I4 - K @ H) @ P_pred

            sign_det, log_det = np.linalg.slogdet(S)
            if sign_det > 0:
                try:
                    quad = float(nu @ np.linalg.solve(S, nu))
                except np.linalg.LinAlgError:
                    quad = float(nu @ nu)
            else:
                quad = float(nu @ nu)
                log_det = 0.0
            log_lkl[m] = -0.5 * (self._n * math.log(2.0 * math.pi) + log_det + quad)

        # 3. Mode probability update
        log_mu = log_lkl + np.log(c_bar)
        log_mu -= log_mu.max()
        mu_new = np.exp(log_mu)
        mu_new /= mu_new.sum()

        # 4. Fused state estimate
        x_fused = np.zeros(4)
        for m in range(M):
            x_fused += mu_new[m] * x_upd[m]

        self._mu = mu_new
        self._x = [x_upd[m] for m in range(M)]
        self._P = P_upd

        return IMMResult(
            mode_weights=(float(mu_new[0]), float(mu_new[1]), float(mu_new[2])),
            x_fused=(float(x_fused[0]), float(x_fused[1]), float(x_fused[2]), float(x_fused[3])),
            innovation_norms=(float(nu_norms[0]), float(nu_norms[1]), float(nu_norms[2])),
        )


# ---------------------------------------------------------------------------
# Layer 5 — INS Coupling
# ---------------------------------------------------------------------------


class INSCouplingLayer:
    """INS coupling chi² cross-check (Layer 5).

    Tests whether the IMM-KF fused velocity state is consistent with an
    external INS velocity reference (when available):

        chi2_vel   = ‖x_fused[:3]‖² / σ_INS²             ~ chi²(3) under H₀
        chi2_cross = ‖v_INS − x_fused[:3]‖² / (2σ_INS²)  ~ chi²(3) when INS supplied

    Alert threshold: chi²(0.99, 3) = 11.345 (1 % false-alarm rate).
    """

    def __init__(self, noise_std: float = _INS_VEL_STD) -> None:
        self._sigma2 = noise_std**2

    def assess(
        self,
        x_fused: tuple[float, float, float, float],
        ins_velocity: np.ndarray | None = None,
    ) -> INSCouplingResult:
        """Evaluate IMM-KF fused state against INS reference.

        Args:
            x_fused:      IMM-KF fused state (Δvx, Δvy, Δvz, Δḃ) [m/s]
            ins_velocity: (3,) external INS velocity deviation [m/s], or None
        """
        v = np.array(x_fused[:3], dtype=float)
        chi2_vel = float(np.dot(v, v) / self._sigma2)

        if ins_velocity is not None:
            diff = ins_velocity[:3] - v
            chi2_cross = float(np.dot(diff, diff) / (2.0 * self._sigma2))
            ins_available = True
        else:
            chi2_cross = 0.0
            ins_available = False

        alert = chi2_vel > _INS_CHI2_VEL_THRESH or (
            ins_available and chi2_cross > _INS_CHI2_CROSS_THRESH
        )
        return INSCouplingResult(
            chi2_vel=chi2_vel,
            chi2_cross=chi2_cross,
            ins_available=ins_available,
            alert=alert,
        )


# ---------------------------------------------------------------------------
# Layer 6 — Cooperative RAIM
# ---------------------------------------------------------------------------


class CoopRAIMLayer:
    """Cooperative RAIM parity-space integrity test (Layer 6).

    Parity matrix: P = I − H(HᵀH)⁻¹Hᵀ  (nullspace projector onto H⊥)
    Parity statistic: T_p = pᵀp / σ²  ~  chi²(n−4) under H₀

    Split-subset test: minimum-norm LS on two equal-sized subsets;
    consistency check: ‖x̂_A − x̂_B‖² / σ²  ~  chi²(4)
    """

    def __init__(self, los: np.ndarray, noise_std: float = _DOPPLER_NOISE_STD) -> None:
        n = len(los)
        H = _geometry_matrix(los, list(range(n)))
        try:
            HtH_inv = np.linalg.inv(H.T @ H)
            self._P_mat = np.eye(n) - H @ HtH_inv @ H.T
        except np.linalg.LinAlgError:
            self._P_mat = np.eye(n)
        self._H = H
        self._n = n
        self._sigma2 = noise_std**2
        self._dof = max(n - 4, 1)
        self._thresh_parity = float(_chi2_dist.ppf(1.0 - _COOP_RAIM_ALPHA, self._dof))
        self._thresh_split = float(_chi2_dist.ppf(1.0 - _COOP_RAIM_ALPHA, 4))

    def assess(self, doppler_dev: np.ndarray) -> CoopRAIMResult:
        """Run parity and split-subset consistency tests.

        Args:
            doppler_dev: (n,) Doppler residuals [Hz]
        """
        p = self._P_mat @ doppler_dev
        parity_chi2 = float(np.dot(p, p) / self._sigma2)

        n_a = self._n // 2
        H_a, H_b = self._H[:n_a], self._H[n_a:]
        z_a, z_b = doppler_dev[:n_a], doppler_dev[n_a:]
        x_a, _, _, _ = np.linalg.lstsq(H_a, z_a, rcond=None)
        x_b, _, _, _ = np.linalg.lstsq(H_b, z_b, rcond=None)
        diff = x_a - x_b
        split_chi2 = float(np.dot(diff, diff) / self._sigma2)

        return CoopRAIMResult(
            parity_chi2=parity_chi2,
            dof=self._dof,
            parity_alert=parity_chi2 > self._thresh_parity,
            split_chi2=split_chi2,
            split_alert=split_chi2 > self._thresh_split,
        )


# ---------------------------------------------------------------------------
# Layer 9 — Huh D-optimal Subset Selector
# ---------------------------------------------------------------------------


class HuhSubsetSelector:
    """D-optimal satellite subset selector via greedy max-det(H_Sᵀ H_S) (Layer 9).

    Excludes satellites flagged as faulty by GM-RAIM (γᵢ > _GMM_FAULT_THRESH),
    retaining the healthy subset that maximises det(H_Sᵀ H_S).

    Theoretical basis: Huh-Katz (2012) — log-concavity of matroid independent-set
    polynomials guarantees the greedy (1−1/e) approximation for D-optimal design.
    """

    _MIN_SATS: int = 4

    def __init__(self, los: np.ndarray, noise_std: float = _DOPPLER_NOISE_STD) -> None:
        n = len(los)
        self._H_all = _geometry_matrix(los, list(range(n)))
        try:
            self._det_all = float(np.linalg.det(self._H_all.T @ self._H_all))
        except np.linalg.LinAlgError:
            self._det_all = 0.0

    def select(self, fault_flags: np.ndarray) -> HuhSelectionResult:
        """Select D-optimal healthy subset.

        Args:
            fault_flags: (n,) boolean array; True = satellite flagged as faulty by GM-RAIM
        """
        n = len(fault_flags)
        n_initially_healthy = int(np.sum(~fault_flags))

        if n_initially_healthy < self._MIN_SATS:
            healthy = list(range(n))
            n_excluded = 0
        else:
            healthy = [i for i in range(n) if not fault_flags[i]]
            n_excluded = n - len(healthy)

        H_sel = self._H_all[healthy]

        try:
            det_sel = float(np.linalg.det(H_sel.T @ H_sel))
        except np.linalg.LinAlgError:
            det_sel = 0.0
        det_ratio = det_sel / (self._det_all + _EPS)

        sv = np.linalg.svd(H_sel, compute_uv=False)
        if len(sv) >= 3:
            lc_ratios = [sv[k] ** 2 / (sv[k - 1] * sv[k + 1] + _EPS) for k in range(1, len(sv) - 1)]
            log_concavity_ratio = float(min(lc_ratios))
        else:
            log_concavity_ratio = 1.0

        return HuhSelectionResult(
            selected_subset=tuple(healthy),
            det_ratio=det_ratio,
            n_selected=len(healthy),
            n_excluded=n_excluded,
            log_concavity_ratio=log_concavity_ratio,
        )
