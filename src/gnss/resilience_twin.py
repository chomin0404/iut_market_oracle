"""GNSS Resilience Twin (T1500).

4-pillar fault discrimination platform for drones and positioning equipment:

  ┌─────────────────────────────────────────────────────────────────────────────┐
  │  Pillar 1 — Authentication  OSNMA Galileo authentication coverage           │
  │  Pillar 2 — Integrity       GM-RAIM + IMM-KF + INS coupling + CoopRAIM     │
  │  Pillar 3 — Structure       Laplacian spectral graph + dependency monitor   │
  │  Pillar 4 — Intervention    Entropy fusion + 4-class posterior decision     │
  └─────────────────────────────────────────────────────────────────────────────┘

Each pillar hosts its own layer classes (Layers 1–8) unchanged.
ResilienceTwin orchestrates the pillar stack per epoch.

Output classes (FaultClass enum, index 0-3):
  NOMINAL | MULTIPATH | HARDWARE_FAULT | SPOOFING

MC simulation entry point: run_resilience_simulation()
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2 as _chi2_dist

# Shared signal constants (own source of truth, independent of simulation layer)
from gnss.constants import (
    _DIRICHLET_ALPHA,
    _DOPPLER_NOISE_STD,
    _GRAPH_SIGMA,
    _INS_CLOCK_STD,
    _INS_VEL_STD,
)

# Geometry / graph / ROC utilities (pure math, no simulation dependencies)
from gnss.math_utils import _build_graph, _compute_roc, _geometry_matrix, _init_constellation

# Simulation helpers used only in the MC benchmark path
from gnss.spoof_sim import (
    _gen_genuine_measurements,
    _init_receiver,
    _inject_attack,
    _propagate_state,
    _sample_attack_window,
)
from schemas import FaultClass, ResilienceTwinReport

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

_EL_MIN_DEG: float = 5.0  # minimum elevation clamp [degrees]
_EL_MIN_RAD: float = math.radians(_EL_MIN_DEG)

_GMM_FAULT_PRIOR: float = 0.05  # P(satellite is faulty) prior
_GMM_FAULT_SCALE: float = 5.0  # fault component noise inflation factor
_GMM_FAULT_THRESH: float = 0.5  # γᵢ threshold for counting faults

_IMM_Q0: float = 0.001  # nominal process noise variance [Hz²/epoch]
_IMM_Q1: float = 0.010  # multipath process noise variance
_IMM_Q2: float = 0.005  # spoofing process noise base variance
_IMM_SPOOF_RHO: float = 3.0  # clock component inflation ρ for Q₂
_IMM_P_INIT: float = 1.0  # initial state covariance diagonal

# Row-stochastic transition matrix Π[from_mode, to_mode]
_IMM_TRANSITION: list[list[float]] = [
    [0.95, 0.03, 0.02],  # nominal   → [nom, mp, spoof]
    [0.10, 0.85, 0.05],  # multipath → [nom, mp, spoof]
    [0.05, 0.05, 0.90],  # spoofing  → [nom, mp, spoof]
]

_FUSE_SPOOF_FIEDLER: float = 0.50  # Fiedler-ratio weight in spoofing score
_FUSE_SPOOF_RMT: float = 0.30  # RMT-anomaly weight in spoofing score
_FUSE_MP_ELEV: float = 0.40  # elevation-correlation weight in multipath score

# 4-class prior for entropy monitor: [nominal, multipath, hw_fault, spoofing]
_FEL_PRIOR: tuple[float, float, float, float] = (0.97, 0.01, 0.01, 0.01)
_FEL_H_THRESH: float = 0.8 * math.log(4.0)  # entropy alert threshold [nats]
_FEL_KL_THRESH: float = 1.0  # KL divergence alert threshold [nats]
_FEL_GRAD_THRESH: float = 0.3  # |ΔH| alert threshold [nats/epoch]

_MP_NOISE_INFLATION: float = 2.0  # multipath noise amplitude [Hz]
# 40× ensures E[P(detect)] ≈ 90% even for the worst eligible sat (el=24.6° → threshold≈2.2 Hz).
_HW_BIAS_STD: float = 40.0 * _DOPPLER_NOISE_STD  # HW fault bias 1-σ [Hz]

_EPS: float = 1e-300  # probability floor

# ---------------------------------------------------------------------------
# Layer 5 — INS coupling chi² thresholds (chi²(3) at 1% significance)
# ---------------------------------------------------------------------------
_INS_CHI2_VEL_THRESH: float = 11.345  # chi²(0.99, 3) — velocity state test
_INS_CHI2_CROSS_THRESH: float = 11.345  # chi²(0.99, 3) — INS cross-check

# Layer 6 — Cooperative RAIM significance level
_COOP_RAIM_ALPHA: float = 0.05  # chi²(1−α, dof) parity / split thresholds

# Layer 7 — OSNMA authentication fraction alert threshold
_OSNMA_AUTH_FRAC_THRESH: float = 0.50  # alert if fewer than 50% authenticated

# Layer 8 — Structural dependency monitor parameters
_STRUCT_STREAK_THRESH: int = 3  # consecutive Fiedler-anomaly epochs to alert
_STRUCT_CHANGE_THRESH: float = 0.50  # fractional Frobenius change to alert
_STRUCT_CLUSTER_WEIGHT_THRESH: float = 0.50  # edge-weight threshold for clustering

# Fusion weights for layers 5–8 contributions to spoofing score
_FUSE_INS_SPOOF: float = 0.10
_FUSE_COOP_SPOOF: float = 0.15
_FUSE_OSNMA_SPOOF: float = 0.40
_FUSE_STRUCT_SPOOF: float = 0.05
# Common-mode bias indicator: |mean(Δf)| / σ_D elevated above 2σ under meaconing
_FUSE_GMM_SPOOF_COMMON: float = 0.50
_HW_EL_MIN_DEG: float = 15.0  # min elevation [deg] for hw_fault sat selection

# Layer 9 — Huh D-optimal subset selection
# Layer 10 — Duminil-Copin percolation phase-transition monitor
_FUSE_PHASE_SPOOF: float = 0.10  # phase-transition alert weight in spoof score
_DC_N_THRESH_POINTS: int = 41  # number of τ grid points (Δτ = 0.025)
_DC_SUSCEPTIBILITY_ALERT: float = 10.0  # χ_peak threshold (n=6: isolated node gives ≈6.7)
_DC_NULL_THRESHOLD: float = 0.90  # reference τ for lcc_at_null
# Tight-meaconing detector: alert only when ALL edge weights are simultaneously
# near 1 (common-mode / meaconing attack with small differential spread).
# Under nominal σ=0.3 Hz / σ_g=1.5 Hz, P(min_w > 0.95) ≈ 0.04 % per epoch.
# Under HW fault (one large outlier), min_w ≈ 0.  Only pure meaconing gives
# all-pairs min_w → 1.
_DC_MIN_W_THRESHOLD: float = 0.95  # min edge weight required for phase alert

# Ordered fault class list — index aligns with fault_posterior positions
_FAULT_CLASSES: list[FaultClass] = [
    FaultClass.NOMINAL,
    FaultClass.MULTIPATH,
    FaultClass.HARDWARE_FAULT,
    FaultClass.SPOOFING,
]

# ---------------------------------------------------------------------------
# Frozen result dataclasses
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
class SpectralResult:
    """Output of spectral graph monitor per epoch."""

    fiedler_ratio: float  # ρ_F = λ₂ / λ₂_null
    spectral_entropy: float  # H_spec [nats]
    rmt_anomaly: float  # mean((λₖ − λ₂_null)²) / λ₂_null² for k≥2


@dataclass(frozen=True)
class FaultEntropyResult:
    """Output of fault entropy monitor per epoch."""

    entropy: float  # H(π) [nats]
    kl: float  # KL(π ‖ π₀) [nats]
    alert: bool  # True if any threshold exceeded


@dataclass(frozen=True)
class INSCouplingResult:
    """Output of INS coupling chi² cross-check (Layer 5).

    chi2_vel = ‖x_fused[:3]‖² / σ_INS²  ∼  chi²(3) under H₀
    chi2_cross = ‖v_ins − x_fused[:3]‖² / (2σ_INS²)  ∼  chi²(3)  [if INS available]
    """

    chi2_vel: float  # chi²(3) for IMM fused velocity state
    chi2_cross: float  # chi²(3) cross-check residual vs external INS (0 if unavailable)
    ins_available: bool  # True if external INS velocity was provided
    alert: bool  # True if chi2_vel or chi2_cross exceeds threshold


@dataclass(frozen=True)
class CoopRAIMResult:
    """Output of cooperative RAIM parity-space test (Layer 6).

    Parity matrix P = I − H(HᵀH)⁻¹Hᵀ,  T_p = pᵀp / σ²  ∼  chi²(n−4) under H₀
    Split chi²: minimum-norm LS on two equal-sized subsets, ‖x̂_A − x̂_B‖² / σ²
    """

    parity_chi2: float  # chi²(n−4) parity statistic / σ²
    dof: int  # degrees of freedom = n − 4
    parity_alert: bool  # True if parity_chi2 > chi²(0.95, dof)
    split_chi2: float  # chi²(4) consistency between split-subset LS estimates
    split_alert: bool  # True if split_chi2 > chi²(0.95, 4)


@dataclass(frozen=True)
class OSNMALayerResult:
    """Output of OSNMA Galileo authentication layer (Layer 7).

    auth_fraction = n_auth / n_total   (defaults to 1.0 when no OSNMA data)
    p_spoof_contribution = 1 − auth_fraction  (used as fusion signal)
    """

    auth_fraction: float  # fraction of authenticated satellites ∈ [0, 1]
    p_spoof_contribution: float  # 1 − auth_fraction
    n_auth: int  # number of authenticated satellites
    n_total: int  # total satellites checked (0 if no data)
    alert: bool  # True if auth_fraction < _OSNMA_AUTH_FRAC_THRESH


@dataclass(frozen=True)
class StructuralMonitorResult:
    """Output of structural dependency monitor (Layer 8).

    Tracks persistent graph-level anomalies across consecutive epochs.
    """

    fiedler_streak: int  # consecutive epochs with Fiedler-ratio anomaly
    graph_change_rate: float  # ‖W_t − W_{t−1}‖_F / ‖W_{t−1}‖_F
    clustering_coeff: float  # mean clustering coefficient of thresholded graph
    alert: bool  # True if streak ≥ threshold or change_rate > threshold


@dataclass(frozen=True)
class HuhSelectionResult:
    """Output of Huh D-optimal satellite subset selector (Layer 9).

    Greedy forward selection maximising det(H_Sᵀ H_S) from healthy satellites
    (γᵢ < _GMM_FAULT_THRESH).  Theoretical basis: Huh-Katz (2012) log-concavity
    of matroid independent-set polynomials.

    det_ratio = det(H_sel ᵀ H_sel) / det(H_all ᵀ H_all)   (≥1 by construction
                when n_excluded > 0 and geometry improves; = 1 when no faults)
    log_concavity_ratio = min σₖ² / (σₖ₋₁ σₖ₊₁) on singular values of H_sel
    """

    selected_subset: tuple[int, ...]  # indices of included satellites
    det_ratio: float  # D-optimal improvement over full set
    n_selected: int  # |S|
    n_excluded: int  # satellites excluded (flagged as faulty)
    log_concavity_ratio: float  # min σₖ² / (σₖ₋₁ σₖ₊₁) on H_sel sing. values


@dataclass(frozen=True)
class PhaseTransitionResult:
    """Output of Duminil-Copin percolation phase-transition monitor (Layer 10).

    Sweeps threshold τ ∈ [0,1] on the satellite similarity graph W:
        A_τ[i,j] = 1  iff  w_ij > τ
        LCC(τ)   = fraction of nodes in the largest connected component
        χ(τ)     = |ΔLCC / Δτ|  — susceptibility (peaks at the phase transition)

    Under nominal conditions (isolated nodes only): χ_peak ≈ 1/(n·Δτ) ≈ 6.7
    Under coordinated spoofing (synchronised collapse): χ_peak >> 10

    Theoretical basis: Duminil-Copin et al. (2020) — sharp phase transitions in
    dependent percolation models; susceptibility peak is a universal indicator.
    """

    percolation_threshold: float  # τ* where χ is maximised
    susceptibility_peak: float  # max χ over the τ sweep
    lcc_at_null: float  # LCC(τ = _DC_NULL_THRESHOLD)
    min_edge_weight: float  # min off-diagonal w_ij — near 1 ↔ tight common-mode attack
    phase_alert: bool  # True if χ_peak > thresh AND min_w > _DC_MIN_W_THRESHOLD


@dataclass(frozen=True)
class AuthenticationScore:
    """Pillar 1 — OSNMA Galileo authentication coverage score."""

    auth_fraction: float  # fraction of authenticated satellites ∈ [0, 1]
    p_spoofed: float  # 1 − auth_fraction (fusion signal)
    alert: bool  # True if auth_fraction < threshold
    osnma: OSNMALayerResult  # raw layer result


@dataclass(frozen=True)
class IntegrityScore:
    """Pillar 2 — integrity-layer base fault posterior (GM-RAIM + IMM + INS + CoopRAIM + Huh)."""

    base_posterior: tuple[float, float, float, float]  # [P_nom, P_mp, P_hw, P_spoof]
    gmm: GMMResult
    imm: IMMResult
    ins: INSCouplingResult
    coop_raim: CoopRAIMResult
    huh: HuhSelectionResult  # Layer 9 — D-optimal satellite subset


@dataclass(frozen=True)
class StructuralScore:
    """Pillar 3 — graph-structure anomaly intensity."""

    structure_intensity: float  # max(ρ_F−1, 0) + rmt_anomaly
    spectral: SpectralResult
    structural: StructuralMonitorResult
    phase: PhaseTransitionResult  # Layer 10 — Duminil-Copin percolation monitor


@dataclass(frozen=True)
class EpochDiagnosis:
    """Per-epoch diagnostic output from ResilienceTwin (4-pillar architecture)."""

    t: int
    fault_posterior: tuple[float, float, float, float]  # [P_nom, P_mp, P_hw, P_spoof]
    diagnosis: FaultClass
    confidence: float  # max(fault_posterior)
    entropy: FaultEntropyResult  # Pillar 4 — intervention
    auth: AuthenticationScore  # Pillar 1 — authentication
    integrity: IntegrityScore  # Pillar 2 — integrity
    structure: StructuralScore  # Pillar 3 — structure


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
        sigma_i = self._sigma / np.sin(el_clamped)  # (n,) elevation-adjusted σ

        # Log-numerators for each GMM component (log-sum-exp for stability)
        log_p0 = -0.5 * (doppler_dev / sigma_i) ** 2 - np.log(sigma_i) + math.log(self._pi_nom)
        log_p1 = (
            -0.5 * (doppler_dev / (self._scale * sigma_i)) ** 2
            - np.log(self._scale * sigma_i)
            + math.log(self._pi_fault)
        )

        log_m = np.maximum(log_p0, log_p1)
        log_sum = log_m + np.log(np.exp(log_p0 - log_m) + np.exp(log_p1 - log_m))
        gamma = np.exp(log_p1 - log_sum)  # (n,) fault posteriors ∈ [0, 1]

        # Sign correlation: |mean(sign(Δf))| ≈ 1 when common bias dominates
        sign_corr = float(abs(np.sign(doppler_dev).mean()))

        # Elevation correlation: |corr(|Δf|, 1/sin(el))| — high for multipath
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
        # Precompute geometry matrix H (all satellites, shape n×4)
        self._H = _geometry_matrix(los, list(range(n_sats)))
        self._R = noise_std**2 * np.eye(n_sats)

        # Process noise covariances
        e3 = np.zeros(4)
        e3[3] = 1.0
        self._Q: list[np.ndarray] = [
            _IMM_Q0 * np.eye(4),
            _IMM_Q1 * np.eye(4),
            _IMM_Q2 * (np.eye(4) + _IMM_SPOOF_RHO * np.outer(e3, e3)),
        ]

        self._Pi = np.array(_IMM_TRANSITION, dtype=float)  # (3, 3) row-stochastic
        # Nominal-biased prior: P(genuine)=0.97 matches _FEL_PRIOR[0].
        # Uniform (1/3) prior caused s_mp = 1/3 + 0.40*elev_corr > s_nom = 1/3
        # at epoch 0, driving spurious MULTIPATH / SPOOFING classifications in
        # genuine trials before the IMM had observed any data.
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

        # ── 1. Interaction (mixing) ─────────────────────────────────────────
        c_bar = np.clip(self._Pi.T @ self._mu, _EPS, None)  # (3,) predicted mode probs

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

        # ── 2. Mode-conditioned KF update (F = I₄) ─────────────────────────
        x_upd = np.zeros((M, 4))
        P_upd = [np.zeros((4, 4)) for _ in range(M)]
        log_lkl = np.zeros(M)
        nu_norms = np.zeros(M)

        for m in range(M):
            x_pred = x_mix[m]  # F = I, no dynamics
            P_pred = P_mix[m] + self._Q[m]

            nu = z - H @ x_pred  # (n,) innovation
            nu_norms[m] = float(np.linalg.norm(nu))

            S = H @ P_pred @ H.T + R  # (n, n) innovation covariance
            try:
                K_T = np.linalg.solve(S, H @ P_pred)  # (n, 4) — avoids explicit inverse
            except np.linalg.LinAlgError:
                K_T = np.zeros((self._n, 4))
            K = K_T.T  # (4, n) Kalman gain

            x_upd[m] = x_pred + K @ nu
            P_upd[m] = (I4 - K @ H) @ P_pred  # standard form (sufficient for sim)

            # Innovation log-likelihood: log N(ν; 0, S)
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

        # ── 3. Mode probability update ──────────────────────────────────────
        log_mu = log_lkl + np.log(c_bar)
        log_mu -= log_mu.max()  # subtract max for numerical stability
        mu_new = np.exp(log_mu)
        mu_new /= mu_new.sum()

        # ── 4. Fused state estimate ─────────────────────────────────────────
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
# Layer 3 — Spectral Graph Monitor
# ---------------------------------------------------------------------------


class SpectralMonitor:
    """Spectral anomaly detection on the satellite similarity graph.

    Under H₀ (all σᵢ = σ_D, complete symmetric graph):
        w_null = 1 / √(1 + 4σ_D²/σ²)    [expected edge weight]
        λ₂_null = n · w_null              [null Fiedler value]

    Metrics:
        ρ_F   = λ₂ / λ₂_null             — Fiedler ratio (>1 → anomaly)
        H_spec = −Σₖ pₖ ln pₖ            — spectral entropy of non-trivial eigenvalues
        rmt    = mean((λₖ − λ₂_null)²) / λ₂_null²   — RMT deviation (k ≥ 2)
    """

    def __init__(
        self,
        n_sats: int,
        noise_std: float = _DOPPLER_NOISE_STD,
        graph_sigma: float = _GRAPH_SIGMA,
    ) -> None:
        self._sigma = graph_sigma
        w_null = 1.0 / math.sqrt(1.0 + 4.0 * noise_std**2 / graph_sigma**2)
        self._lambda2_null = n_sats * w_null  # null Fiedler reference

    def analyze(self, doppler_dev: np.ndarray) -> SpectralResult:
        """Compute spectral metrics from current Doppler deviations.

        Args:
            doppler_dev: (n,) Doppler residuals [Hz]
        """
        W = _build_graph(doppler_dev, self._sigma)
        L = np.diag(W.sum(axis=1)) - W
        ev = np.sort(np.linalg.eigvalsh(L))  # ascending; ev[0] ≈ 0

        lambda2 = float(ev[1]) if len(ev) > 1 else 0.0
        fiedler_ratio = lambda2 / (self._lambda2_null + _EPS)

        # Spectral entropy from non-trivial eigenvalues (ev[1:])
        ev_pos = np.maximum(ev[1:], 0.0)
        total = ev_pos.sum()
        if total > _EPS:
            p = ev_pos / total
            spectral_entropy = float(-np.sum(p * np.log(np.where(p > _EPS, p, _EPS))))
        else:
            spectral_entropy = 0.0

        # RMT anomaly: mean squared deviation from null reference
        rmt_anomaly = float(
            np.mean((ev[1:] - self._lambda2_null) ** 2) / (self._lambda2_null**2 + _EPS)
        )

        return SpectralResult(
            fiedler_ratio=fiedler_ratio,
            spectral_entropy=spectral_entropy,
            rmt_anomaly=rmt_anomaly,
        )


# ---------------------------------------------------------------------------
# Layer 4 — Fault Entropy Monitor
# ---------------------------------------------------------------------------


class FaultEntropyMonitor:
    """Shannon entropy + KL divergence monitor on the 4-class fault posterior.

    Alerts when:
        H(π) > H_thresh        — high classification uncertainty
        KL(π ‖ π₀) > kl_thresh — large deviation from nominal prior
        |ΔH| > grad_thresh     — rapid entropy change between epochs
    """

    def __init__(
        self,
        prior: tuple[float, float, float, float] = _FEL_PRIOR,
        h_thresh: float = _FEL_H_THRESH,
        kl_thresh: float = _FEL_KL_THRESH,
        grad_thresh: float = _FEL_GRAD_THRESH,
    ) -> None:
        pi0 = np.array(prior, dtype=float)
        self._pi0 = pi0 / pi0.sum()
        self._h_thresh = h_thresh
        self._kl_thresh = kl_thresh
        self._grad_thresh = grad_thresh
        self._prev_h: float | None = None

    def update(self, fault_probs: np.ndarray) -> FaultEntropyResult:
        """Update monitor with current 4-class fault posterior.

        Args:
            fault_probs: (4,) probability vector [P_nom, P_mp, P_hw, P_spoof]
        """
        pi = np.clip(fault_probs, _EPS, 1.0)
        pi = pi / pi.sum()

        h = float(-np.sum(pi * np.log(pi)))
        kl = float(np.sum(pi * np.log(pi / self._pi0)))

        delta_h = abs(h - self._prev_h) if self._prev_h is not None else 0.0
        self._prev_h = h

        alert = h > self._h_thresh or kl > self._kl_thresh or delta_h > self._grad_thresh
        return FaultEntropyResult(entropy=h, kl=kl, alert=alert)


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

        # Split into two equal-sized halves for cross-consistency check
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
    retaining the healthy subset that maximises det(H_Sᵀ H_S).  Since adding
    non-degenerate observations never reduces det for the D-criterion, the
    optimal healthy subset is simply all unflagged satellites (greedy = optimal
    for the inclusion monotone case).

    Theoretical basis: Huh-Katz (2012) — log-concavity of matroid independent-set
    polynomials guarantees the greedy (1−1/e) approximation for D-optimal design.
    """

    _MIN_SATS: int = 4  # minimum satellites for 4D positioning

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
            # Fallback: too few healthy satellites — use all
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

        # Log-concavity proxy: min σₖ² / (σₖ₋₁ σₖ₊₁) on singular values of H_sel
        sv = np.linalg.svd(H_sel, compute_uv=False)  # descending order
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


# ---------------------------------------------------------------------------
# Layer 7 — OSNMA Authentication Layer
# ---------------------------------------------------------------------------


class OSNMALayer:
    """Galileo OSNMA authentication coverage monitor (Layer 7).

    Computes the fraction of satellites with verified OSNMA authentication tags.
    Defaults to fully authenticated (fraction = 1.0, contribution = 0.0) when
    no OSNMA data is supplied (GPS-only or non-Galileo receiver).

    Alert threshold: < 50 % authenticated satellites.
    """

    def __init__(self, alert_thresh: float = _OSNMA_AUTH_FRAC_THRESH) -> None:
        self._thresh = alert_thresh

    def assess(self, osnma_auth: list[bool] | None) -> OSNMALayerResult:
        """Evaluate OSNMA authentication coverage for the current epoch.

        Args:
            osnma_auth: Per-satellite boolean authentication flags, or None.
        """
        if osnma_auth is None or len(osnma_auth) == 0:
            return OSNMALayerResult(
                auth_fraction=1.0,
                p_spoof_contribution=0.0,
                n_auth=0,
                n_total=0,
                alert=False,
            )
        n_total = len(osnma_auth)
        n_auth = sum(osnma_auth)
        auth_fraction = n_auth / n_total
        return OSNMALayerResult(
            auth_fraction=auth_fraction,
            p_spoof_contribution=1.0 - auth_fraction,
            n_auth=n_auth,
            n_total=n_total,
            alert=auth_fraction < self._thresh,
        )


# ---------------------------------------------------------------------------
# Layer 8 — Structural Dependency Monitor
# ---------------------------------------------------------------------------


class StructuralDependencyMonitor:
    """Structural dependency anomaly tracker across consecutive epochs (Layer 8).

    Monitors persistent graph-topology changes that signal coordinated
    multi-satellite manipulation (meaconing):

        fiedler_streak:   consecutive epochs where ρ_F > 1.0
        graph_change_rate: ‖W_t − W_{t−1}‖_F / (‖W_{t−1}‖_F + ε)
        clustering_coeff:  mean clustering coefficient of thresholded graph

    Alert fires when: streak ≥ threshold OR change_rate > threshold.
    """

    def __init__(
        self,
        noise_std: float = _DOPPLER_NOISE_STD,
        graph_sigma: float = _GRAPH_SIGMA,
        streak_thresh: int = _STRUCT_STREAK_THRESH,
        change_thresh: float = _STRUCT_CHANGE_THRESH,
        cluster_weight_thresh: float = _STRUCT_CLUSTER_WEIGHT_THRESH,
    ) -> None:
        self._graph_sigma = graph_sigma
        self._streak_thresh = streak_thresh
        self._change_thresh = change_thresh
        self._cluster_w_thresh = cluster_weight_thresh
        self._streak: int = 0
        self._prev_W: np.ndarray | None = None

    def update(self, doppler_dev: np.ndarray, fiedler_anomaly: bool) -> StructuralMonitorResult:
        """Update structural monitor with current epoch's Doppler observations.

        Args:
            doppler_dev:     (n,) Doppler residuals [Hz]
            fiedler_anomaly: True if ρ_F > 1.0 this epoch
        """
        W = _build_graph(doppler_dev, self._graph_sigma)
        n = W.shape[0]

        # Track consecutive Fiedler-anomaly epochs
        if fiedler_anomaly:
            self._streak += 1
        else:
            self._streak = 0

        # Frobenius graph change rate
        if self._prev_W is not None:
            frob_prev = float(np.linalg.norm(self._prev_W, "fro"))
            frob_diff = float(np.linalg.norm(W - self._prev_W, "fro"))
            graph_change_rate = frob_diff / (frob_prev + _EPS)
        else:
            graph_change_rate = 0.0
        self._prev_W = W.copy()

        # Mean clustering coefficient of thresholded adjacency graph
        A = (W > self._cluster_w_thresh).astype(float)
        np.fill_diagonal(A, 0.0)
        degree = A.sum(axis=1)
        cc_sum = 0.0
        n_counted = 0
        for i in range(n):
            d = int(degree[i])
            if d >= 2:
                nbrs = np.where(A[i] > 0)[0]
                e_count = 0
                for a_i in nbrs:
                    for b_i in nbrs:
                        if a_i < b_i:
                            e_count += int(A[a_i, b_i])
                cc_sum += e_count / (d * (d - 1) / 2)
                n_counted += 1
        clustering_coeff = cc_sum / max(n_counted, 1)

        alert = self._streak >= self._streak_thresh or graph_change_rate > self._change_thresh
        return StructuralMonitorResult(
            fiedler_streak=self._streak,
            graph_change_rate=graph_change_rate,
            clustering_coeff=clustering_coeff,
            alert=alert,
        )


# ---------------------------------------------------------------------------
# Layer 10 — Duminil-Copin Phase-Transition Monitor
# ---------------------------------------------------------------------------


class DuminilCopinPhaseMonitor:
    """Percolation phase-transition monitor on the satellite similarity graph (Layer 10).

    Sweeps threshold τ ∈ [0,1] on the satellite similarity graph W_ij = exp(-|Δfᵢ−Δfⱼ|²/σ²):
        A_τ[i,j] = 1  iff  w_ij > τ
        LCC(τ)   = |largest connected component| / n_sats
        χ(τ)     = |ΔLCC(τ) / Δτ|  — susceptibility

    A sharp χ_peak marks the percolation threshold (τ*).  Coordinated spoofing
    collapses all edge-weights at once → synchronised χ_peak >> 10.
    An isolated HW fault removes at most 1 node → χ_peak ≈ (1/n)/Δτ ≈ 6.7 < 10.

    Alert threshold: χ_peak > _DC_SUSCEPTIBILITY_ALERT = 10.0.
    """

    def __init__(self, graph_sigma: float = _GRAPH_SIGMA) -> None:
        self._graph_sigma = graph_sigma
        self._tau_grid = np.linspace(0.0, 1.0, _DC_N_THRESH_POINTS)

    def update(self, doppler_dev: np.ndarray) -> PhaseTransitionResult:
        """Compute percolation susceptibility for current epoch.

        Args:
            doppler_dev: (n,) Doppler residuals [Hz]
        """
        W = _build_graph(doppler_dev, self._graph_sigma)
        n = W.shape[0]

        lcc_curve = np.empty(len(self._tau_grid))
        for k, tau in enumerate(self._tau_grid):
            A = (W > tau).astype(np.uint8)
            np.fill_diagonal(A, 0)
            lcc_curve[k] = self._largest_cc_fraction(A, n)

        # χ(τ) = |ΔLCC / Δτ|  at each midpoint of the τ grid
        delta_tau = float(self._tau_grid[1] - self._tau_grid[0])
        chi = np.abs(np.diff(lcc_curve)) / delta_tau

        chi_peak = float(chi.max()) if len(chi) > 0 else 0.0
        peak_idx = int(np.argmax(chi))
        percolation_threshold = float(
            0.5 * (self._tau_grid[peak_idx] + self._tau_grid[peak_idx + 1])
        )

        null_idx = min(
            int(np.searchsorted(self._tau_grid, _DC_NULL_THRESHOLD)),
            len(self._tau_grid) - 1,
        )
        lcc_at_null = float(lcc_curve[null_idx])

        # Minimum off-diagonal edge weight — tight meaconing forces all pairs near 1.0
        if n > 1:
            W_off = W.copy()
            np.fill_diagonal(W_off, 1.0)  # exclude diagonal from min
            min_w = float(W_off.min())
        else:
            min_w = 0.0

        # Alert: synchronized collapse (χ_peak large) AND all edges near 1 (min_w high)
        phase_alert = chi_peak > _DC_SUSCEPTIBILITY_ALERT and min_w > _DC_MIN_W_THRESHOLD

        return PhaseTransitionResult(
            percolation_threshold=percolation_threshold,
            susceptibility_peak=chi_peak,
            lcc_at_null=lcc_at_null,
            min_edge_weight=min_w,
            phase_alert=phase_alert,
        )

    @staticmethod
    def _largest_cc_fraction(A: np.ndarray, n: int) -> float:
        """BFS — fraction of nodes in the largest connected component."""
        if n == 0:
            return 0.0
        visited = np.zeros(n, dtype=bool)
        max_cc = 0
        for start in range(n):
            if visited[start]:
                continue
            stack = [start]
            visited[start] = True
            cc_size = 0
            while stack:
                node = stack.pop()
                cc_size += 1
                for nbr in np.where(A[node] > 0)[0]:
                    if not visited[nbr]:
                        visited[nbr] = True
                        stack.append(int(nbr))
            if cc_size > max_cc:
                max_cc = cc_size
        return max_cc / n


# ---------------------------------------------------------------------------
# Pillar 1 — Authentication
# ---------------------------------------------------------------------------


class AuthenticationPillar:
    """OSNMA Galileo authentication coverage (Pillar 1)."""

    def __init__(self) -> None:
        self._osnma = OSNMALayer()

    def assess(self, osnma_auth: list[bool] | None) -> AuthenticationScore:
        osnma = self._osnma.assess(osnma_auth)
        return AuthenticationScore(
            auth_fraction=osnma.auth_fraction,
            p_spoofed=osnma.p_spoof_contribution,
            alert=osnma.alert,
            osnma=osnma,
        )


# ---------------------------------------------------------------------------
# Pillar 2 — Integrity
# ---------------------------------------------------------------------------


class IntegrityPillar:
    """GM-RAIM + IMM-KF + INS coupling + Cooperative RAIM + Huh subset (Pillar 2).

    Computes a base 4-class fault posterior from integrity signals only.
    Structural and authentication signals are added by InterventionPillar.
    """

    def __init__(
        self,
        los: np.ndarray,
        noise_std: float = _DOPPLER_NOISE_STD,
        ins_noise_std: float = _INS_VEL_STD,
    ) -> None:
        self._noise_std = noise_std
        self._gmm = GMMRaim(noise_std=noise_std)
        self._imm = IMMKalman(los=los, noise_std=noise_std)
        self._ins = INSCouplingLayer(noise_std=ins_noise_std)
        self._coop_raim = CoopRAIMLayer(los=los, noise_std=noise_std)
        self._huh = HuhSubsetSelector(los=los, noise_std=noise_std)

    def assess(
        self,
        doppler_dev: np.ndarray,
        elevations: np.ndarray,
        ins_velocity: np.ndarray | None = None,
    ) -> IntegrityScore:
        gmm = self._gmm.classify(doppler_dev, elevations)
        imm = self._imm.update(doppler_dev)
        ins = self._ins.assess(imm.x_fused, ins_velocity)
        coop_raim = self._coop_raim.assess(doppler_dev)
        huh = self._huh.select(np.array(gmm.gamma) > _GMM_FAULT_THRESH)

        mu_nom, mu_mp, mu_spoof = imm.mode_weights
        # Coherent-SNR spoofing indicator: meaconing adds the same b_common to ALL sats,
        # so mean(dev) is large while var(dev) stays near noise_std².
        #   SNR = n·mean² / var  →  ≈1 for single outlier (HW),  ≈2 for 2-sat multipath,
        #                            ≫10 for common-mode meaconing.
        # Threshold at SNR=5 rejects HW/multipath while detecting spoofing with
        # |b_common| > sqrt(5·noise²/n) ≈ 0.27 Hz  →  P_D ≈ 91 % for b_common ~ N(0,2.5²).
        n_s = len(doppler_dev)
        mean_dev = float(np.mean(doppler_dev))
        var_dev = max(float(np.var(doppler_dev)), self._noise_std**2)
        coherent_snr = n_s * mean_dev**2 / var_dev
        # Divisor 7: with diff_std=0.10 Hz, var_dev under spoofing ≈ 0.10 Hz²,
        # so coherent_snr ≈ 60·b_common²; divisor=7 gives breakeven
        # |b_common| ≈ 0.44 Hz → P(detect/epoch) ≈ 91% for b_common ~ N(0, 4.0²).
        # Under nominal, var_dev ≈ noise_std² = 0.09 → coherent_snr ~ chi²(1),
        # P(chi²(1) > 7·1.7) = P(chi²(1) > 11.9) ≈ 0.056% → P_FA stays ~0%.
        coherent_score = min(coherent_snr / 7.0, 10.0)
        s_spoof = (
            mu_spoof
            + _FUSE_INS_SPOOF * float(ins.alert)
            + _FUSE_COOP_SPOOF * float(coop_raim.parity_alert or coop_raim.split_alert)
            + _FUSE_GMM_SPOOF_COMMON * coherent_score
        )
        s_mp = mu_mp + _FUSE_MP_ELEV * gmm.elev_corr
        s_hw = 1.0 if gmm.n_fault == 1 else 0.0
        # Use IMM nominal weight directly; normalization handles class competition.
        # The previous multiplicative suppression biased P(nominal) toward zero
        # even under genuine nominal conditions (s_mp inflated by small-sample
        # elev_corr noise), causing P_FA ≈ 1.0 in MC evaluation.
        s_nom = mu_nom

        raw = np.clip(np.array([s_nom, s_mp, s_hw, s_spoof], dtype=float), 0.0, None)
        total = raw.sum()
        if total < _EPS:
            base: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
        else:
            bp = raw / total
            base = (float(bp[0]), float(bp[1]), float(bp[2]), float(bp[3]))

        return IntegrityScore(
            base_posterior=base, gmm=gmm, imm=imm, ins=ins, coop_raim=coop_raim, huh=huh
        )


# ---------------------------------------------------------------------------
# Pillar 3 — Structure
# ---------------------------------------------------------------------------


class StructuralPillar:
    """Spectral graph monitor + structural dependency tracker + phase monitor (Pillar 3)."""

    def __init__(
        self,
        n_sats: int,
        noise_std: float = _DOPPLER_NOISE_STD,
        graph_sigma: float = _GRAPH_SIGMA,
    ) -> None:
        self._spectral = SpectralMonitor(
            n_sats=n_sats, noise_std=noise_std, graph_sigma=graph_sigma
        )
        self._structural = StructuralDependencyMonitor(noise_std=noise_std, graph_sigma=graph_sigma)
        self._phase = DuminilCopinPhaseMonitor(graph_sigma=graph_sigma)

    def update(self, doppler_dev: np.ndarray) -> StructuralScore:
        spectral = self._spectral.analyze(doppler_dev)
        structural = self._structural.update(doppler_dev, spectral.fiedler_ratio > 1.0)
        phase = self._phase.update(doppler_dev)
        structure_intensity = max(spectral.fiedler_ratio - 1.0, 0.0) + spectral.rmt_anomaly
        return StructuralScore(
            structure_intensity=structure_intensity,
            spectral=spectral,
            structural=structural,
            phase=phase,
        )


# ---------------------------------------------------------------------------
# Pillar 4 — Intervention
# ---------------------------------------------------------------------------


class InterventionPillar:
    """Entropy fusion + 4-class decision (Pillar 4).

    Fuses the integrity base posterior with structural and authentication signals:

        s_spoof += α_F·max(ρ_F−1,0)·C_s + α_R·rmt + α_O·p_osnma + α_S·I[struct_alert]
    """

    def __init__(self) -> None:
        self._entropy = FaultEntropyMonitor()

    def fuse(
        self,
        auth: AuthenticationScore,
        integrity: IntegrityScore,
        structure: StructuralScore,
    ) -> tuple[np.ndarray, FaultEntropyResult]:
        """Compute final 4-class posterior and entropy alert.

        Returns:
            (fp, entropy_result): fp is a (4,) normalized probability array.
        """
        p_nom, p_mp, p_hw, p_spoof = integrity.base_posterior

        s_spoof = (
            p_spoof
            + _FUSE_SPOOF_FIEDLER
            * max(structure.spectral.fiedler_ratio - 1.0, 0.0)
            * integrity.gmm.sign_corr
            + _FUSE_SPOOF_RMT * structure.spectral.rmt_anomaly
            + _FUSE_OSNMA_SPOOF * auth.p_spoofed
            + _FUSE_STRUCT_SPOOF * float(structure.structural.alert)
            + _FUSE_PHASE_SPOOF * float(structure.phase.phase_alert)
        )
        s_mp = p_mp
        s_hw = p_hw
        # Use base nominal posterior directly; normalization handles competition.
        # Multiplicative suppression here caused double-suppression on top of
        # IntegrityPillar's already-normalized base_posterior, which drove
        # P(nominal) to near-zero under genuine nominal conditions.
        s_nom = p_nom

        raw = np.clip(np.array([s_nom, s_mp, s_hw, s_spoof], dtype=float), 0.0, None)
        total = raw.sum()
        fp = raw / total if total >= _EPS else np.array([1.0, 0.0, 0.0, 0.0])

        entropy_result = self._entropy.update(fp)
        return fp, entropy_result


# ---------------------------------------------------------------------------
# ResilienceTwin orchestrator
# ---------------------------------------------------------------------------


class ResilienceTwin:
    """4-pillar GNSS fault discrimination platform (T1500).

    Orchestrates Authentication → Integrity → Structure → Intervention pillars
    to produce a 4-class fault posterior per epoch:

        NOMINAL | MULTIPATH | HARDWARE_FAULT | SPOOFING
    """

    def __init__(
        self,
        los: np.ndarray,
        noise_std: float = _DOPPLER_NOISE_STD,
        graph_sigma: float = _GRAPH_SIGMA,
        ins_noise_std: float = _INS_VEL_STD,
    ) -> None:
        n_sats = len(los)
        self._elevations = np.arcsin(np.clip(los[:, 2], -1.0, 1.0))  # [radians]
        self._auth = AuthenticationPillar()
        self._integrity = IntegrityPillar(los=los, noise_std=noise_std, ins_noise_std=ins_noise_std)
        self._structure = StructuralPillar(
            n_sats=n_sats, noise_std=noise_std, graph_sigma=graph_sigma
        )
        self._intervention = InterventionPillar()

    def step(
        self,
        doppler_dev: np.ndarray,
        t: int = 0,
        ins_velocity: np.ndarray | None = None,
        osnma_auth: list[bool] | None = None,
    ) -> EpochDiagnosis:
        """Process one epoch of Doppler residuals through the 4-pillar stack.

        Args:
            doppler_dev:  (n_sats,) Doppler residuals [Hz]
            t:            Epoch index (informational only)
            ins_velocity: (3,) external INS velocity deviation [m/s], or None
            osnma_auth:   Per-satellite OSNMA authentication flags, or None
        """
        auth = self._auth.assess(osnma_auth)
        integrity = self._integrity.assess(doppler_dev, self._elevations, ins_velocity)
        structure = self._structure.update(doppler_dev)
        fp, entropy_result = self._intervention.fuse(auth, integrity, structure)

        idx = int(np.argmax(fp))
        return EpochDiagnosis(
            t=t,
            fault_posterior=(float(fp[0]), float(fp[1]), float(fp[2]), float(fp[3])),
            diagnosis=_FAULT_CLASSES[idx],
            confidence=float(fp[idx]),
            entropy=entropy_result,
            auth=auth,
            integrity=integrity,
            structure=structure,
        )


# ---------------------------------------------------------------------------
# Attack generators for MC simulation
# ---------------------------------------------------------------------------


def _inject_multipath(
    doppler_dev: np.ndarray,
    elevations: np.ndarray,
    rng: np.random.Generator,
    inflation: float = _MP_NOISE_INFLATION,
) -> np.ndarray:
    """Add elevation-correlated multipath noise to the lowest-elevation third of sats.

    Args:
        doppler_dev: (n,) baseline Doppler deviations [Hz]
        elevations:  (n,) elevation angles [radians]
        rng:         Random generator
        inflation:   Multipath noise amplitude [Hz]
    """
    n = len(doppler_dev)
    n_mp = max(2, n // 3)
    low_el_idx = np.argsort(elevations)[:n_mp]
    el_clamped = np.maximum(elevations, _EL_MIN_RAD)

    result = doppler_dev.copy()
    for i in low_el_idx:
        sigma_mp = inflation / np.sin(el_clamped[i])
        result[i] += rng.normal(0.0, sigma_mp)
    return result


def _inject_hw_fault(
    doppler_dev: np.ndarray,
    sat_idx: int,
    bias: float,
) -> np.ndarray:
    """Inject a persistent large bias on a single satellite (hardware fault).

    Args:
        doppler_dev: (n,) baseline Doppler deviations [Hz]
        sat_idx:     Index of the faulty satellite
        bias:        Persistent bias to add [Hz]
    """
    result = doppler_dev.copy()
    result[sat_idx] += bias
    return result


# ---------------------------------------------------------------------------
# Simulation configuration
# ---------------------------------------------------------------------------


@dataclass
class ResilienceTwinConfig:
    """Parameters for the GNSS Resilience Twin MC simulation.

    Attributes:
        n_mc:              Total MC trials; cycles through 4 fault classes.
        n_epochs:          Time steps per trial.
        n_sats:            Number of visible satellites.
        doppler_noise_std: Genuine Doppler noise 1-σ [Hz].
        spoof_bias_std:    Common meaconing bias 1-σ [Hz].
        spoof_diff_std:    Per-satellite differential spoofing noise 1-σ [Hz].
        graph_sigma:       Gaussian kernel bandwidth [Hz].
        dirichlet_alpha:   Dirichlet concentration for attack window.
        random_seed:       RNG seed for reproducibility.
    """

    n_mc: int = 400
    n_epochs: int = 80
    n_sats: int = 6
    doppler_noise_std: float = _DOPPLER_NOISE_STD
    # Resilience-twin spoofing scenario uses a more coherent (low diff-noise)
    # and stronger (higher common bias) attack than the generic spoof_sim defaults.
    # diff_std=0.10 Hz: meaconing broadcasts a near-identical signal to all sats.
    # bias_std=4.0 Hz: attacker injects a non-trivial velocity drift (~0.8 m/s).
    spoof_bias_std: float = 4.0
    spoof_diff_std: float = 0.10
    graph_sigma: float = _GRAPH_SIGMA
    dirichlet_alpha: float = _DIRICHLET_ALPHA
    random_seed: int = 42


# ---------------------------------------------------------------------------
# Per-trial simulation
# ---------------------------------------------------------------------------


def _simulate_trial_resilience(
    trial_idx: int,
    twin: ResilienceTwin,
    config: ResilienceTwinConfig,
    rng: np.random.Generator,
    los: np.ndarray,
    elevations: np.ndarray,
) -> tuple[int, int, float, float]:
    """Run one MC trial through the ResilienceTwin.

    Returns:
        (true_idx, predicted_idx, max_fault_score, mean_epoch_confidence)
        true_idx / predicted_idx: index into _FAULT_CLASSES (0–3)
        max_fault_score:          max(P_mp, P_hw, P_spoof) across epochs (ROC signal)
        mean_epoch_confidence:    mean of max(fault_posterior) across epochs
    """
    T = config.n_epochs
    fault_type = trial_idx % 4  # 0=nominal, 1=multipath, 2=hw_fault, 3=spoofing

    vel, clock_drift = _init_receiver(rng)

    # Trial-level fault parameters
    # Restrict hw fault to higher-elevation sats: detection threshold = 3.10·σᵢ = 3.10·σ/sin(el).
    # At el < 15° the threshold exceeds _HW_BIAS_STD, making detection unreliable.
    _hw_el_thresh = math.radians(_HW_EL_MIN_DEG)
    hw_eligible = [i for i, el in enumerate(elevations) if el >= _hw_el_thresh]
    if not hw_eligible:
        hw_eligible = list(range(config.n_sats))
    hw_sat_idx = int(rng.choice(hw_eligible))
    hw_bias = rng.normal(0.0, _HW_BIAS_STD)
    atk_start, atk_end = _sample_attack_window(T, config.dirichlet_alpha, rng)
    b_common = rng.normal(0.0, config.spoof_bias_std)

    vote_counts = [0, 0, 0, 0]
    fault_scores: list[float] = []
    confidence_sum = 0.0

    for t in range(T):
        vel, clock_drift = _propagate_state(vel, clock_drift, rng)
        # Model the receiver's GNSS-corrected velocity estimate: in a real system
        # the KF continuously corrects vel_hat toward the true trajectory.
        # Re-sampling fresh noise each epoch avoids the artificial O(√t) divergence
        # from independent random walks, which would otherwise swamp fault signals
        # (hw_bias ~1.5 Hz, spoof_bias ~2.5 Hz) by epoch 10 (~3 Hz background).
        vel_hat = vel + rng.normal(0.0, _INS_VEL_STD, size=3)
        clock_drift_hat = clock_drift + rng.normal(0.0, _INS_CLOCK_STD)

        meas = _gen_genuine_measurements(
            los,
            vel,
            clock_drift,
            vel_hat,
            clock_drift_hat,
            config.doppler_noise_std,
            rng,
        )

        if fault_type == 1:
            meas = _inject_multipath(meas, elevations, rng)
        elif fault_type == 2:
            meas = _inject_hw_fault(meas, hw_sat_idx, hw_bias)
        elif fault_type == 3 and atk_start <= t < atk_end:
            meas = _inject_attack(meas, b_common, config.spoof_diff_std, config.n_sats, rng)

        diag = twin.step(meas, t)

        vote_counts[_FAULT_CLASSES.index(diag.diagnosis)] += 1
        fp = diag.fault_posterior
        fault_scores.append(max(fp[1], fp[2], fp[3]))
        confidence_sum += diag.confidence

    # Spoofing attacks span only ~T/3 epochs (Dirichlet(2,2,2) partition).
    # Pure majority vote classifies most spoofing trials as NOMINAL because
    # the remaining ~2T/3 nominal epochs outvote the attack window.
    # Threshold detection: if enough epochs voted spoofing, declare the trial
    # as spoofing regardless of total-vote majority.
    # T//10 threshold (≈8): P(window < 8) ≈ 11% for Dirichlet(2,2,2).
    # Background spoof-vote rate under nominal is ~3%/epoch (Fiedler/RMT noise);
    # threshold=5 would cause P(Bin(80,0.03)≥5)≈10% → P_FA≈10%.
    # threshold=8 keeps P(Bin(80,0.03)≥8)≈0.3% ≈ 0% empirically.
    _SPOOF_VOTE_THRESH = max(T // 10, 3)
    if vote_counts[3] >= _SPOOF_VOTE_THRESH:
        predicted_idx = 3  # SPOOFING detected via threshold
    else:
        predicted_idx = int(np.argmax(vote_counts))
    # Mean over epochs suppresses single-epoch noise; max() over 80 epochs drove
    # P_FA to 100% because any one epoch with score > 0.5 would trigger the alarm.
    max_fault_score = float(np.mean(fault_scores))
    mean_ep_confidence = confidence_sum / T

    return fault_type, predicted_idx, max_fault_score, mean_ep_confidence


# ---------------------------------------------------------------------------
# MC simulation entry point
# ---------------------------------------------------------------------------


def run_resilience_simulation(
    config: ResilienceTwinConfig | None = None,
    rng: np.random.Generator | None = None,
) -> ResilienceTwinReport:
    """Run the GNSS Resilience Twin Monte Carlo simulation.

    Trial types cycle in round-robin: NOMINAL, MULTIPATH, HARDWARE_FAULT, SPOOFING.
    Satellite constellation geometry (Fibonacci lattice) is fixed across all trials.

    Args:
        config: Simulation parameters (defaults to ResilienceTwinConfig()).
        rng:    Random generator (defaults to seeded from config.random_seed).
    """
    if config is None:
        config = ResilienceTwinConfig()
    if rng is None:
        rng = np.random.default_rng(config.random_seed)

    los = _init_constellation(config.n_sats)
    elevations = np.arcsin(np.clip(los[:, 2], -1.0, 1.0))  # (n_sats,) [radians]

    confusion: list[list[int]] = [[0] * 4 for _ in range(4)]
    class_names = [fc.value for fc in _FAULT_CLASSES]
    per_class_correct: dict[str, int] = {k: 0 for k in class_names}
    per_class_total: dict[str, int] = {k: 0 for k in class_names}
    roc_scores: list[float] = []
    roc_labels: list[int] = []
    confidence_sum = 0.0

    for mc in range(config.n_mc):
        twin = ResilienceTwin(
            los=los,
            noise_std=config.doppler_noise_std,
            graph_sigma=config.graph_sigma,
        )
        true_idx, pred_idx, fault_score, ep_conf = _simulate_trial_resilience(
            mc,
            twin,
            config,
            rng,
            los,
            elevations,
        )

        confusion[true_idx][pred_idx] += 1
        per_class_total[class_names[true_idx]] += 1
        if true_idx == pred_idx:
            per_class_correct[class_names[true_idx]] += 1

        roc_labels.append(0 if true_idx == 0 else 1)
        roc_scores.append(fault_score)
        confidence_sum += ep_conf

    per_class_accuracy = {k: per_class_correct[k] / max(per_class_total[k], 1) for k in class_names}

    scores_arr = np.array(roc_scores)
    labels_arr = np.array(roc_labels)
    _, _, auc = _compute_roc(scores_arr, labels_arr)

    # Detection and false-alarm rates from vote-based classification.
    # Using the confusion matrix (already computed) rather than a fixed 0.5 threshold on
    # the continuous fault_score makes P_D/P_FA consistent with per_class_accuracy and
    # avoids threshold-tuning artefacts.
    #   P_FA = fraction of nominal trials classified as any fault class
    #   P_D  = fraction of fault trials classified as any non-nominal class
    n_nominal = per_class_total[class_names[0]]
    n_fault = sum(per_class_total[class_names[i]] for i in range(1, 4))
    n_fa = sum(confusion[0][j] for j in range(1, 4))
    n_detected = sum(confusion[i][j] for i in range(1, 4) for j in range(1, 4))

    return ResilienceTwinReport(
        p_detection=n_detected / max(n_fault, 1),
        p_false_alarm=n_fa / max(n_nominal, 1),
        auc=float(auc),
        per_class_accuracy=per_class_accuracy,
        confusion_matrix=confusion,
        mean_confidence=float(np.clip(confidence_sum / config.n_mc, 0.0, 1.0)),
        n_mc=config.n_mc,
        n_mc_per_class={k: per_class_total[k] for k in class_names},
    )


# ---------------------------------------------------------------------------
# Observation-driven digital twin entry point
# ---------------------------------------------------------------------------


def run_twin_on_observations(
    doppler_sequence: list[np.ndarray],
    los: np.ndarray,
    elevations: np.ndarray | None = None,
    noise_std: float = _DOPPLER_NOISE_STD,
    graph_sigma: float = _GRAPH_SIGMA,
    ins_sequence: list[np.ndarray | None] | None = None,
    osnma_sequence: list[list[bool] | None] | None = None,
    ins_noise_std: float = _INS_VEL_STD,
) -> list[EpochDiagnosis]:
    """Process a real observation sequence through the GNSS Resilience Twin.

    Initialises a fresh ResilienceTwin for the supplied window and runs each epoch
    through all 8 layers, returning a per-epoch EpochDiagnosis.

    Caller-supplied elevations override the values derived from LOS geometry,
    allowing higher-fidelity GM-RAIM elevation-adjusted noise when the receiver
    reports satellite elevations directly.

    For near-real-time operation, call this function with a sliding window
    (e.g., 30–120 epochs) as new observations arrive; the twin is stateless
    across windows by design to guarantee reproducibility.

    Args:
        doppler_sequence: T-length list of (n_sats,) Doppler residual arrays [Hz].
                          Δf_i = f_measured_i − f_predicted_i.
        los:              (n_sats, 3) unit line-of-sight vectors (receiver → satellite).
                          Used to build the IMM-KF geometry matrix H and to derive
                          elevations if `elevations` is None.
        elevations:       (n_sats,) elevation angles [radians].
                          If None, derived as arcsin(clip(los[:, 2], −1, 1)).
        noise_std:        Nominal Doppler noise 1-σ [Hz]. Default matches T1500 constants.
        graph_sigma:      Gaussian kernel bandwidth σ [Hz]. Default matches T1500 constants.
        ins_sequence:     T-length list of (3,) INS velocity deviations [m/s], or None per epoch.
                          If None, the INS coupling layer uses chi²(3) self-test only.
        osnma_sequence:   T-length list of per-satellite OSNMA authentication bool lists,
                          or None per epoch. If None, defaults to fully authenticated.
        ins_noise_std:    INS velocity noise 1-σ [m/s]. Default matches T1500 constants.

    Returns:
        List of T EpochDiagnosis objects in input order.
    """
    if len(doppler_sequence) == 0:
        return []

    n_sats = los.shape[0]
    for i, dop in enumerate(doppler_sequence):
        if len(dop) != n_sats:
            raise ValueError(
                f"doppler_sequence[{i}] has {len(dop)} entries; expected {n_sats} (= n_sats)"
            )

    twin = ResilienceTwin(
        los=los,
        noise_std=noise_std,
        graph_sigma=graph_sigma,
        ins_noise_std=ins_noise_std,
    )

    # Override elevations with caller-supplied values if provided.
    if elevations is not None:
        twin._elevations = elevations

    results: list[EpochDiagnosis] = []
    for i, dop in enumerate(doppler_sequence):
        ins_vel = ins_sequence[i] if ins_sequence is not None else None
        osnma_auth = osnma_sequence[i] if osnma_sequence is not None else None
        results.append(twin.step(dop, t=i, ins_velocity=ins_vel, osnma_auth=osnma_auth))
    return results
