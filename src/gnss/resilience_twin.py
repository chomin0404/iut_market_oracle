"""GNSS Resilience Twin (T1500).

4-layer fault discrimination platform for drones and positioning equipment:
  Layer 1 — GM-RAIM:  Gaussian-mixture per-satellite fault posteriors
  Layer 2 — IMM-KF:   Interacting Multiple Model Kalman filter (3 regimes)
  Layer 3 — Spectral: Laplacian Fiedler + RMT anomaly on satellite similarity graph
  Layer 4 — Entropy:  Shannon entropy + KL divergence on 4-class fault posterior

Output classes (FaultClass enum, index 0-3):
  NOMINAL | MULTIPATH | HARDWARE_FAULT | SPOOFING

MC simulation entry point: run_resilience_simulation()
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Private helpers and constants imported from T1300 spoofing sim
from gnss.spoof_sim import (
    _DIRICHLET_ALPHA,
    _DOPPLER_NOISE_STD,
    _GRAPH_SIGMA,
    _INS_CLOCK_STD,
    _INS_VEL_STD,
    _SPOOF_BIAS_STD,
    _SPOOF_DIFF_STD,
    _build_graph,
    _compute_roc,
    _gen_genuine_measurements,
    _geometry_matrix,
    _init_constellation,
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
_HW_BIAS_STD: float = 5.0 * _DOPPLER_NOISE_STD  # HW fault bias 1-σ [Hz]

_EPS: float = 1e-300  # probability floor

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
class EpochDiagnosis:
    """Complete per-epoch diagnostic output from ResilienceTwin."""

    t: int
    fault_posterior: tuple[float, float, float, float]  # [P_nom, P_mp, P_hw, P_spoof]
    diagnosis: FaultClass
    confidence: float  # max(fault_posterior)
    gmm: GMMResult
    imm: IMMResult
    spectral: SpectralResult
    entropy: FaultEntropyResult


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
        self._mu = np.ones(3) / 3.0  # uniform initial mode probs
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
# ResilienceTwin orchestrator
# ---------------------------------------------------------------------------


class ResilienceTwin:
    """4-layer GNSS fault discrimination platform.

    Fuses GM-RAIM, IMM-KF, spectral, and entropy layers via heuristic
    Bayesian scoring into a 4-class fault posterior at each epoch:

        s_spoof ∝ μ_spoof + α_F · max(ρ_F−1, 0) · C_s + α_R · rmt
        s_mp    ∝ μ_mp   + α_E · ρ_el
        s_hw    = 1 if n_gmm_faults == 1 else 0   (isolated single outlier)
        s_nom   ∝ μ_nom  · max(0, 1 − max(s_spoof, s_mp, s_hw))

    Posterior: softmax-normalise [s_nom, s_mp, s_hw, s_spoof].
    """

    def __init__(
        self,
        los: np.ndarray,
        noise_std: float = _DOPPLER_NOISE_STD,
        graph_sigma: float = _GRAPH_SIGMA,
    ) -> None:
        n_sats = len(los)
        self._elevations = np.arcsin(np.clip(los[:, 2], -1.0, 1.0))  # [radians]
        self._gmm = GMMRaim(noise_std=noise_std)
        self._imm = IMMKalman(los=los, noise_std=noise_std)
        self._spectral = SpectralMonitor(
            n_sats=n_sats, noise_std=noise_std, graph_sigma=graph_sigma
        )
        self._entropy = FaultEntropyMonitor()

    def step(self, doppler_dev: np.ndarray, t: int = 0) -> EpochDiagnosis:
        """Process one epoch of Doppler residuals.

        Args:
            doppler_dev: (n_sats,) Doppler residuals [Hz]
            t:           Epoch index (informational only)
        """
        gmm = self._gmm.classify(doppler_dev, self._elevations)
        imm = self._imm.update(doppler_dev)
        spectral = self._spectral.analyze(doppler_dev)

        fp = self._fuse(gmm, imm, spectral)
        entropy_result = self._entropy.update(fp)

        idx = int(np.argmax(fp))
        return EpochDiagnosis(
            t=t,
            fault_posterior=(float(fp[0]), float(fp[1]), float(fp[2]), float(fp[3])),
            diagnosis=_FAULT_CLASSES[idx],
            confidence=float(fp[idx]),
            gmm=gmm,
            imm=imm,
            spectral=spectral,
            entropy=entropy_result,
        )

    def _fuse(self, gmm: GMMResult, imm: IMMResult, spectral: SpectralResult) -> np.ndarray:
        """Compute 4-class fault posterior from layer outputs."""
        mu_nom, mu_mp, mu_spoof = imm.mode_weights

        s_spoof = (
            mu_spoof
            + _FUSE_SPOOF_FIEDLER * max(spectral.fiedler_ratio - 1.0, 0.0) * gmm.sign_corr
            + _FUSE_SPOOF_RMT * spectral.rmt_anomaly
        )
        s_mp = mu_mp + _FUSE_MP_ELEV * gmm.elev_corr
        s_hw = 1.0 if gmm.n_fault == 1 else 0.0
        s_nom = mu_nom * max(0.0, 1.0 - max(s_spoof, s_mp, s_hw))

        raw = np.clip(np.array([s_nom, s_mp, s_hw, s_spoof], dtype=float), 0.0, None)
        total = raw.sum()
        if total < _EPS:
            return np.array([1.0, 0.0, 0.0, 0.0])
        return raw / total


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
    spoof_bias_std: float = _SPOOF_BIAS_STD
    spoof_diff_std: float = _SPOOF_DIFF_STD
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
    vel_hat = vel + rng.normal(0.0, _INS_VEL_STD, size=3)
    clock_drift_hat = clock_drift + rng.normal(0.0, _INS_CLOCK_STD)

    # Trial-level fault parameters
    hw_sat_idx = int(rng.integers(config.n_sats))
    hw_bias = rng.normal(0.0, _HW_BIAS_STD)
    atk_start, atk_end = _sample_attack_window(T, config.dirichlet_alpha, rng)
    b_common = rng.normal(0.0, config.spoof_bias_std)

    vote_counts = [0, 0, 0, 0]
    fault_scores: list[float] = []
    confidence_sum = 0.0

    for t in range(T):
        vel, clock_drift = _propagate_state(vel, clock_drift, rng)
        vel_hat, clock_drift_hat = _propagate_state(vel_hat, clock_drift_hat, rng)

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

    predicted_idx = int(np.argmax(vote_counts))
    max_fault_score = max(fault_scores)
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

    # Detection and false-alarm rates at threshold = 0.5
    n_fault = sum(1 for lbl in roc_labels if lbl == 1)
    n_nominal = sum(1 for lbl in roc_labels if lbl == 0)
    n_detected = sum(1 for s, lbl in zip(roc_scores, roc_labels) if lbl == 1 and s > 0.5)
    n_fa = sum(1 for s, lbl in zip(roc_scores, roc_labels) if lbl == 0 and s > 0.5)

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
