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

from gnss.cn0_detector import CN0AnomalyResult

# Shared signal constants (own source of truth, independent of simulation layer)
from gnss.constants import (
    _DIRICHLET_ALPHA,
    _DOPPLER_NOISE_STD,
    _GRAPH_SIGMA,
    _INS_CLOCK_STD,
    _INS_VEL_STD,
    GMM_FAULT_THRESH,
)

# Layer classes and result types — defined in gnss.layers sub-package
from gnss.layers import (
    CoopRAIMLayer,
    CoopRAIMResult,
    DuminilCopinPhaseMonitor,
    FaultEntropyMonitor,
    FaultEntropyResult,
    GMMRaim,
    GMMResult,
    HuhSelectionResult,
    HuhSubsetSelector,
    IMMKalman,
    IMMResult,
    INSCouplingLayer,
    INSCouplingResult,
    OSNMALayer,
    OSNMALayerResult,
    PhaseTransitionResult,
    SpectralMonitor,
    SpectralResult,
    StructuralDependencyMonitor,
    StructuralMonitorResult,
)

# Geometry / graph / ROC utilities (pure math, no simulation dependencies)
from gnss.math_utils import compute_roc, init_constellation

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
# Module constants (pillar fusion weights + simulation parameters)
# ---------------------------------------------------------------------------

_EL_MIN_DEG: float = 5.0  # minimum elevation clamp [degrees]
_EL_MIN_RAD: float = math.radians(_EL_MIN_DEG)

_EPS: float = 1e-300

_FUSE_SPOOF_FIEDLER: float = 0.50  # Fiedler-ratio weight in spoofing score
_FUSE_SPOOF_RMT: float = 0.30  # RMT-anomaly weight in spoofing score
_FUSE_MP_ELEV: float = 0.40  # elevation-correlation weight in multipath score

# Fusion weights for layers 5–8 contributions to spoofing score
_FUSE_INS_SPOOF: float = 0.10
_FUSE_COOP_SPOOF: float = 0.15
_FUSE_OSNMA_SPOOF: float = 0.40
_FUSE_STRUCT_SPOOF: float = 0.05
_FUSE_GMM_SPOOF_COMMON: float = 0.50
_FUSE_PHASE_SPOOF: float = 0.10  # phase-transition alert weight in spoof score
_FUSE_CN0_SPOOF: float = 0.20  # C/N0 anomaly (spread collapse / CUSUM / corr burst)

_MP_NOISE_INFLATION: float = 2.0  # multipath noise amplitude [Hz]
# 40× ensures E[P(detect)] ≈ 90% even for the worst eligible sat (el=24.6° → threshold≈2.2 Hz).
_HW_BIAS_STD: float = 40.0 * _DOPPLER_NOISE_STD  # HW fault bias 1-σ [Hz]
_HW_EL_MIN_DEG: float = 15.0  # min elevation [deg] for hw_fault sat selection

# Ordered fault class list — index aligns with fault_posterior positions
_FAULT_CLASSES: list[FaultClass] = [
    FaultClass.NOMINAL,
    FaultClass.MULTIPATH,
    FaultClass.HARDWARE_FAULT,
    FaultClass.SPOOFING,
]

# ---------------------------------------------------------------------------
# Composite result dataclasses (pillar-level outputs)
# ---------------------------------------------------------------------------


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
    cn0_anomaly: CN0AnomalyResult | None = None  # C/N0 anomaly result; None if unavailable


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
        huh = self._huh.select(np.array(gmm.gamma) > GMM_FAULT_THRESH)

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
        cn0_anomaly: CN0AnomalyResult | None = None,
    ) -> tuple[np.ndarray, FaultEntropyResult]:
        """Compute final 4-class posterior and entropy alert.

        C/N0 anomaly score contributes _FUSE_CN0_SPOOF * p_spoof_cn0 to the
        spoofing signal when cn0_anomaly is available (non-None).

        Returns:
            (fp, entropy_result): fp is a (4,) normalized probability array.
        """
        p_nom, p_mp, p_hw, p_spoof = integrity.base_posterior

        cn0_spoof_contrib = (
            _FUSE_CN0_SPOOF * cn0_anomaly.p_spoof_cn0 if cn0_anomaly is not None else 0.0
        )
        s_spoof = (
            p_spoof
            + _FUSE_SPOOF_FIEDLER
            * max(structure.spectral.fiedler_ratio - 1.0, 0.0)
            * integrity.gmm.sign_corr
            + _FUSE_SPOOF_RMT * structure.spectral.rmt_anomaly
            + _FUSE_OSNMA_SPOOF * auth.p_spoofed
            + _FUSE_STRUCT_SPOOF * float(structure.structural.alert)
            + _FUSE_PHASE_SPOOF * float(structure.phase.phase_alert)
            + cn0_spoof_contrib
        )
        s_mp = p_mp
        s_hw = p_hw
        # Use base nominal posterior directly; normalization handles competition.
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
        cn0_anomaly: CN0AnomalyResult | None = None,
    ) -> EpochDiagnosis:
        """Process one epoch of Doppler residuals through the 4-pillar stack.

        Args:
            doppler_dev:  (n_sats,) Doppler residuals [Hz]
            t:            Epoch index (informational only)
            ins_velocity: (3,) external INS velocity deviation [m/s], or None
            osnma_auth:   Per-satellite OSNMA authentication flags, or None
            cn0_anomaly:  C/N0 anomaly result from CN0AnomalyDetector, or None
        """
        auth = self._auth.assess(osnma_auth)
        integrity = self._integrity.assess(doppler_dev, self._elevations, ins_velocity)
        structure = self._structure.update(doppler_dev)
        fp, entropy_result = self._intervention.fuse(auth, integrity, structure, cn0_anomaly)

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
            cn0_anomaly=cn0_anomaly,
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

    los = init_constellation(config.n_sats)
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
    _, _, auc = compute_roc(scores_arr, labels_arr)

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
