"""GNSS Resilience Twin — 4-pillar classes and orchestrator (T1500).

Implements the per-pillar logic and the ResilienceTwin orchestrator:

    AuthenticationPillar  — Pillar 1: OSNMA authentication coverage
    IntegrityPillar       — Pillar 2: GM-RAIM + IMM-KF + INS + CoopRAIM + Huh
    StructuralPillar      — Pillar 3: Spectral graph + dependency + phase
    InterventionPillar    — Pillar 4: Entropy fusion + 4-class posterior decision
    ResilienceTwin        — Orchestrator (Authentication → Integrity → Structure → Intervention)

Module-level constants (fusion weights and simulation defaults) are also defined
here so that both the pillar classes and the MC simulation entry point
(gnss.resilience_twin) can import from a single canonical location.
"""

from __future__ import annotations

import math

import numpy as np

from gnss.cn0_detector import CN0AnomalyResult
from gnss.constants import (
    _DIRICHLET_ALPHA,  # noqa: F401 — re-exported for resilience_twin.py consumers
    _DOPPLER_NOISE_STD,
    _GRAPH_SIGMA,
    _INS_CLOCK_STD,  # noqa: F401
    _INS_VEL_STD,
    GMM_FAULT_THRESH,
)
from gnss.layers import (
    CoopRAIMLayer,
    DuminilCopinPhaseMonitor,
    FaultEntropyMonitor,
    FaultEntropyResult,
    GMMRaim,
    HuhSubsetSelector,
    IMMKalman,
    INSCouplingLayer,
    OSNMALayer,
    SpectralMonitor,
    StructuralDependencyMonitor,
)
from gnss.twin_schemas import (
    AuthenticationScore,
    EpochDiagnosis,
    IntegrityScore,
    StructuralScore,
)
from schemas import FaultClass

# ---------------------------------------------------------------------------
# Module constants (pillar fusion weights + simulation defaults)
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
