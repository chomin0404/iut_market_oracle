"""GNSS Resilience Twin (T1500) — MC simulation hub and backward-compatible re-export layer.

4-pillar fault discrimination platform:

  ┌─────────────────────────────────────────────────────────────────────────────┐
  │  Pillar 1 — Authentication  OSNMA Galileo authentication coverage           │
  │  Pillar 2 — Integrity       GM-RAIM + IMM-KF + INS coupling + CoopRAIM     │
  │  Pillar 3 — Structure       Laplacian spectral graph + dependency monitor   │
  │  Pillar 4 — Intervention    Entropy fusion + 4-class posterior decision     │
  └─────────────────────────────────────────────────────────────────────────────┘

Pillar classes and schema dataclasses live in dedicated sub-modules:
    gnss.twin_pillars  — constants + AuthenticationPillar, IntegrityPillar,
                          StructuralPillar, InterventionPillar, ResilienceTwin
    gnss.twin_schemas  — AuthenticationScore, IntegrityScore, StructuralScore,
                          EpochDiagnosis

This module retains the MC simulation entry point (run_resilience_simulation)
and the observation-driven entry point (run_twin_on_observations), and
re-exports all names so existing ``from gnss.resilience_twin import ...`` call
sites continue to work without modification.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Re-exports — constants, layer classes, pillar classes, schema dataclasses
# All noqa: F401 tags preserve backward-compatible import paths.
# ---------------------------------------------------------------------------
from gnss.constants import (  # noqa: F401
    _DIRICHLET_ALPHA,
    _DOPPLER_NOISE_STD,
    _GRAPH_SIGMA,
    _INS_CLOCK_STD,
    _INS_VEL_STD,
    GMM_FAULT_THRESH,
)
from gnss.layers import (  # noqa: F401
    DuminilCopinPhaseMonitor,
    FaultEntropyMonitor,
    GMMRaim,
    HuhSubsetSelector,
    IMMKalman,
    SpectralMonitor,
)
from gnss.math_utils import compute_roc, init_constellation
from gnss.spoof_sim import (
    _gen_genuine_measurements,
    _init_receiver,
    _inject_attack,
    _propagate_state,
    _sample_attack_window,
)
from gnss.twin_pillars import (  # noqa: F401
    _EL_MIN_DEG,
    _EL_MIN_RAD,
    _EPS,
    _FAULT_CLASSES,
    _FUSE_CN0_SPOOF,
    _FUSE_COOP_SPOOF,
    _FUSE_GMM_SPOOF_COMMON,
    _FUSE_INS_SPOOF,
    _FUSE_MP_ELEV,
    _FUSE_OSNMA_SPOOF,
    _FUSE_PHASE_SPOOF,
    _FUSE_SPOOF_FIEDLER,
    _FUSE_SPOOF_RMT,
    _FUSE_STRUCT_SPOOF,
    _HW_BIAS_STD,
    _HW_EL_MIN_DEG,
    _MP_NOISE_INFLATION,
    AuthenticationPillar,
    IntegrityPillar,
    InterventionPillar,
    ResilienceTwin,
    StructuralPillar,
)
from gnss.twin_schemas import (  # noqa: F401
    AuthenticationScore,
    EpochDiagnosis,
    IntegrityScore,
    StructuralScore,
)
from schemas import FaultClass, ResilienceTwinReport  # noqa: F401

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
        true_idx / predicted_idx: index into _FAULT_CLASSES (0-3)
        max_fault_score:          mean(P_fault) across epochs (ROC signal)
        mean_epoch_confidence:    mean of max(fault_posterior) across epochs
    """
    T = config.n_epochs
    fault_type = trial_idx % 4  # 0=nominal, 1=multipath, 2=hw_fault, 3=spoofing

    vel, clock_drift = _init_receiver(rng)

    # Restrict hw fault to higher-elevation sats: detection threshold = 3.10*sigma_i.
    # At el < 15 deg the threshold exceeds _HW_BIAS_STD, making detection unreliable.
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
        # Model the receiver GNSS-corrected velocity estimate: re-sampling fresh
        # noise each epoch avoids O(sqrt(t)) random-walk divergence that would
        # swamp fault signals by epoch 10.
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
    # T//10 threshold (approx 8): P(window < 8) approx 11% for Dirichlet(2,2,2).
    # Background spoof-vote rate under nominal is ~3%/epoch;
    # threshold=8 keeps P(Bin(80,0.03)>=8) approx 0.3% empirically.
    _SPOOF_VOTE_THRESH = max(T // 10, 3)
    if vote_counts[3] >= _SPOOF_VOTE_THRESH:
        predicted_idx = 3  # SPOOFING detected via threshold
    else:
        predicted_idx = int(np.argmax(vote_counts))
    # Mean over epochs suppresses single-epoch noise.
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

    per_class_accuracy = {
        k: per_class_correct[k] / max(per_class_total[k], 1) for k in class_names
    }

    scores_arr = np.array(roc_scores)
    labels_arr = np.array(roc_labels)
    _, _, auc = compute_roc(scores_arr, labels_arr)

    # Detection and false-alarm rates from vote-based classification.
    # Using the confusion matrix makes P_D/P_FA consistent with per_class_accuracy
    # and avoids threshold-tuning artefacts.
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

    Args:
        doppler_sequence: T-length list of (n_sats,) Doppler residual arrays [Hz].
        los:              (n_sats, 3) unit line-of-sight vectors (receiver to satellite).
        elevations:       (n_sats,) elevation angles [radians]; derived from los if None.
        noise_std:        Nominal Doppler noise 1-sigma [Hz].
        graph_sigma:      Gaussian kernel bandwidth sigma [Hz].
        ins_sequence:     T-length list of (3,) INS velocity deviations [m/s] or None.
        osnma_sequence:   T-length list of per-satellite OSNMA bool lists or None.
        ins_noise_std:    INS velocity noise 1-sigma [m/s].

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
