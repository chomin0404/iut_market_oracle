"""Stanford GPS Lab benchmark adapter for GNSS spoofing detection validation.

Source repository: https://github.com/stanford-gps-lab/spoofing-detection
Paper: Rothmaier et al., "GNSS Spoofing Detection through Metric Combinations",
       ION GNSS+ 2021. https://web.stanford.edu/group/scpnt/gpslab/pubs/papers/
       Rothmaier_IONGNSS_2021_MultiMetricDetection.pdf

Notes
-----
The Stanford repository is MATLAB-only with no data files.
This module re-implements the DoA_prr_combinationStudy parameter space in Python
and provides a chi-squared RAIM baseline for cross-validation with our ResilienceTwin.

Two scenarios are provided:

1. Pseudorange chi-squared RAIM (Stanford baseline)
   Rothmaier et al. parameter space:
     N = 12 satellites, σ_pr = 3 m, P_FA_max = 1e-7,
     attack = 10 m vertical position bias (xoffset = [0, 0, 10, 0]).

   Key result: a coherent meaconing attack shifts all pseudoranges by
   b_i = dot(los_i, [0,0,10]) m, which lies in the column space of the
   geometry matrix H.  After projection onto the residual space,
   (I − H(HᵀH)⁻¹Hᵀ)·b = 0, so the chi-squared test statistic under H1
   equals the test statistic under H0 → P_D ≈ P_FA.
   This is the fundamental limitation that motivates multi-metric detection.

2. ResilienceTwin with Stanford N = 12 satellite geometry
   Uses our Doppler-domain attack model (coherent meaconing bias) and
   the full 4-pillar detection stack.
   Expected: P_D >> P_FA because spectral / IMM / CN0 pillars detect
   the coherence structure change that chi-squared RAIM cannot see.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2

from gnss.math_utils import init_constellation

# ---------------------------------------------------------------------------
# Stanford published parameters (DoA_prr_combinationStudy.m)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StanfordParams:
    """Published parameters from Rothmaier et al. ION GNSS+ 2021.

    Attributes:
        n_sats:            Number of visible satellites (N = 12 in the paper).
        sig_pr:            Pseudorange noise 1-σ [m] (sigPr = 3 m).
        p_fa_max:          Maximum false alert probability (P_FAmax = 1e-7).
        bias_vertical_m:   Spoofing attack: vertical position bias [m] (xoffset_z = 10 m).
        n_mc:              Monte Carlo trials.
        random_seed:       RNG seed for reproducibility.
    """

    n_sats: int = 12
    sig_pr: float = 3.0  # pseudorange noise 1-σ [m]
    p_fa_max: float = 1e-7  # false alert probability
    bias_vertical_m: float = 10.0  # spoofing vertical bias [m]
    n_mc: int = 10_000
    random_seed: int = 42


# ---------------------------------------------------------------------------
# Pseudorange observation generator
# ---------------------------------------------------------------------------


def generate_pr_obs(
    los: np.ndarray,
    rng: np.random.Generator,
    *,
    spoofed: bool,
    sig_pr: float = 3.0,
    bias_vertical_m: float = 10.0,
) -> np.ndarray:
    """Generate pseudorange observations (genuine or coherently spoofed).

    Observation model:
        H0 (genuine): ρ_i = ε_i,          ε_i ~ N(0, σ_pr²)
        H1 (spoofed): ρ_i = b_i + ε_i,    b_i = los_i[2] · bias_vertical_m

    The attack vector b = H · xoffset where xoffset = [0, 0, bias_vertical_m, 0]ᵀ.
    This is a coherent meaconing attack: every satellite is shifted by the LOS
    projection of the spoofed vertical displacement.

    Args:
        los:              (n_sats, 3) LOS unit vectors.
        rng:              Random generator.
        spoofed:          True to inject the coherent position bias.
        sig_pr:           Pseudorange noise 1-σ [m].
        bias_vertical_m:  Vertical position bias magnitude [m].

    Returns:
        (n_sats,) pseudorange residuals [m].
    """
    noise = rng.normal(0.0, sig_pr, size=len(los))
    if not spoofed:
        return noise
    # b_i = dot(los_i, [0, 0, bias_z])
    bias = los[:, 2] * bias_vertical_m
    return noise + bias


# ---------------------------------------------------------------------------
# Chi-squared RAIM baseline
# ---------------------------------------------------------------------------


def chi2_raim_score(
    pr_obs: np.ndarray,
    los: np.ndarray,
    sig_pr: float = 3.0,
) -> float:
    """Chi-squared RAIM test statistic on pseudorange residuals.

    Model: ρ = H·x + w,  w ~ N(0, σ²·I),  H = [los | 1] ∈ ℝ^{n×4}

    After removing the WLS position/clock estimate:
        ρ̂ = (I − H(HᵀH)⁻¹Hᵀ) ρ   (projection onto null space of Hᵀ)
        T  = ||ρ̂||² / σ²            ~ χ²(n − 4)  under H0

    Coherent attack note:
        The attack bias b = H·xoffset lies in col(H), so (I − P)·b = 0.
        Under H1, E[T] = E[T_H0] → P_D ≈ P_FA for any coherent meaconing.

    Args:
        pr_obs: (n_sats,) pseudorange residuals [m].
        los:    (n_sats, 3) LOS unit vectors.
        sig_pr: Pseudorange noise 1-σ [m].

    Returns:
        Chi-squared test statistic (scalar, non-negative).
    """
    n = len(pr_obs)
    # Geometry matrix H = [los | ones]
    H = np.hstack([los, np.ones((n, 1))])
    # Projection onto residual space: (I − H(HᵀH)⁻¹Hᵀ)
    HTH_inv = np.linalg.pinv(H.T @ H)
    P_res = np.eye(n) - H @ HTH_inv @ H.T
    residuals = P_res @ pr_obs
    return float(residuals @ residuals) / (sig_pr**2)


def chi2_raim_threshold(n_sats: int, p_fa: float) -> float:
    """Chi-squared RAIM detection threshold for a given false alert probability.

    T ~ χ²(n_sats − 4) under H0.
    Returns γ such that P(T > γ | H0) = p_fa.

    Args:
        n_sats: Number of satellites.
        p_fa:   False alert probability.

    Returns:
        Detection threshold γ.
    """
    dof = n_sats - 4
    if dof <= 0:
        raise ValueError(f"Need at least 5 satellites; got {n_sats}")
    return float(chi2.ppf(1.0 - p_fa, dof))


# ---------------------------------------------------------------------------
# Monte Carlo result
# ---------------------------------------------------------------------------


@dataclass
class StanfordMCResult:
    """Monte Carlo evaluation result for the Stanford benchmark."""

    p_fa: float  # Empirical false alert probability
    p_d: float  # Empirical spoofing detection probability
    n_mc: int  # Number of genuine trials (= number of spoofed trials)
    threshold: float  # Detection threshold used
    method: str  # "chi2_raim" or "resilience_twin"


# ---------------------------------------------------------------------------
# Chi-squared RAIM Monte Carlo (Stanford baseline)
# ---------------------------------------------------------------------------


def run_chi2_raim_mc(
    params: StanfordParams | None = None,
) -> StanfordMCResult:
    """Monte Carlo P_D / P_FA evaluation of the chi-squared RAIM baseline.

    Demonstrates the fundamental limitation of single-metric RAIM against
    coherent meaconing: P_D ≈ P_FA because the attack bias is invisible
    after residual projection.

    Args:
        params: Stanford parameter spec; uses defaults if None.

    Returns:
        StanfordMCResult with empirical P_FA, P_D, and the detection threshold.
    """
    if params is None:
        params = StanfordParams()

    rng = np.random.default_rng(params.random_seed)
    los = init_constellation(params.n_sats)
    threshold = chi2_raim_threshold(params.n_sats, params.p_fa_max)

    false_alarms = 0
    detections = 0

    for _ in range(params.n_mc):
        # Genuine trial
        pr_gen = generate_pr_obs(los, rng, spoofed=False, sig_pr=params.sig_pr)
        if chi2_raim_score(pr_gen, los, params.sig_pr) > threshold:
            false_alarms += 1

        # Spoofed trial (coherent meaconing: vertical position bias)
        pr_spoof = generate_pr_obs(
            los,
            rng,
            spoofed=True,
            sig_pr=params.sig_pr,
            bias_vertical_m=params.bias_vertical_m,
        )
        if chi2_raim_score(pr_spoof, los, params.sig_pr) > threshold:
            detections += 1

    return StanfordMCResult(
        p_fa=false_alarms / params.n_mc,
        p_d=detections / params.n_mc,
        n_mc=params.n_mc,
        threshold=threshold,
        method="chi2_raim",
    )


# ---------------------------------------------------------------------------
# ResilienceTwin Monte Carlo in Stanford N=12 geometry
# ---------------------------------------------------------------------------


def run_resilience_twin_stanford_mc(
    n_mc: int = 400,
    random_seed: int = 42,
) -> StanfordMCResult:
    """Monte Carlo P_D / P_FA of ResilienceTwin in Stanford N=12 geometry.

    Uses our Doppler-domain coherent meaconing attack and the full 4-pillar
    detection stack, tested with the 12-satellite constellation geometry
    from the Stanford benchmark.

    P_FA: fraction of NOMINAL trials classified as any non-nominal class.
    P_D:  fraction of SPOOFING trials correctly classified as SPOOFING.

    _FAULT_CLASSES index order: [NOMINAL=0, MULTIPATH=1, HARDWARE_FAULT=2, SPOOFING=3]
    confusion_matrix[true_idx][pred_idx]

    Args:
        n_mc:        Total MC trials (round-robin across 4 fault classes).
        random_seed: RNG seed.

    Returns:
        StanfordMCResult with empirical P_FA and P_D.
    """
    from gnss.resilience_twin import ResilienceTwinConfig, run_resilience_simulation

    # Stanford N=12 geometry, matched spoofing parameters
    config = ResilienceTwinConfig(
        n_mc=n_mc,
        n_epochs=80,
        n_sats=12,  # Stanford geometry
        spoof_bias_std=4.0,
        spoof_diff_std=0.10,
        random_seed=random_seed,
    )
    report = run_resilience_simulation(config)

    # P_FA: already computed in report (fraction of nominal trials alarmed)
    p_fa = report.p_false_alarm

    # P_D for SPOOFING specifically: confusion_matrix[3][3] / n_mc_per_class["spoofing"]
    spoof_total = report.n_mc_per_class.get("spoofing", 0)
    spoof_correct = report.confusion_matrix[3][3] if len(report.confusion_matrix) == 4 else 0
    p_d = spoof_correct / spoof_total if spoof_total > 0 else 0.0

    return StanfordMCResult(
        p_fa=p_fa,
        p_d=p_d,
        n_mc=n_mc,
        threshold=float("nan"),  # threshold is internal to ResilienceTwin
        method="resilience_twin",
    )
