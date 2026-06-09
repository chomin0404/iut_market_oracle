"""Large-scale Monte Carlo validation for GNSS spoofing detection (T1300/T1500).

Evaluates three attack scenarios against the operating-point target:
    P_fa < 1e-4  AND  P_miss < 0.01

Fisher-combined statistic (fully vectorisable, N = 10⁶ via NumPy batching)
───────────────────────────────────────────────────────────────────────────
    chi_t  = Σ(δfᵢ − mean)² / σ_D²            ~ χ²(n−1) under H₀  [differential]
    coh_t  = n · mean(δf)²  / σ_D²             ~ χ²(1)   under H₀  [common-mode]
    F_t    = −2(ln p_chi + ln p_coh)            ~ χ²(4)   under H₀  [Fisher fusion]

    Trial score = max_{t ∈ [0,T)} F_t

chi_t detects differential spoofing (SIMPLISTIC); coh_t detects coherent
common-mode bias that chi-squared RAIM cannot see (MEACONING, SOPHISTICATED).

NP threshold for trial-level P_fa = α_trial:
    P(max_t F_t > τ) = 1 − P(F ≤ τ)^T = α_trial
    ⟹  τ = χ²(4).ppf(1 − α_trial^{1/T})

Scenarios
─────────
  SIMPLISTIC    b_common ~ N(0, 64),  δᵢ ~ N(0, 4)   Hz  — easy baseline
  MEACONING     b_common ~ N(0, 16),  δᵢ ~ N(0, 0.01) Hz  — coherent, RAIM-blind
  SOPHISTICATED b(t) = rate·t, rate ~ U(0.05, 0.20) Hz/epoch, δᵢ ~ N(0, 0.0025) Hz

DET curve
─────────
  x-axis : P_fa   (false alarm rate)
  y-axis : P_miss (missed detection = 1 − P_D)
  AUC_DET: 0 = perfect, 0.5 = random
  AUC_ROC: 1 = perfect, 0.5 = random
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.special import gammaincc as _gammaincc
from scipy.stats import chi2 as _chi2_dist

from gnss.constants import _DOPPLER_NOISE_STD

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

_FISHER_DF: int = 4  # chi²(4): chi(df=n−1) + coh(df=1) → 2 components × 2 df
_EPS: float = 1e-300  # p-value floor to prevent log(0)


# ---------------------------------------------------------------------------
# Scenario enum
# ---------------------------------------------------------------------------


class SpoofingScenario(str, Enum):
    """Three spoofing scenarios with distinct detection difficulty.

    SIMPLISTIC:
        Sudden large bias with substantial differential noise.
        Both chi_t and coh_t elevated → easy detection.

    MEACONING:
        Coherent re-broadcast: all satellites receive nearly identical bias.
        chi_t insensitive (removes mean); coh_t detects common-mode → essential.

    SOPHISTICATED:
        Gradual take-over: bias ramps linearly as b(t) = rate · t.
        Chi/coh near zero at t=0; max over window detects growing ramp.
    """

    SIMPLISTIC = "simplistic"
    MEACONING = "meaconing"
    SOPHISTICATED = "sophisticated"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class MCValidationConfig:
    """Parameters for the large-scale Monte Carlo validation.

    Attributes:
        n_trials:             Trials per scenario (nominal + attack).
        n_epochs:             Time steps per trial.
        n_sats:               Number of visible satellites.
        doppler_noise_std:    Genuine Doppler noise 1-σ [Hz].
        simplistic_bias_std:  Common bias 1-σ for SIMPLISTIC [Hz].
        simplistic_diff_std:  Per-satellite differential noise 1-σ for SIMPLISTIC [Hz].
        meaconing_bias_std:   Common bias 1-σ for MEACONING [Hz].
        meaconing_diff_std:   Per-satellite differential noise 1-σ for MEACONING [Hz].
        sophisticated_ramp_lo: Min ramp rate for SOPHISTICATED [Hz/epoch].
        sophisticated_ramp_hi: Max ramp rate for SOPHISTICATED [Hz/epoch].
        sophisticated_diff_std: Per-satellite differential noise 1-σ for SOPHISTICATED [Hz].
        p_fa_target:          Trial-level false alarm target.
        p_miss_target:        Missed detection target (= 1 − P_D target).
        n_thresholds:         Number of threshold sweep points for DET/ROC curves.
        random_seed:          RNG seed.
        batch_size:           Trials per batch for memory-bounded processing.
    """

    n_trials: int = 1_000_000
    n_epochs: int = 80
    n_sats: int = 6
    doppler_noise_std: float = _DOPPLER_NOISE_STD  # 0.30 Hz
    # SIMPLISTIC scenario
    simplistic_bias_std: float = 8.0  # Hz — large sudden step
    simplistic_diff_std: float = 2.0  # Hz — large differential noise
    # MEACONING scenario
    meaconing_bias_std: float = 4.0  # Hz — coherent meaconing bias
    meaconing_diff_std: float = 0.10  # Hz — small differential noise
    # SOPHISTICATED scenario
    sophisticated_ramp_lo: float = 0.05  # Hz/epoch — min ramp rate
    sophisticated_ramp_hi: float = 0.20  # Hz/epoch — max ramp rate
    sophisticated_diff_std: float = 0.05  # Hz — very small differential
    # Operating point
    p_fa_target: float = 1e-4
    p_miss_target: float = 0.01
    # Curve resolution
    n_thresholds: int = 2_000
    random_seed: int = 42
    batch_size: int = 10_000


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DETCurveData:
    """DET and ROC curve data for one scenario.

    Attributes:
        thresholds:           (K,) threshold values (ascending).
        p_fa:                 (K,) false alarm probabilities (decreasing with threshold).
        p_miss:               (K,) missed detection probabilities (increasing with threshold).
        auc_roc:              Area under ROC curve (trapezoid, probability space).
                              1.0 = perfect, 0.5 = random.
        auc_det:              Area under DET curve (P_miss vs P_fa, probability space).
                              0.0 = perfect, 0.5 = random.
        p_miss_at_target_fa:  P_miss at the p_fa_target operating point.
        p_fa_at_target_miss:  P_fa at the p_miss_target operating point.
    """

    thresholds: np.ndarray
    p_fa: np.ndarray
    p_miss: np.ndarray
    auc_roc: float
    auc_det: float
    p_miss_at_target_fa: float
    p_fa_at_target_miss: float


@dataclass
class ScenarioResult:
    """Validation results for one spoofing scenario.

    Attributes:
        scenario:          Scenario enum value.
        det:               Full DET/ROC curve data.
        p_fa_empirical:    Empirical P_fa at the NP threshold (≈ p_fa_target).
        p_miss_empirical:  Empirical P_miss at the NP threshold.
        np_threshold:      Analytically derived NP threshold for p_fa_target.
        target_met:        True iff p_fa_empirical < p_fa_target AND
                           p_miss_empirical < p_miss_target.
        n_trials:          Number of trials used.
    """

    scenario: SpoofingScenario
    det: DETCurveData
    p_fa_empirical: float
    p_miss_empirical: float
    np_threshold: float
    target_met: bool
    n_trials: int


@dataclass
class MCValidationResult:
    """Full Monte Carlo validation results for all three scenarios.

    Attributes:
        simplistic:    Results for the SIMPLISTIC scenario.
        meaconing:     Results for the MEACONING scenario.
        sophisticated: Results for the SOPHISTICATED scenario.
        config:        Configuration used for this run.
    """

    simplistic: ScenarioResult
    meaconing: ScenarioResult
    sophisticated: ScenarioResult
    config: MCValidationConfig

    def scenarios(self) -> dict[str, ScenarioResult]:
        """Return all scenario results keyed by scenario name."""
        return {
            SpoofingScenario.SIMPLISTIC.value: self.simplistic,
            SpoofingScenario.MEACONING.value: self.meaconing,
            SpoofingScenario.SOPHISTICATED.value: self.sophisticated,
        }

    def all_targets_met(self) -> bool:
        """True iff all three scenarios meet (P_fa < target AND P_miss < target)."""
        return (
            self.simplistic.target_met
            and self.meaconing.target_met
            and self.sophisticated.target_met
        )


# ---------------------------------------------------------------------------
# Core statistics (vectorised)
# ---------------------------------------------------------------------------


def _chi2_sf_vec(x: np.ndarray, df: float) -> np.ndarray:
    """Upper tail P(chi²(df) > x), vectorised over x via scipy.special.

    Uses the regularised upper incomplete gamma function:
        P(chi²(k) > x) = Γ(k/2, x/2) / Γ(k/2) = gammaincc(k/2, x/2)

    Args:
        x:  Non-negative array of chi-squared realisations.
        df: Degrees of freedom (positive scalar).

    Returns:
        Array of survival probabilities, same shape as x.
    """
    return _gammaincc(df * 0.5, np.maximum(x, 0.0) * 0.5)  # type: ignore[return-value]


def _fisher_score_batch(
    dev: np.ndarray,
    sigma_d: float,
    n_sats: int,
) -> np.ndarray:
    """Compute per-trial Fisher-combined (chi + coherent-SNR) score for one batch.

    Statistics per epoch t:
        chi_t  = Σ(δfᵢ − mean_t)² / σ_D²   — differential anomaly (df = n−1)
        coh_t  = n · mean_t²         / σ_D²  — common-mode anomaly  (df = 1)
        F_t    = −2(ln p_chi + ln p_coh)      — Fisher combination   (df = 4)

    Trial score = max_{t ∈ [0,T)} F_t

    Under H₀, both p_chi and p_coh are Uniform(0,1), so F_t ~ chi²(4)
    (Fisher's combination theorem).

    Args:
        dev:    (B, T, n) Doppler deviation array [Hz].
        sigma_d: Nominal Doppler noise 1-σ [Hz].
        n_sats: Number of satellites n.

    Returns:
        (B,) array of per-trial max Fisher scores (≥ 0).
    """
    sigma_sq = sigma_d**2

    # Differential statistic: chi_t ~ chi²(n-1) under H₀
    dev_mean = dev.mean(axis=2, keepdims=True)  # (B, T, 1)
    dev_centered = dev - dev_mean  # (B, T, n)
    chi = (dev_centered**2).sum(axis=2) / sigma_sq  # (B, T)

    # Common-mode statistic: coh_t ~ chi²(1) under H₀
    coh = n_sats * (dev_mean[..., 0] ** 2) / sigma_sq  # (B, T)

    # p-values (floor at _EPS to avoid log(0))
    p_chi = np.maximum(_chi2_sf_vec(chi, df=float(n_sats - 1)), _EPS)
    p_coh = np.maximum(_chi2_sf_vec(coh, df=1.0), _EPS)

    # Fisher combination: F_t = −2(ln p_chi + ln p_coh) ~ chi²(4) under H₀
    fisher_per_epoch = -2.0 * (np.log(p_chi) + np.log(p_coh))  # (B, T)

    # Per-trial score: max over T epochs
    return fisher_per_epoch.max(axis=1)  # (B,)


# ---------------------------------------------------------------------------
# Attack signal generation
# ---------------------------------------------------------------------------


def _generate_attack_bias(
    scenario: SpoofingScenario,
    batch_size: int,
    n_epochs: int,
    n_sats: int,
    config: MCValidationConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate (B, T, n) attack bias array for the given scenario.

    SIMPLISTIC:
        b_i(t) = b_common + δᵢ(t)
        b_common ~ N(0, σ_b²)  [constant over t, drawn once per trial]
        δᵢ(t)   ~ N(0, σ_d²)  [iid per satellite and epoch]

    MEACONING:
        Same structure as SIMPLISTIC but with much smaller σ_d,
        making satellites nearly identical → chi_t insensitive.

    SOPHISTICATED:
        b_i(t) = rate · t + δᵢ(t)
        rate ~ Uniform(lo, hi)  [drawn once per trial]
        δᵢ(t) ~ N(0, σ_d²)    [very small differential noise]
        The ramp grows to a detectable level after O(σ_D / rate) epochs.
    """
    B, T, n = batch_size, n_epochs, n_sats

    if scenario == SpoofingScenario.SIMPLISTIC:
        b_common = rng.normal(0.0, config.simplistic_bias_std, size=(B, 1, 1))
        delta = rng.normal(0.0, config.simplistic_diff_std, size=(B, T, n))
        return b_common + delta  # broadcasts (B,1,1) + (B,T,n) = (B,T,n)

    if scenario == SpoofingScenario.MEACONING:
        b_common = rng.normal(0.0, config.meaconing_bias_std, size=(B, 1, 1))
        delta = rng.normal(0.0, config.meaconing_diff_std, size=(B, T, n))
        return b_common + delta

    if scenario == SpoofingScenario.SOPHISTICATED:
        rate = rng.uniform(
            config.sophisticated_ramp_lo,
            config.sophisticated_ramp_hi,
            size=(B, 1, 1),
        )
        t_vec = np.arange(T, dtype=np.float64)  # (T,)
        b_ramp = rate * t_vec[np.newaxis, :, np.newaxis]  # (B, T, 1)
        delta = rng.normal(0.0, config.sophisticated_diff_std, size=(B, T, n))
        return b_ramp + delta  # (B, T, 1) + (B, T, n) = (B, T, n)

    raise ValueError(f"Unknown scenario: {scenario}")  # pragma: no cover


# ---------------------------------------------------------------------------
# NP threshold
# ---------------------------------------------------------------------------


def _np_threshold_fisher(p_fa_target: float, n_epochs: int) -> float:
    """Analytically derived NP threshold for the max-over-T Fisher score.

    Under H₀, each epoch contributes an iid F_t ~ chi²(4) sample.
    The trial score = max_t F_t has CDF:
        P(max_t F_t ≤ τ) = P(F ≤ τ)^T

    Setting P_fa = 1 − P(F ≤ τ)^T = p_fa_target:
        τ = chi²(4).ppf((1 − p_fa_target)^{1/T})

    Args:
        p_fa_target: Desired trial-level false alarm probability.
        n_epochs:    Number of epochs T per trial.

    Returns:
        Threshold τ ≥ 0.
    """
    cdf_per_epoch = (1.0 - p_fa_target) ** (1.0 / n_epochs)
    return float(_chi2_dist.ppf(cdf_per_epoch, df=_FISHER_DF))


# ---------------------------------------------------------------------------
# DET / ROC curve computation
# ---------------------------------------------------------------------------


def _compute_det_roc(
    nominal_scores: np.ndarray,
    attack_scores: np.ndarray,
    n_thresholds: int = 2_000,
    p_fa_target: float = 1e-4,
    p_miss_target: float = 0.01,
) -> DETCurveData:
    """Compute DET and ROC curves from score arrays.

    Uses sorted-array searchsorted for O(N log N + K log N) complexity,
    enabling efficient processing of N = 10⁶ score arrays.

    Args:
        nominal_scores: (N,) Fisher trial scores under H₀.
        attack_scores:  (N,) Fisher trial scores under H₁.
        n_thresholds:   Number of threshold sweep points K.
        p_fa_target:    P_fa operating point to evaluate P_miss.
        p_miss_target:  P_miss operating point to evaluate P_fa.

    Returns:
        DETCurveData with DET/ROC curves and operating-point metrics.
    """
    nom_sorted = np.sort(nominal_scores)  # ascending
    atk_sorted = np.sort(attack_scores)  # ascending
    N = len(nom_sorted)
    A = len(atk_sorted)

    # Threshold grid spanning 0.01–99.99th percentile of combined distribution
    all_scores = np.concatenate([nom_sorted, atk_sorted])
    lo = float(np.percentile(all_scores, 0.01))
    hi = float(np.percentile(all_scores, 99.99))
    thresholds = np.linspace(lo, hi, n_thresholds)  # (K,) ascending

    # P_fa(τ)   = fraction of nominal scores above τ
    # P_miss(τ) = fraction of attack scores at or below τ
    idx_nom = np.searchsorted(nom_sorted, thresholds, side="right")  # (K,)
    idx_atk = np.searchsorted(atk_sorted, thresholds, side="right")  # (K,)
    p_fa = 1.0 - idx_nom / N  # (K,) — decreasing with threshold
    p_miss = idx_atk / A  # (K,) — increasing with threshold
    tpr = 1.0 - p_miss  # (K,) P_D

    # Sort by ascending P_fa for correct trapezoid integration
    order = np.argsort(p_fa)
    p_fa_asc = p_fa[order]
    tpr_asc = tpr[order]
    p_miss_asc = p_miss[order]

    auc_roc = float(np.trapezoid(tpr_asc, p_fa_asc))
    auc_det = float(np.trapezoid(p_miss_asc, p_fa_asc))

    # Operating point: P_miss at p_fa_target
    fa_idx = int(np.argmin(np.abs(p_fa - p_fa_target)))
    p_miss_at_target_fa = float(p_miss[fa_idx])

    # Operating point: P_fa at p_miss_target
    miss_idx = int(np.argmin(np.abs(p_miss - p_miss_target)))
    p_fa_at_target_miss = float(p_fa[miss_idx])

    return DETCurveData(
        thresholds=thresholds,
        p_fa=p_fa,
        p_miss=p_miss,
        auc_roc=auc_roc,
        auc_det=auc_det,
        p_miss_at_target_fa=p_miss_at_target_fa,
        p_fa_at_target_miss=p_fa_at_target_miss,
    )


# ---------------------------------------------------------------------------
# Per-scenario Monte Carlo
# ---------------------------------------------------------------------------


def _run_scenario_mc(
    scenario: SpoofingScenario,
    config: MCValidationConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate nominal and attack trial scores for one scenario.

    Processes trials in batches of config.batch_size to bound peak memory.
    Peak memory per batch: O(B × T × n × 8 bytes) = 38 MB for default B=10,000.

    Args:
        scenario: Attack scenario enum value.
        config:   Simulation parameters.
        rng:      NumPy Generator (mutated in-place).

    Returns:
        (nominal_scores, attack_scores): each (n_trials,) float64 array.
    """
    N = config.n_trials
    B = config.batch_size
    n = config.n_sats
    T = config.n_epochs
    sigma_d = config.doppler_noise_std

    nominal_scores = np.empty(N, dtype=np.float64)
    attack_scores = np.empty(N, dtype=np.float64)

    n_done = 0
    while n_done < N:
        bs = min(B, N - n_done)

        # Nominal: pure Doppler noise, no attack
        noise_nom = rng.normal(0.0, sigma_d, size=(bs, T, n))
        nominal_scores[n_done : n_done + bs] = _fisher_score_batch(noise_nom, sigma_d, n)

        # Attack: Doppler noise + scenario-specific bias
        noise_atk = rng.normal(0.0, sigma_d, size=(bs, T, n))
        bias = _generate_attack_bias(scenario, bs, T, n, config, rng)
        attack_scores[n_done : n_done + bs] = _fisher_score_batch(noise_atk + bias, sigma_d, n)

        n_done += bs

    return nominal_scores, attack_scores


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_mc_validation(
    config: MCValidationConfig | None = None,
    rng: np.random.Generator | None = None,
) -> MCValidationResult:
    """Run large-scale Monte Carlo validation for all three spoofing scenarios.

    For each scenario, N = config.n_trials independent trials are simulated:
      - Nominal trials: pure Doppler noise, no attack.
      - Attack trials:  Doppler noise + scenario-specific bias.

    The trial-level Fisher score (max over T epochs) is used to compute:
      - DET curve: P_miss vs P_fa sweeping threshold.
      - ROC curve: TPR vs FPR (same sweep).
      - Empirical P_fa and P_miss at the analytically derived NP threshold.
      - Operating-point metrics at p_fa_target and p_miss_target.

    Args:
        config: Simulation parameters; uses MCValidationConfig() defaults if None.
        rng:    NumPy Generator; constructed from config.random_seed if None.

    Returns:
        MCValidationResult with per-scenario DET/ROC curves and target checks.
    """
    if config is None:
        config = MCValidationConfig()
    if rng is None:
        rng = np.random.default_rng(config.random_seed)

    np_tau = _np_threshold_fisher(config.p_fa_target, config.n_epochs)

    results: dict[str, ScenarioResult] = {}
    for scenario in SpoofingScenario:
        nominal_scores, attack_scores = _run_scenario_mc(scenario, config, rng)

        det = _compute_det_roc(
            nominal_scores,
            attack_scores,
            n_thresholds=config.n_thresholds,
            p_fa_target=config.p_fa_target,
            p_miss_target=config.p_miss_target,
        )

        p_fa_emp = float(np.mean(nominal_scores > np_tau))
        p_miss_emp = float(np.mean(attack_scores <= np_tau))

        results[scenario.value] = ScenarioResult(
            scenario=scenario,
            det=det,
            p_fa_empirical=p_fa_emp,
            p_miss_empirical=p_miss_emp,
            np_threshold=np_tau,
            target_met=(p_fa_emp < config.p_fa_target and p_miss_emp < config.p_miss_target),
            n_trials=config.n_trials,
        )

    return MCValidationResult(
        simplistic=results[SpoofingScenario.SIMPLISTIC.value],
        meaconing=results[SpoofingScenario.MEACONING.value],
        sophisticated=results[SpoofingScenario.SOPHISTICATED.value],
        config=config,
    )
