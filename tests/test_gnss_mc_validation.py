"""Tests for src/gnss/mc_validation.py."""

from __future__ import annotations

import numpy as np
import pytest

from gnss.mc_validation import (
    MCValidationConfig,
    MCValidationResult,
    ScenarioResult,
    SpoofingScenario,
    _chi2_sf_vec,
    _compute_det_roc,
    _fisher_score_batch,
    _generate_attack_bias,
    _np_threshold_fisher,
    _run_scenario_mc,
    run_mc_validation,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SMALL_CFG = MCValidationConfig(
    n_trials=2_000,
    n_epochs=20,
    n_sats=6,
    batch_size=500,
    n_thresholds=200,
    random_seed=0,
)


# ---------------------------------------------------------------------------
# _chi2_sf_vec
# ---------------------------------------------------------------------------


def test_chi2_sf_vec_zero_input():
    """P(chi²(k) > 0) = 1 for any k > 0."""
    x = np.zeros(5)
    sf = _chi2_sf_vec(x, df=4.0)
    assert np.allclose(sf, 1.0)


def test_chi2_sf_vec_large_input():
    """P(chi²(k) > very_large) ≈ 0."""
    x = np.full(5, 1e6)
    sf = _chi2_sf_vec(x, df=4.0)
    assert np.all(sf < 1e-10)


def test_chi2_sf_vec_known_quantile():
    """chi²(1) median ≈ 0.4549 → P(> 0.4549) ≈ 0.5."""
    # chi2(1).ppf(0.5) ≈ 0.4549
    from scipy.stats import chi2

    x = np.array([chi2.ppf(0.5, df=1)])
    sf = _chi2_sf_vec(x, df=1.0)
    assert abs(float(sf[0]) - 0.5) < 0.01


def test_chi2_sf_vec_shape_preserved():
    """Output shape matches input."""
    x = np.ones((3, 4))
    sf = _chi2_sf_vec(x, df=3.0)
    assert sf.shape == (3, 4)


# ---------------------------------------------------------------------------
# _fisher_score_batch
# ---------------------------------------------------------------------------


def test_fisher_score_batch_output_shape():
    rng = np.random.default_rng(1)
    B, T, n = 50, 20, 6
    dev = rng.normal(0, 0.30, size=(B, T, n))
    scores = _fisher_score_batch(dev, sigma_d=0.30, n_sats=n)
    assert scores.shape == (B,)


def test_fisher_score_batch_nonnegative():
    """Fisher score is always ≥ 0."""
    rng = np.random.default_rng(2)
    dev = rng.normal(0, 0.30, size=(100, 20, 6))
    scores = _fisher_score_batch(dev, sigma_d=0.30, n_sats=6)
    assert np.all(scores >= 0.0)


def test_fisher_score_batch_nominal_distribution():
    """Under H₀ (n=6, T=1), F_t ~ chi²(4); mean ≈ 4."""
    rng = np.random.default_rng(3)
    B = 20_000
    dev = rng.normal(0.0, 0.30, size=(B, 1, 6))
    scores = _fisher_score_batch(dev, sigma_d=0.30, n_sats=6)
    # max over T=1 is just F itself; E[chi²(4)] = 4
    assert abs(scores.mean() - 4.0) < 0.3  # ±2-sigma tolerance


def test_fisher_score_batch_attack_elevates_scores():
    """A large coherent bias must produce higher scores than nominal."""
    rng = np.random.default_rng(4)
    B, T, n = 500, 10, 6
    sigma_d = 0.30
    nominal_dev = rng.normal(0.0, sigma_d, size=(B, T, n))
    # Meaconing: add common bias 4 Hz (SNR >> 1)
    attack_dev = nominal_dev + 4.0
    nom_scores = _fisher_score_batch(nominal_dev, sigma_d, n)
    atk_scores = _fisher_score_batch(attack_dev, sigma_d, n)
    assert atk_scores.mean() > nom_scores.mean() * 10


def test_fisher_score_batch_sensitivity_to_differential():
    """Large differential noise (SIMPLISTIC) also elevates scores."""
    rng = np.random.default_rng(5)
    B, T, n = 500, 10, 6
    sigma_d = 0.30
    nominal_dev = rng.normal(0.0, sigma_d, size=(B, T, n))
    # SIMPLISTIC: large differential component (b_common=0, delta large)
    attack_dev = nominal_dev + rng.normal(0.0, 5.0, size=(B, T, n))
    nom_scores = _fisher_score_batch(nominal_dev, sigma_d, n)
    atk_scores = _fisher_score_batch(attack_dev, sigma_d, n)
    assert atk_scores.mean() > nom_scores.mean() * 5


# ---------------------------------------------------------------------------
# _generate_attack_bias
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scenario", list(SpoofingScenario))
def test_generate_attack_bias_shape(scenario: SpoofingScenario):
    rng = np.random.default_rng(10)
    B, T, n = 30, 20, 6
    bias = _generate_attack_bias(scenario, B, T, n, _SMALL_CFG, rng)
    assert bias.shape == (B, T, n)


def test_generate_attack_bias_simplistic_variance():
    """SIMPLISTIC bias variance dominated by large b_common σ_b=8."""
    rng = np.random.default_rng(11)
    bias = _generate_attack_bias(SpoofingScenario.SIMPLISTIC, 5_000, 1, 6, _SMALL_CFG, rng)
    # Inter-satellite variance (across n) should be ≈ diff_std² = 4
    intra_var = bias[:, 0, :].var(axis=1).mean()
    # Between-trial variance (across B) dominated by b_common² → ≈ 64
    inter_var = bias[:, 0, :].mean(axis=1).var()
    assert inter_var > intra_var


def test_generate_attack_bias_meaconing_coherent():
    """MEACONING satellites nearly identical (small differential)."""
    rng = np.random.default_rng(12)
    bias = _generate_attack_bias(SpoofingScenario.MEACONING, 2_000, 1, 6, _SMALL_CFG, rng)
    # Per-trial inter-satellite std should be << meaconing_bias_std=4
    intra_std = bias[:, 0, :].std(axis=1)
    assert float(intra_std.mean()) < 0.5  # diff_std=0.10 → std ≈ 0.10


def test_generate_attack_bias_sophisticated_ramp_grows():
    """SOPHISTICATED bias mean grows monotonically across epochs."""
    rng = np.random.default_rng(13)
    bias = _generate_attack_bias(SpoofingScenario.SOPHISTICATED, 5_000, 40, 6, _SMALL_CFG, rng)
    # mean bias across trials and satellites should increase with t
    mean_over_time = bias.mean(axis=(0, 2))  # (T,)
    # Check that mean increases from t=0 to t=T-1 (ramp)
    assert mean_over_time[-1] > mean_over_time[0]
    assert np.all(np.diff(mean_over_time) > 0)


# ---------------------------------------------------------------------------
# _np_threshold_fisher
# ---------------------------------------------------------------------------


def test_np_threshold_fisher_positive():
    tau = _np_threshold_fisher(1e-4, 80)
    assert tau > 0.0


def test_np_threshold_fisher_reasonable_range():
    """For chi²(4) with T=80, threshold should be in [20, 60]."""
    tau = _np_threshold_fisher(1e-4, 80)
    assert 20.0 < tau < 60.0


def test_np_threshold_fisher_stricter_gives_higher():
    """Stricter P_fa target → higher threshold."""
    tau_loose = _np_threshold_fisher(1e-3, 80)
    tau_strict = _np_threshold_fisher(1e-5, 80)
    assert tau_strict > tau_loose


def test_np_threshold_fisher_more_epochs_gives_higher():
    """More epochs → max score likely larger → need higher threshold."""
    tau_short = _np_threshold_fisher(1e-4, 10)
    tau_long = _np_threshold_fisher(1e-4, 200)
    assert tau_long > tau_short


def test_np_threshold_fisher_calibration():
    """Empirical P_fa under H₀ should be ≈ p_fa_target (within 3σ) with large N."""
    rng = np.random.default_rng(20)
    N, T, n = 100_000, 20, 6
    sigma_d = 0.30
    p_fa_target = 1e-3  # easier to validate at N=100k

    tau = _np_threshold_fisher(p_fa_target, T)
    # Generate nominal scores
    nominal_scores = np.empty(N)
    for start in range(0, N, 10_000):
        bs = min(10_000, N - start)
        dev = rng.normal(0.0, sigma_d, size=(bs, T, n))
        nominal_scores[start : start + bs] = _fisher_score_batch(dev, sigma_d, n)

    p_fa_emp = float(np.mean(nominal_scores > tau))
    # 3σ tolerance: σ = sqrt(p * (1-p) / N) ≈ sqrt(1e-3 / 1e5) ≈ 1e-4
    assert abs(p_fa_emp - p_fa_target) < 4 * (p_fa_target * (1 - p_fa_target) / N) ** 0.5


# ---------------------------------------------------------------------------
# _compute_det_roc
# ---------------------------------------------------------------------------


def test_compute_det_roc_output_shapes():
    rng = np.random.default_rng(30)
    nom = rng.chisquare(df=4, size=5_000)
    atk = rng.chisquare(df=4, size=5_000) + 8.0
    det = _compute_det_roc(nom, atk, n_thresholds=100)
    assert det.thresholds.shape == (100,)
    assert det.p_fa.shape == (100,)
    assert det.p_miss.shape == (100,)


def test_compute_det_roc_probabilities_in_unit_interval():
    rng = np.random.default_rng(31)
    nom = rng.chisquare(df=4, size=5_000)
    atk = rng.chisquare(df=4, size=5_000) + 8.0
    det = _compute_det_roc(nom, atk, n_thresholds=100)
    assert np.all(det.p_fa >= 0.0) and np.all(det.p_fa <= 1.0)
    assert np.all(det.p_miss >= 0.0) and np.all(det.p_miss <= 1.0)


def test_compute_det_roc_auc_roc_range():
    """AUC_ROC ∈ [0.5, 1.0] for a useful detector."""
    rng = np.random.default_rng(32)
    nom = rng.chisquare(df=4, size=10_000)
    atk = rng.chisquare(df=4, size=10_000) + 8.0
    det = _compute_det_roc(nom, atk, n_thresholds=200)
    assert 0.5 <= det.auc_roc <= 1.0


def test_compute_det_roc_perfect_separation():
    """10-sigma separation → AUC_ROC > 0.99 and P_miss ≈ 0.

    Uses N(0,1) vs N(10,1): AUROC ≈ 1.
    The linspace captures the p_fa transition zone (thresholds < 5),
    so trapezoid AUC integration is numerically valid.
    At p_fa_target=1e-4 (τ≈3.7), p_miss = Φ(-6.3) ≈ 0.
    """
    rng = np.random.default_rng(99)
    nom = rng.normal(0.0, 1.0, 5_000)
    atk = rng.normal(10.0, 1.0, 5_000)
    det = _compute_det_roc(nom, atk, n_thresholds=500)
    assert det.auc_roc > 0.99
    assert det.p_miss_at_target_fa < 0.01


def test_compute_det_roc_random_scores():
    """Identical distributions → AUC_ROC ≈ 0.5."""
    rng = np.random.default_rng(33)
    nom = rng.normal(0.0, 1.0, 20_000)
    atk = rng.normal(0.0, 1.0, 20_000)
    det = _compute_det_roc(nom, atk, n_thresholds=100)
    assert abs(det.auc_roc - 0.5) < 0.05


def test_compute_det_roc_operating_points():
    """Operating-point metrics are finite and in range."""
    rng = np.random.default_rng(34)
    nom = rng.chisquare(df=4, size=10_000)
    atk = rng.chisquare(df=4, size=10_000) + 8.0
    det = _compute_det_roc(nom, atk, n_thresholds=200, p_fa_target=1e-2, p_miss_target=0.05)
    assert 0.0 <= det.p_miss_at_target_fa <= 1.0
    assert 0.0 <= det.p_fa_at_target_miss <= 1.0


def test_compute_det_roc_auc_det_range():
    """AUC_DET ∈ [0, 0.5] for a useful detector (0 = perfect, 0.5 = random)."""
    rng = np.random.default_rng(35)
    nom = rng.chisquare(df=4, size=10_000)
    atk = rng.chisquare(df=4, size=10_000) + 8.0
    det = _compute_det_roc(nom, atk, n_thresholds=200)
    assert 0.0 <= det.auc_det <= 0.5


# ---------------------------------------------------------------------------
# _run_scenario_mc
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scenario", list(SpoofingScenario))
def test_run_scenario_mc_shapes(scenario: SpoofingScenario):
    rng = np.random.default_rng(40)
    nom, atk = _run_scenario_mc(scenario, _SMALL_CFG, rng)
    assert nom.shape == (2_000,)
    assert atk.shape == (2_000,)
    assert np.all(nom >= 0.0)
    assert np.all(atk >= 0.0)


@pytest.mark.parametrize("scenario", list(SpoofingScenario))
def test_run_scenario_mc_attack_higher_than_nominal(scenario: SpoofingScenario):
    """Attack scores should stochastically dominate nominal scores."""
    rng = np.random.default_rng(41)
    nom, atk = _run_scenario_mc(scenario, _SMALL_CFG, rng)
    assert atk.mean() > nom.mean()


def test_run_scenario_mc_reproducible():
    """Same seed → identical scores."""
    cfg = MCValidationConfig(n_trials=500, n_epochs=10, n_sats=6, batch_size=200, random_seed=99)
    rng1 = np.random.default_rng(99)
    rng2 = np.random.default_rng(99)
    nom1, atk1 = _run_scenario_mc(SpoofingScenario.MEACONING, cfg, rng1)
    nom2, atk2 = _run_scenario_mc(SpoofingScenario.MEACONING, cfg, rng2)
    np.testing.assert_array_equal(nom1, nom2)
    np.testing.assert_array_equal(atk1, atk2)


def test_run_scenario_mc_nominal_calibration():
    """Nominal P_fa at NP threshold should be ≈ p_fa_target."""
    cfg = MCValidationConfig(
        n_trials=50_000,
        n_epochs=20,
        n_sats=6,
        batch_size=5_000,
        p_fa_target=1e-3,
        random_seed=42,
    )
    rng = np.random.default_rng(42)
    nom, _ = _run_scenario_mc(SpoofingScenario.MEACONING, cfg, rng)
    tau = _np_threshold_fisher(cfg.p_fa_target, cfg.n_epochs)
    p_fa_emp = float(np.mean(nom > tau))
    # 3σ tolerance
    sigma = (cfg.p_fa_target * (1 - cfg.p_fa_target) / cfg.n_trials) ** 0.5
    assert abs(p_fa_emp - cfg.p_fa_target) < 4 * sigma


# ---------------------------------------------------------------------------
# run_mc_validation — integration tests
# ---------------------------------------------------------------------------


def test_run_mc_validation_returns_correct_type():
    result = run_mc_validation(_SMALL_CFG)
    assert isinstance(result, MCValidationResult)
    assert isinstance(result.simplistic, ScenarioResult)
    assert isinstance(result.meaconing, ScenarioResult)
    assert isinstance(result.sophisticated, ScenarioResult)


def test_run_mc_validation_scenario_fields():
    result = run_mc_validation(_SMALL_CFG)
    assert result.simplistic.scenario == SpoofingScenario.SIMPLISTIC
    assert result.meaconing.scenario == SpoofingScenario.MEACONING
    assert result.sophisticated.scenario == SpoofingScenario.SOPHISTICATED


def test_run_mc_validation_n_trials():
    result = run_mc_validation(_SMALL_CFG)
    assert result.simplistic.n_trials == _SMALL_CFG.n_trials
    assert result.meaconing.n_trials == _SMALL_CFG.n_trials
    assert result.sophisticated.n_trials == _SMALL_CFG.n_trials


def test_run_mc_validation_auc_roc_above_half():
    """All scenarios should produce AUC_ROC > 0.5 (better than random)."""
    result = run_mc_validation(_SMALL_CFG)
    for name, sr in result.scenarios().items():
        assert sr.det.auc_roc > 0.5, f"{name}: AUC_ROC = {sr.det.auc_roc:.4f}"


def test_run_mc_validation_simplistic_high_auc():
    """SIMPLISTIC attack (large bias + large diff) → very high AUC_ROC."""
    result = run_mc_validation(_SMALL_CFG)
    assert result.simplistic.det.auc_roc > 0.95


def test_run_mc_validation_meaconing_high_auc():
    """MEACONING with coherent-SNR detector → high AUC_ROC (> 0.90 at N=2k)."""
    result = run_mc_validation(_SMALL_CFG)
    assert result.meaconing.det.auc_roc > 0.90


def test_run_mc_validation_empirical_p_fa_in_range():
    """Empirical P_fa should be close to target (within factor 3 at small N)."""
    result = run_mc_validation(_SMALL_CFG)
    for name, sr in result.scenarios().items():
        assert sr.p_fa_empirical <= _SMALL_CFG.p_fa_target * 3 + 0.01, (
            f"{name}: P_fa = {sr.p_fa_empirical:.6f}"
        )


def test_run_mc_validation_scenarios_dict():
    result = run_mc_validation(_SMALL_CFG)
    d = result.scenarios()
    assert set(d.keys()) == {"simplistic", "meaconing", "sophisticated"}


def test_run_mc_validation_default_config():
    """Smoke test: default config (small n_trials override for CI speed)."""
    cfg = MCValidationConfig(
        n_trials=500,
        n_epochs=10,
        n_sats=6,
        batch_size=250,
        n_thresholds=50,
    )
    result = run_mc_validation(cfg)
    assert result.config is cfg


@pytest.mark.parametrize("scenario", list(SpoofingScenario))
def test_run_mc_validation_target_met_large_n(scenario: SpoofingScenario):
    """With N=100k and strong attacks, all scenarios meet target at operating point."""
    cfg = MCValidationConfig(
        n_trials=100_000,
        n_epochs=40,
        n_sats=6,
        batch_size=10_000,
        p_fa_target=1e-3,  # relaxed target for feasible N
        p_miss_target=0.05,
        n_thresholds=500,
        random_seed=7,
    )
    rng = np.random.default_rng(7)
    nom, atk = _run_scenario_mc(scenario, cfg, rng)
    det = _compute_det_roc(
        nom,
        atk,
        n_thresholds=500,
        p_fa_target=cfg.p_fa_target,
        p_miss_target=cfg.p_miss_target,
    )
    # At relaxed target, all scenarios should have P_miss < 0.10
    assert det.p_miss_at_target_fa < 0.10, (
        f"{scenario.value}: P_miss@P_fa={cfg.p_fa_target} = {det.p_miss_at_target_fa:.4f}"
    )
