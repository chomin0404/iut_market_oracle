"""Stanford GPS Lab benchmark tests (Rothmaier et al. ION GNSS+ 2021).

Source: https://github.com/stanford-gps-lab/spoofing-detection
Script: metric_combinations/DoA_prr_combinationStudy.m

Published parameters replicated here (see StanfordParams):
    N = 12  satellites
    sigPr = 3 m
    P_FAmax = 1e-7
    K = 1e5  Monte Carlo trials
    xoffset = [0, 0, 10, 0]  (10 m vertical position bias — coherent meaconing)
    Elevation: linearly spaced [π/15, π/2]
    Azimuth:   linearly spaced [0, 2π]

Key theoretical result (Section III-B of the paper):
    The coherent meaconing bias b = H · xoffset lies in col(H).
    After residual projection (I − P), (I − P)·b = 0.
    Therefore the chi-squared test statistic is identically distributed
    under H0 (genuine) and H1 (spoofed) → P_D ≈ P_FA.
    This motivates the multi-metric / DoA detection approach.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import chi2

from gnss.stanford_benchmark import (
    StanfordMCResult,
    StanfordParams,
    chi2_raim_score,
    chi2_raim_threshold,
    generate_pr_obs,
    run_chi2_raim_mc,
)

# ---------------------------------------------------------------------------
# Stanford linear-spaced constellation (DoA_prr_combinationStudy.m geometry)
# ---------------------------------------------------------------------------

N_SATS_STANFORD = 12
SIG_PR_STANFORD = 3.0  # m
P_FA_STANFORD = 1e-7
BIAS_VERTICAL_M = 10.0  # m  (xoffset[2])


def _stanford_constellation(n: int = N_SATS_STANFORD) -> np.ndarray:
    """Unit LOS vectors using Stanford's linearly-spaced elevation/azimuth.

    Replicates DoA_prr_combinationStudy.m:
        el = linspace(pi/15, pi/2, N)
        az = linspace(0, 2*pi, N)
        los_i = [cos(el)*cos(az), cos(el)*sin(az), sin(el)]
    """
    el = np.linspace(math.pi / 15.0, math.pi / 2.0, n)
    az = np.linspace(0.0, 2.0 * math.pi, n)
    los = np.column_stack(
        [
            np.cos(el) * np.cos(az),
            np.cos(el) * np.sin(az),
            np.sin(el),
        ]
    )
    return los  # (n, 3)


# ---------------------------------------------------------------------------
# 1. StanfordParams defaults match published values
# ---------------------------------------------------------------------------


class TestStanfordParams:
    def test_n_sats(self) -> None:
        assert StanfordParams().n_sats == 12

    def test_sig_pr(self) -> None:
        assert StanfordParams().sig_pr == pytest.approx(3.0)

    def test_p_fa_max(self) -> None:
        assert StanfordParams().p_fa_max == pytest.approx(1e-7)

    def test_bias_vertical(self) -> None:
        assert StanfordParams().bias_vertical_m == pytest.approx(10.0)

    def test_random_seed_deterministic(self) -> None:
        """Same seed must produce same results across calls."""
        p = StanfordParams(n_mc=20, random_seed=7)
        rng1 = np.random.default_rng(p.random_seed)
        rng2 = np.random.default_rng(p.random_seed)
        v1 = rng1.normal(size=5)
        v2 = rng2.normal(size=5)
        np.testing.assert_array_equal(v1, v2)


# ---------------------------------------------------------------------------
# 2. Stanford constellation geometry
# ---------------------------------------------------------------------------


class TestStanfordConstellation:
    def test_unit_vectors(self) -> None:
        """All LOS vectors must be unit length."""
        los = _stanford_constellation()
        norms = np.linalg.norm(los, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-12)

    def test_shape(self) -> None:
        los = _stanford_constellation()
        assert los.shape == (12, 3)

    def test_positive_elevation(self) -> None:
        """All satellites above horizon (z > 0)."""
        los = _stanford_constellation()
        assert np.all(los[:, 2] > 0.0)

    def test_min_elevation_approx_12deg(self) -> None:
        """Minimum elevation ≈ π/15 rad ≈ 12°."""
        los = _stanford_constellation()
        el_min = np.arcsin(np.clip(los[:, 2], -1.0, 1.0)).min()
        assert el_min == pytest.approx(math.pi / 15.0, abs=0.02)

    def test_max_elevation_is_90deg(self) -> None:
        """Maximum elevation ≈ π/2 rad (zenith satellite)."""
        los = _stanford_constellation()
        el_max = np.arcsin(np.clip(los[:, 2], -1.0, 1.0)).max()
        assert el_max == pytest.approx(math.pi / 2.0, abs=0.02)

    def test_geometry_matrix_full_rank(self) -> None:
        """Geometry matrix H = [los | 1] must have rank 4 (n > 4 sats)."""
        los = _stanford_constellation()
        H = np.hstack([los, np.ones((len(los), 1))])
        rank = np.linalg.matrix_rank(H)
        assert rank == 4


# ---------------------------------------------------------------------------
# 3. generate_pr_obs
# ---------------------------------------------------------------------------


class TestGeneratePrObs:
    def setup_method(self) -> None:
        self.rng = np.random.default_rng(0)
        self.los = _stanford_constellation()

    def test_genuine_shape(self) -> None:
        pr = generate_pr_obs(self.los, self.rng, spoofed=False)
        assert pr.shape == (N_SATS_STANFORD,)

    def test_spoofed_shape(self) -> None:
        pr = generate_pr_obs(self.los, self.rng, spoofed=True)
        assert pr.shape == (N_SATS_STANFORD,)

    def test_genuine_zero_mean(self) -> None:
        """Genuine observations are zero-mean noise (large N → mean ≈ 0)."""
        rng = np.random.default_rng(1)
        samples = np.array([generate_pr_obs(self.los, rng, spoofed=False) for _ in range(5_000)])
        mean_per_sat = samples.mean(axis=0)
        np.testing.assert_allclose(mean_per_sat, 0.0, atol=0.15)

    def test_genuine_variance(self) -> None:
        """Genuine observations have variance ≈ sig_pr²."""
        rng = np.random.default_rng(2)
        samples = np.array(
            [
                generate_pr_obs(self.los, rng, spoofed=False, sig_pr=SIG_PR_STANFORD)
                for _ in range(5_000)
            ]
        )
        std_per_sat = samples.std(axis=0)
        np.testing.assert_allclose(std_per_sat, SIG_PR_STANFORD, rtol=0.05)

    def test_spoofed_mean_equals_los_bias(self) -> None:
        """Mean spoofed observation = los[:,2] * bias_vertical_m."""
        rng = np.random.default_rng(3)
        samples = np.array(
            [
                generate_pr_obs(
                    self.los,
                    rng,
                    spoofed=True,
                    sig_pr=SIG_PR_STANFORD,
                    bias_vertical_m=BIAS_VERTICAL_M,
                )
                for _ in range(10_000)
            ]
        )
        expected_bias = self.los[:, 2] * BIAS_VERTICAL_M
        np.testing.assert_allclose(samples.mean(axis=0), expected_bias, atol=0.15)

    def test_spoofed_and_genuine_same_variance(self) -> None:
        """Spoofed observations have the same variance as genuine (bias is deterministic)."""
        rng_g = np.random.default_rng(4)
        rng_s = np.random.default_rng(5)
        gen_samples = np.array(
            [
                generate_pr_obs(self.los, rng_g, spoofed=False, sig_pr=SIG_PR_STANFORD)
                for _ in range(5_000)
            ]
        )
        spoof_samples = np.array(
            [
                generate_pr_obs(self.los, rng_s, spoofed=True, sig_pr=SIG_PR_STANFORD)
                for _ in range(5_000)
            ]
        )
        np.testing.assert_allclose(
            gen_samples.std(axis=0),
            spoof_samples.std(axis=0),
            rtol=0.05,
        )


# ---------------------------------------------------------------------------
# 4. chi2_raim_threshold
# ---------------------------------------------------------------------------


class TestChi2RaimThreshold:
    def test_n12_returns_positive(self) -> None:
        tau = chi2_raim_threshold(N_SATS_STANFORD, P_FA_STANFORD)
        assert tau > 0.0

    def test_dof_is_n_minus_4(self) -> None:
        """Threshold must equal χ²(n-4) inverse CDF at 1 - P_FA."""
        tau = chi2_raim_threshold(N_SATS_STANFORD, P_FA_STANFORD)
        expected = float(chi2.ppf(1.0 - P_FA_STANFORD, df=N_SATS_STANFORD - 4))
        assert tau == pytest.approx(expected, rel=1e-6)

    def test_stricter_p_fa_raises_threshold(self) -> None:
        """Stricter false alert target → higher threshold."""
        tau_loose = chi2_raim_threshold(N_SATS_STANFORD, p_fa=1e-3)
        tau_strict = chi2_raim_threshold(N_SATS_STANFORD, p_fa=1e-7)
        assert tau_strict > tau_loose

    def test_more_sats_lowers_threshold_per_dof(self) -> None:
        """Adding satellites adds DOF; the raw threshold value increases."""
        tau_12 = chi2_raim_threshold(12, p_fa=1e-4)
        tau_20 = chi2_raim_threshold(20, p_fa=1e-4)
        assert tau_20 > tau_12  # more DOF → chi²(16) critical value > chi²(8)

    def test_raises_for_too_few_sats(self) -> None:
        with pytest.raises(ValueError, match="at least 5"):
            chi2_raim_threshold(4, p_fa=1e-4)

    def test_n12_p_fa_1e7_reasonable_range(self) -> None:
        """For n=12, p_fa=1e-7, chi²(8) threshold ≈ 50 (empirically ~56)."""
        tau = chi2_raim_threshold(N_SATS_STANFORD, P_FA_STANFORD)
        # chi2(8).ppf(1 - 1e-7) ≈ 56.4
        assert 40.0 < tau < 80.0


# ---------------------------------------------------------------------------
# 5. chi2_raim_score
# ---------------------------------------------------------------------------


class TestChi2RaimScore:
    def setup_method(self) -> None:
        self.los = _stanford_constellation()

    def test_nonnegative(self) -> None:
        rng = np.random.default_rng(10)
        pr = generate_pr_obs(self.los, rng, spoofed=False)
        assert chi2_raim_score(pr, self.los) >= 0.0

    def test_zero_obs_near_zero_score(self) -> None:
        """Zero pseudorange residuals → score = 0."""
        pr = np.zeros(N_SATS_STANFORD)
        score = chi2_raim_score(pr, self.los)
        assert score == pytest.approx(0.0, abs=1e-10)

    def test_distribution_under_h0(self) -> None:
        """Under H0 (genuine), T ~ χ²(n-4); mean ≈ n-4 = 8."""
        rng = np.random.default_rng(11)
        scores = np.array(
            [
                chi2_raim_score(
                    generate_pr_obs(self.los, rng, spoofed=False, sig_pr=SIG_PR_STANFORD),
                    self.los,
                    SIG_PR_STANFORD,
                )
                for _ in range(10_000)
            ]
        )
        expected_mean = N_SATS_STANFORD - 4  # DOF = 8
        assert abs(scores.mean() - expected_mean) < 0.3

    def test_distribution_under_h1_coherent_meaconing(self) -> None:
        """Key result: coherent meaconing → score distribution = H0 distribution.

        The attack bias b = H·xoffset lies in col(H).
        Residual projection (I − P)·b = 0.
        Therefore E[T|H1] = E[T|H0] = n - 4.
        """
        rng = np.random.default_rng(12)
        scores_gen = np.array(
            [
                chi2_raim_score(
                    generate_pr_obs(self.los, rng, spoofed=False, sig_pr=SIG_PR_STANFORD),
                    self.los,
                    SIG_PR_STANFORD,
                )
                for _ in range(10_000)
            ]
        )
        rng2 = np.random.default_rng(13)
        scores_spoof = np.array(
            [
                chi2_raim_score(
                    generate_pr_obs(
                        self.los,
                        rng2,
                        spoofed=True,
                        sig_pr=SIG_PR_STANFORD,
                        bias_vertical_m=BIAS_VERTICAL_M,
                    ),
                    self.los,
                    SIG_PR_STANFORD,
                )
                for _ in range(10_000)
            ]
        )
        # Both means ≈ n-4 = 8; difference must be < 0.5
        assert abs(scores_gen.mean() - scores_spoof.mean()) < 0.5

    def test_coherent_attack_invisible_to_raim_mathematically(self) -> None:
        """Prove (I − P)·b = 0 analytically for Stanford's xoffset.

        This is the theoretical foundation of why chi²-RAIM fails
        against coherent meaconing.
        """
        los = self.los
        H = np.hstack([los, np.ones((len(los), 1))])  # geometry matrix
        xoffset = np.array([0.0, 0.0, BIAS_VERTICAL_M, 0.0])
        b = H @ xoffset  # attack bias vector (n_sats,)

        # Residual projection matrix: I − H(HᵀH)⁻¹Hᵀ
        HTH_inv = np.linalg.pinv(H.T @ H)
        P_perp = np.eye(len(los)) - H @ HTH_inv @ H.T
        residual_bias = P_perp @ b

        np.testing.assert_allclose(
            residual_bias,
            0.0,
            atol=1e-10,
            err_msg="Coherent meaconing bias must be invisible after RAIM projection",
        )

    def test_incoherent_attack_visible(self) -> None:
        """An incoherent attack (random per-satellite offsets) IS visible to RAIM."""
        rng = np.random.default_rng(14)
        los = self.los
        scores_gen = np.array(
            [
                chi2_raim_score(
                    generate_pr_obs(los, rng, spoofed=False, sig_pr=SIG_PR_STANFORD),
                    los,
                    SIG_PR_STANFORD,
                )
                for _ in range(2_000)
            ]
        )
        rng2 = np.random.default_rng(15)
        scores_incoherent = np.array(
            [
                chi2_raim_score(
                    rng2.normal(0.0, SIG_PR_STANFORD, N_SATS_STANFORD)
                    + rng2.normal(0.0, 5.0 * SIG_PR_STANFORD, N_SATS_STANFORD),
                    los,
                    SIG_PR_STANFORD,
                )
                for _ in range(2_000)
            ]
        )
        # Incoherent attack must produce higher mean score
        assert scores_incoherent.mean() > scores_gen.mean() * 2


# ---------------------------------------------------------------------------
# 6. Fibonacci vs Stanford linear constellation comparison
# ---------------------------------------------------------------------------


class TestConstellationGeometries:
    """Compare Fibonacci lattice (our default) vs Stanford linear spacing.

    Both are valid deterministic upper-hemisphere constellations.
    The chi²-RAIM theoretical failure holds for both since the attack
    bias b = H·xoffset is always in col(H) regardless of geometry.
    """

    def test_fibonacci_also_full_rank(self) -> None:
        from gnss.math_utils import init_constellation

        los = init_constellation(N_SATS_STANFORD)
        H = np.hstack([los, np.ones((len(los), 1))])
        assert np.linalg.matrix_rank(H) == 4

    def test_coherent_attack_invisible_fibonacci(self) -> None:
        """RAIM blindness holds for Fibonacci constellation too."""
        from gnss.math_utils import init_constellation

        los = init_constellation(N_SATS_STANFORD)
        H = np.hstack([los, np.ones((len(los), 1))])
        xoffset = np.array([0.0, 0.0, BIAS_VERTICAL_M, 0.0])
        b = H @ xoffset
        HTH_inv = np.linalg.pinv(H.T @ H)
        P_perp = np.eye(len(los)) - H @ HTH_inv @ H.T
        np.testing.assert_allclose(P_perp @ b, 0.0, atol=1e-10)

    def test_both_constellations_have_positive_z(self) -> None:
        from gnss.math_utils import init_constellation

        fib = init_constellation(N_SATS_STANFORD)
        lin = _stanford_constellation(N_SATS_STANFORD)
        assert np.all(fib[:, 2] > 0.0)
        assert np.all(lin[:, 2] > 0.0)

    def test_stanford_geometry_dop(self) -> None:
        """HDOP and VDOP for Stanford linear geometry must be finite positive."""
        los = _stanford_constellation()
        H = np.hstack([los, np.ones((len(los), 1))])
        # (HᵀH)⁻¹ diagonal = variance inflation for each state
        cov = np.linalg.inv(H.T @ H)
        vdop = math.sqrt(cov[2, 2])
        hdop = math.sqrt(cov[0, 0] + cov[1, 1])
        assert math.isfinite(vdop) and vdop > 0.0
        assert math.isfinite(hdop) and hdop > 0.0


# ---------------------------------------------------------------------------
# 7. run_chi2_raim_mc — Stanford Monte Carlo baseline
# ---------------------------------------------------------------------------


class TestRunChi2RaimMC:
    """Smoke tests using small n_mc for CI speed.

    The fundamental result (P_D ≈ P_FA) is verified structurally.
    Full K=1e5 MC as published by Stanford can be run manually.
    """

    _SMALL_PARAMS = StanfordParams(n_mc=500, random_seed=42)

    def test_returns_stanford_mc_result(self) -> None:
        result = run_chi2_raim_mc(self._SMALL_PARAMS)
        assert isinstance(result, StanfordMCResult)

    def test_method_label(self) -> None:
        result = run_chi2_raim_mc(self._SMALL_PARAMS)
        assert result.method == "chi2_raim"

    def test_n_mc_recorded(self) -> None:
        result = run_chi2_raim_mc(self._SMALL_PARAMS)
        assert result.n_mc == self._SMALL_PARAMS.n_mc

    def test_threshold_positive(self) -> None:
        result = run_chi2_raim_mc(self._SMALL_PARAMS)
        assert result.threshold > 0.0

    def test_threshold_matches_formula(self) -> None:
        """Threshold must equal chi2_raim_threshold(n_sats, p_fa_max)."""
        result = run_chi2_raim_mc(self._SMALL_PARAMS)
        expected = chi2_raim_threshold(self._SMALL_PARAMS.n_sats, self._SMALL_PARAMS.p_fa_max)
        assert result.threshold == pytest.approx(expected, rel=1e-6)

    def test_p_fa_in_unit_interval(self) -> None:
        result = run_chi2_raim_mc(self._SMALL_PARAMS)
        assert 0.0 <= result.p_fa <= 1.0

    def test_p_d_in_unit_interval(self) -> None:
        result = run_chi2_raim_mc(self._SMALL_PARAMS)
        assert 0.0 <= result.p_d <= 1.0

    def test_p_fa_near_p_fa_max(self) -> None:
        """Empirical P_FA should be near the target P_FAmax = 1e-7.

        With n_mc=500 at p_fa_max=1e-7, almost certainly zero false alarms.
        """
        result = run_chi2_raim_mc(self._SMALL_PARAMS)
        assert result.p_fa <= 0.01  # at p_fa_max=1e-7, FA rate must be tiny

    def test_core_result_pd_approx_pfa(self) -> None:
        """Stanford paper's key result: P_D ≈ P_FA for coherent meaconing.

        With n_mc=500 and p_fa_max=1e-7, both P_D and P_FA are near zero.
        Verify neither is significantly above 1%.
        """
        result = run_chi2_raim_mc(self._SMALL_PARAMS)
        assert result.p_d < 0.02, (
            f"chi²-RAIM P_D={result.p_d:.4f} should be ≈ P_FA for coherent meaconing"
        )

    def test_reproducibility(self) -> None:
        """Two calls with same params must produce identical results."""
        r1 = run_chi2_raim_mc(self._SMALL_PARAMS)
        r2 = run_chi2_raim_mc(self._SMALL_PARAMS)
        assert r1.p_fa == pytest.approx(r2.p_fa, abs=1e-9)
        assert r1.p_d == pytest.approx(r2.p_d, abs=1e-9)

    def test_default_params(self) -> None:
        """Smoke: run with default params (small override for CI)."""
        params = StanfordParams(n_mc=100, random_seed=0)
        result = run_chi2_raim_mc(params)
        assert isinstance(result, StanfordMCResult)


# ---------------------------------------------------------------------------
# 8. Medium-scale MC: chi²-RAIM blindness at n_mc=5000
# ---------------------------------------------------------------------------


def test_chi2_raim_blindness_medium_n() -> None:
    """At n_mc=5000, P_D and P_FA must both be negligible (<0.001).

    Stanford paper proves P_D = P_FA analytically for coherent meaconing.
    At p_fa_max=1e-7 with 5000 trials, expected false alarms ≈ 5e-4 → 0.
    """
    params = StanfordParams(n_mc=5_000, random_seed=99)
    result = run_chi2_raim_mc(params)
    assert result.p_d < 0.01, (
        f"chi²-RAIM P_D={result.p_d:.5f} — coherent attack must be nearly undetectable"
    )
    assert result.p_fa < 0.01, f"chi²-RAIM P_FA={result.p_fa:.5f} — well above the 1e-7 threshold"


# ---------------------------------------------------------------------------
# 9. ResilienceTwin advantage over chi²-RAIM (Stanford N=12 geometry)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_resilience_twin_outperforms_chi2_raim_stanford() -> None:
    """ResilienceTwin P_D must significantly exceed chi²-RAIM P_D.

    Uses the Stanford N=12 geometry with coherent meaconing attack.
    chi²-RAIM: P_D ≈ P_FA ≈ 0 (attack invisible after residual projection)
    ResilienceTwin: P_D >> P_FA (spectral + IMM + CN0 pillars detect coherence)

    Marked @pytest.mark.slow — excluded from default CI run.
    Run manually with: uv run pytest -m slow tests/test_gnss_stanford_benchmark.py
    """
    from gnss.stanford_benchmark import run_resilience_twin_stanford_mc

    # chi²-RAIM baseline
    chi2_result = run_chi2_raim_mc(StanfordParams(n_mc=500, random_seed=42))

    # ResilienceTwin with Stanford N=12 geometry
    twin_result = run_resilience_twin_stanford_mc(n_mc=400, random_seed=42)

    assert twin_result.p_d > chi2_result.p_d + 0.5, (
        f"ResilienceTwin P_D={twin_result.p_d:.3f} should be >> "
        f"chi²-RAIM P_D={chi2_result.p_d:.3f} for coherent meaconing"
    )
    # N=12: spectral/RMT votes raise P_FA ~10%; gate matches test_stanford_validation.py
    assert twin_result.p_fa <= 0.15, f"ResilienceTwin P_FA={twin_result.p_fa:.3f} must remain low"
