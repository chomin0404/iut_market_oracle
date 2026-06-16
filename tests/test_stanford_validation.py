"""Cross-validation of ResilienceTwin against Stanford GPS Lab chi-squared RAIM baseline.

Source:  https://github.com/stanford-gps-lab/spoofing-detection  (MATLAB-only, no data files)
Paper:   Rothmaier et al., "GNSS Spoofing Detection through Metric Combinations",
         ION GNSS+ 2021.

Scope: chi-squared RAIM vs ResilienceTwin MC comparison only.
Unit tests for individual benchmark functions are in test_gnss_stanford_benchmark.py.

Acceptance criteria:
  1. chi2_raim P_D ≈ P_FA for coherent vertical bias (fundamental RAIM limitation).
  2. ResilienceTwin achieves P_D ≥ 0.85 with P_FA ≤ 0.15 in Stanford N=12 geometry.
  3. ResilienceTwin P_D strictly exceeds chi2_raim P_D on the same geometry.
"""

from __future__ import annotations

import pytest

from gnss.stanford_benchmark import (
    StanfordParams,
    run_chi2_raim_mc,
    run_resilience_twin_stanford_mc,
)

# ---------------------------------------------------------------------------
# Monte Carlo comparison (fast, reduced n_mc for CI speed)
# ---------------------------------------------------------------------------


class TestStanfordMCComparison:
    """Compare chi-squared RAIM baseline vs ResilienceTwin on Stanford geometry."""

    @pytest.fixture(scope="class")
    def chi2_result(self):
        """Run chi-squared RAIM MC once per test class (fast)."""
        params = StanfordParams(n_sats=12, sig_pr=3.0, p_fa_max=0.05, n_mc=5_000, random_seed=42)
        return run_chi2_raim_mc(params)

    @pytest.fixture(scope="class")
    def rt_result(self):
        """Run ResilienceTwin MC once per test class (heavier, n_mc=400)."""
        return run_resilience_twin_stanford_mc(n_mc=400, random_seed=42)

    def test_chi2_raim_pd_near_pfa(self, chi2_result) -> None:
        """chi-squared RAIM: P_D ≈ P_FA for coherent meaconing attack."""
        # |P_D - P_FA| < 0.05 confirms the attack is invisible to RAIM
        assert abs(chi2_result.p_d - chi2_result.p_fa) < 0.05, (
            f"chi2_raim P_D={chi2_result.p_d:.3f} should ≈ P_FA={chi2_result.p_fa:.3f} "
            f"for coherent meaconing"
        )

    def test_resilience_twin_spoofing_detection(self, rt_result) -> None:
        """ResilienceTwin: P_D ≥ 0.85 for spoofing in Stanford N=12 geometry."""
        assert rt_result.p_d >= 0.85, (
            f"ResilienceTwin P_D={rt_result.p_d:.3f} below 0.85 in N=12 geometry"
        )

    def test_resilience_twin_false_alarm(self, rt_result) -> None:
        """ResilienceTwin: P_FA ≤ 0.15 for nominal trials in Stanford N=12 geometry.

        Note: The vote-threshold _SPOOF_VOTE_THRESH was calibrated for N=6 sats.
        With N=12 sats the spectral/RMT pillars produce more background votes,
        raising P_FA to ~10%.  P_FA < 0.15 is the acceptance gate for N=12.
        """
        assert rt_result.p_fa <= 0.15, f"ResilienceTwin P_FA={rt_result.p_fa:.3f} exceeds 0.15"

    def test_resilience_twin_outperforms_chi2_raim(self, chi2_result, rt_result) -> None:
        """ResilienceTwin P_D must strictly exceed chi-squared RAIM P_D.

        This reproduces the core result of Stanford's paper: single-metric
        chi-squared RAIM cannot detect coherent meaconing; multi-metric
        detection is required.
        """
        assert rt_result.p_d > chi2_result.p_d + 0.50, (
            f"ResilienceTwin P_D={rt_result.p_d:.3f} should be ≫ "
            f"chi2_raim P_D={chi2_result.p_d:.3f}"
        )
