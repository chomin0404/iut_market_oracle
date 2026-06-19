"""Tests for ROICBayesNet — structure, inference, and Dirichlet updates."""

from __future__ import annotations

import pytest

from bayesian.roic_net import ROICBayesNet, _build_roic_cpt, _roic_cpt_row
from schemas.roic import ROICEvidence, ROICObservation

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def net() -> ROICBayesNet:
    return ROICBayesNet(equivalent_sample_size=10.0)


# ---------------------------------------------------------------------------
# Network structure and prior marginal
# ---------------------------------------------------------------------------


class TestNetworkStructure:
    def test_prior_marginal_is_valid_distribution(self, net: ROICBayesNet) -> None:
        result = net.evaluate()
        total = sum(result.roic_posterior.values())
        assert abs(total - 1.0) < 1e-6

    def test_prior_marginal_has_all_three_states(self, net: ROICBayesNet) -> None:
        result = net.evaluate()
        assert set(result.roic_posterior.keys()) == {"high", "medium", "low"}

    def test_all_posterior_values_non_negative(self, net: ROICBayesNet) -> None:
        result = net.evaluate()
        for p in result.roic_posterior.values():
            assert p >= 0.0

    def test_expected_score_in_unit_interval(self, net: ROICBayesNet) -> None:
        result = net.evaluate()
        assert 0.0 <= result.expected_roic_score <= 1.0

    def test_dominant_state_is_argmax(self, net: ROICBayesNet) -> None:
        result = net.evaluate()
        best = max(result.roic_posterior, key=lambda s: result.roic_posterior[s])
        assert result.dominant_state == best

    def test_empty_evidence_gives_same_as_none(self, net: ROICBayesNet) -> None:
        r_none = net.evaluate(None)
        r_empty = net.evaluate(ROICEvidence())
        for state in ("high", "medium", "low"):
            assert abs(r_none.roic_posterior[state] - r_empty.roic_posterior[state]) < 1e-9


# ---------------------------------------------------------------------------
# Directional inference — root nodes
# ---------------------------------------------------------------------------


class TestDirectionalInferenceRootNodes:
    def test_high_industry_raises_roic_high(self, net: ROICBayesNet) -> None:
        """Strong industry competitiveness → higher P(roic=high)."""
        hi = net.evaluate(ROICEvidence(industry_competitiveness="high"))
        lo = net.evaluate(ROICEvidence(industry_competitiveness="low"))
        assert hi.roic_posterior["high"] > lo.roic_posterior["high"]
        assert hi.roic_posterior["low"] < lo.roic_posterior["low"]

    def test_high_management_raises_roic_high(self, net: ROICBayesNet) -> None:
        hi = net.evaluate(ROICEvidence(management_efficiency="high"))
        lo = net.evaluate(ROICEvidence(management_efficiency="low"))
        assert hi.roic_posterior["high"] > lo.roic_posterior["high"]

    def test_favorable_macro_raises_roic_high(self, net: ROICBayesNet) -> None:
        hi = net.evaluate(ROICEvidence(macro_environment="high"))
        lo = net.evaluate(ROICEvidence(macro_environment="low"))
        assert hi.roic_posterior["high"] > lo.roic_posterior["high"]

    def test_expected_score_order_high_gt_medium_gt_low(self, net: ROICBayesNet) -> None:
        """Expected score must decrease monotonically across root-node states."""
        s_hi = net.evaluate(
            ROICEvidence(
                industry_competitiveness="high",
                management_efficiency="high",
                macro_environment="high",
            )
        ).expected_roic_score
        s_md = net.evaluate(
            ROICEvidence(
                industry_competitiveness="medium",
                management_efficiency="medium",
                macro_environment="medium",
            )
        ).expected_roic_score
        s_lo = net.evaluate(
            ROICEvidence(
                industry_competitiveness="low",
                management_efficiency="low",
                macro_environment="low",
            )
        ).expected_roic_score
        assert s_hi > s_md > s_lo


# ---------------------------------------------------------------------------
# Extreme scenarios
# ---------------------------------------------------------------------------


class TestExtremeScenarios:
    def test_all_high_dominant_state_is_high(self, net: ROICBayesNet) -> None:
        result = net.evaluate(
            ROICEvidence(
                industry_competitiveness="high",
                management_efficiency="high",
                macro_environment="high",
            )
        )
        assert result.dominant_state == "high"

    def test_all_low_dominant_state_is_low(self, net: ROICBayesNet) -> None:
        result = net.evaluate(
            ROICEvidence(
                industry_competitiveness="low",
                management_efficiency="low",
                macro_environment="low",
            )
        )
        assert result.dominant_state == "low"

    def test_all_high_expected_score_above_threshold(self, net: ROICBayesNet) -> None:
        result = net.evaluate(
            ROICEvidence(
                industry_competitiveness="high",
                management_efficiency="high",
                macro_environment="high",
            )
        )
        assert result.expected_roic_score >= 0.65

    def test_all_low_expected_score_below_threshold(self, net: ROICBayesNet) -> None:
        result = net.evaluate(
            ROICEvidence(
                industry_competitiveness="low",
                management_efficiency="low",
                macro_environment="low",
            )
        )
        assert result.expected_roic_score <= 0.35


# ---------------------------------------------------------------------------
# Direct evidence on intermediate nodes
# ---------------------------------------------------------------------------


class TestIntermediateNodeEvidence:
    def test_pricing_power_high_raises_roic(self, net: ROICBayesNet) -> None:
        hi = net.evaluate(ROICEvidence(pricing_power="high"))
        lo = net.evaluate(ROICEvidence(pricing_power="low"))
        assert hi.roic_posterior["high"] > lo.roic_posterior["high"]
        assert hi.roic_posterior["low"] < lo.roic_posterior["low"]

    def test_capital_allocation_high_raises_roic(self, net: ROICBayesNet) -> None:
        hi = net.evaluate(ROICEvidence(capital_allocation="high"))
        lo = net.evaluate(ROICEvidence(capital_allocation="low"))
        assert hi.roic_posterior["high"] > lo.roic_posterior["high"]

    def test_combined_intermediate_evidence(self, net: ROICBayesNet) -> None:
        """Pricing=high + capital=high should give higher score than pricing=low + capital=low."""
        strong = net.evaluate(ROICEvidence(pricing_power="high", capital_allocation="high"))
        weak = net.evaluate(ROICEvidence(pricing_power="low", capital_allocation="low"))
        assert strong.expected_roic_score > weak.expected_roic_score


# ---------------------------------------------------------------------------
# CPT builder unit tests
# ---------------------------------------------------------------------------


class TestROICCPTBuilder:
    def test_cpt_has_27_rows(self) -> None:
        cpt = _build_roic_cpt()
        assert len(cpt) == 27  # 3 × 3 × 3

    def test_all_cpt_rows_sum_to_one(self) -> None:
        cpt = _build_roic_cpt()
        for key, probs in cpt.items():
            total = sum(probs)
            assert abs(total - 1.0) < 1e-9, f"CPT row {key} sums to {total}"

    def test_all_cpt_rows_non_negative(self) -> None:
        cpt = _build_roic_cpt()
        for probs in cpt.values():
            assert all(p >= 0.0 for p in probs)

    def test_best_combination_gives_highest_high_prob(self) -> None:
        row_best = _roic_cpt_row("high", "high", "high")
        row_worst = _roic_cpt_row("low", "low", "low")
        assert row_best[0] > row_worst[0]  # P(high) best > P(high) worst
        assert row_best[2] < row_worst[2]  # P(low) best < P(low) worst

    def test_score_band_boundaries(self) -> None:
        # score=1.0 (h,h,h) → top band [0.75, 0.20, 0.05]
        assert _roic_cpt_row("high", "high", "high") == [0.75, 0.20, 0.05]
        # score=0.0 (l,l,l) → bottom band [0.05, 0.25, 0.70]
        assert _roic_cpt_row("low", "low", "low") == [0.05, 0.25, 0.70]


# ---------------------------------------------------------------------------
# Dirichlet online update
# ---------------------------------------------------------------------------


class TestDirichletUpdate:
    def test_initial_n_observations_is_zero(self, net: ROICBayesNet) -> None:
        assert net.n_observations == 0

    def test_update_increments_n_observations(self, net: ROICBayesNet) -> None:
        obs = [
            ROICObservation(industry_competitiveness="high", roic_level="high"),
            ROICObservation(management_efficiency="low", roic_level="low"),
        ]
        net.update_from_observations(obs)
        assert net.n_observations == 2

    def test_multiple_updates_accumulate_count(self, net: ROICBayesNet) -> None:
        net.update_from_observations([ROICObservation(roic_level="high")] * 5)
        net.update_from_observations([ROICObservation(roic_level="low")] * 3)
        assert net.n_observations == 8

    def test_empty_update_is_noop(self, net: ROICBayesNet) -> None:
        before = net.evaluate().roic_posterior.copy()
        net.update_from_observations([])
        after = net.evaluate().roic_posterior
        assert net.n_observations == 0
        for state in ("high", "medium", "low"):
            assert abs(before[state] - after[state]) < 1e-9

    def test_many_high_observations_shift_posterior_up(self, net: ROICBayesNet) -> None:
        """50 all-high observations must raise P(roic=high) above baseline."""
        baseline = net.evaluate().roic_posterior["high"]
        obs = [
            ROICObservation(
                industry_competitiveness="high",
                management_efficiency="high",
                macro_environment="high",
                pricing_power="high",
                capital_allocation="high",
                roic_level="high",
            )
        ] * 50
        net.update_from_observations(obs)
        updated = net.evaluate().roic_posterior["high"]
        assert updated > baseline

    def test_many_low_observations_shift_posterior_down(self, net: ROICBayesNet) -> None:
        """50 all-low observations must raise P(roic=low) above baseline."""
        baseline = net.evaluate().roic_posterior["low"]
        obs = [
            ROICObservation(
                industry_competitiveness="low",
                management_efficiency="low",
                macro_environment="low",
                pricing_power="low",
                capital_allocation="low",
                roic_level="low",
            )
        ] * 50
        net.update_from_observations(obs)
        updated = net.evaluate().roic_posterior["low"]
        assert updated > baseline

    def test_partial_observations_do_not_raise(self, net: ROICBayesNet) -> None:
        """Observations with only some fields set must not raise."""
        obs = [
            ROICObservation(roic_level="medium"),
            ROICObservation(macro_environment="low"),
            ROICObservation(industry_competitiveness="high", roic_level="high"),
        ]
        net.update_from_observations(obs)  # must not raise
        assert net.n_observations == 3

    def test_posterior_remains_valid_after_update(self, net: ROICBayesNet) -> None:
        """After Dirichlet update, posterior must still be a valid distribution."""
        net.update_from_observations([ROICObservation(roic_level="high")] * 20)
        result = net.evaluate()
        total = sum(result.roic_posterior.values())
        assert abs(total - 1.0) < 1e-6
        assert all(p >= 0.0 for p in result.roic_posterior.values())


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


class TestConstructorValidation:
    def test_invalid_ess_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="equivalent_sample_size"):
            ROICBayesNet(equivalent_sample_size=0.0)

    def test_invalid_ess_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="equivalent_sample_size"):
            ROICBayesNet(equivalent_sample_size=-5.0)

    def test_ess_property(self) -> None:
        net = ROICBayesNet(equivalent_sample_size=25.0)
        assert net.equivalent_sample_size == 25.0

    def test_large_ess_resists_update(self) -> None:
        """High ESS → prior marginal changes less than low ESS after same observations.

        Root node observations (no parent dependency) propagate through the
        pipeline to roic_level.  With ESS=1000 the root CPT barely moves;
        with ESS=5 it shifts substantially, producing a larger delta in
        P(roic=high).
        """
        net_rigid = ROICBayesNet(equivalent_sample_size=1000.0)
        net_flex = ROICBayesNet(equivalent_sample_size=5.0)

        # Root node observations: no parent state required, always counted.
        obs = [ROICObservation(industry_competitiveness="high")] * 10

        baseline_rigid = net_rigid.evaluate().roic_posterior["high"]
        baseline_flex = net_flex.evaluate().roic_posterior["high"]

        net_rigid.update_from_observations(obs)
        net_flex.update_from_observations(obs)

        delta_rigid = abs(net_rigid.evaluate().roic_posterior["high"] - baseline_rigid)
        delta_flex = abs(net_flex.evaluate().roic_posterior["high"] - baseline_flex)

        assert delta_rigid < delta_flex
