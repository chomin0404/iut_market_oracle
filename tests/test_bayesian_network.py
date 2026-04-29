"""Tests for src/bayesian/network.py — discrete Bayesian Network."""

from __future__ import annotations

import pytest

from bayesian.network import BayesianNetwork

# ---------------------------------------------------------------------------
# Helpers: canonical small networks
# ---------------------------------------------------------------------------


def _single_node_net() -> BayesianNetwork:
    """P(rain) = [0.3, 0.7]  (no edges)."""
    net = BayesianNetwork()
    net.add_node("rain", states=["yes", "no"])
    net.set_prior("rain", [0.3, 0.7])
    return net


def _two_node_net() -> BayesianNetwork:
    """economy → regime.

    P(economy): expansion=0.70, recession=0.30

    P(regime | economy):
        expansion → bull=0.60, bear=0.10, neutral=0.30
        recession → bull=0.20, bear=0.60, neutral=0.20
    """
    net = BayesianNetwork()
    net.add_node("economy", states=["expansion", "recession"])
    net.add_node("regime", states=["bull", "bear", "neutral"])
    net.add_edge("economy", "regime")
    net.set_prior("economy", [0.70, 0.30])
    net.set_cpt(
        "regime",
        {
            ("expansion",): [0.60, 0.10, 0.30],
            ("recession",): [0.20, 0.60, 0.20],
        },
    )
    return net


def _chain_net() -> BayesianNetwork:
    """economy → regime → return.

    P(economy): expansion=0.70, recession=0.30

    P(regime | economy):
        expansion → bull=0.60, bear=0.40
        recession → bull=0.25, bear=0.75

    P(return | regime):
        bull → high=0.80, low=0.20
        bear → high=0.20, low=0.80
    """
    net = BayesianNetwork()
    net.add_node("economy", states=["expansion", "recession"])
    net.add_node("regime", states=["bull", "bear"])
    net.add_node("return", states=["high", "low"])
    net.add_edge("economy", "regime")
    net.add_edge("regime", "return")
    net.set_prior("economy", [0.70, 0.30])
    net.set_cpt(
        "regime",
        {
            ("expansion",): [0.60, 0.40],
            ("recession",): [0.25, 0.75],
        },
    )
    net.set_cpt(
        "return",
        {
            ("bull",): [0.80, 0.20],
            ("bear",): [0.20, 0.80],
        },
    )
    return net


def _diamond_net() -> BayesianNetwork:
    """Classic diamond: A → B, A → C, B → D, C → D.

    P(A): t=0.6, f=0.4

    P(B | A): t|t=0.9, f|t=0.1 ; t|f=0.2, f|f=0.8
    P(C | A): t|t=0.7, f|t=0.3 ; t|f=0.3, f|f=0.7

    P(D | B, C):
        (t,t)→ t=0.95, f=0.05
        (t,f)→ t=0.70, f=0.30
        (f,t)→ t=0.65, f=0.35
        (f,f)→ t=0.10, f=0.90
    """
    net = BayesianNetwork()
    net.add_node("A", states=["t", "f"])
    net.add_node("B", states=["t", "f"])
    net.add_node("C", states=["t", "f"])
    net.add_node("D", states=["t", "f"])
    net.add_edge("A", "B")
    net.add_edge("A", "C")
    net.add_edge("B", "D")
    net.add_edge("C", "D")
    net.set_prior("A", [0.6, 0.4])
    net.set_cpt("B", {("t",): [0.9, 0.1], ("f",): [0.2, 0.8]})
    net.set_cpt("C", {("t",): [0.7, 0.3], ("f",): [0.3, 0.7]})
    net.set_cpt(
        "D",
        {
            ("t", "t"): [0.95, 0.05],
            ("t", "f"): [0.70, 0.30],
            ("f", "t"): [0.65, 0.35],
            ("f", "f"): [0.10, 0.90],
        },
    )
    return net


# ---------------------------------------------------------------------------
# Structure and CPT validation
# ---------------------------------------------------------------------------


class TestStructure:
    def test_add_node_duplicate_raises(self):
        net = BayesianNetwork()
        net.add_node("x", states=["a", "b"])
        with pytest.raises(ValueError, match="already exists"):
            net.add_node("x", states=["a", "b"])

    def test_add_node_too_few_states(self):
        net = BayesianNetwork()
        with pytest.raises(ValueError, match="at least 2"):
            net.add_node("x", states=["only_one"])

    def test_add_node_duplicate_states(self):
        net = BayesianNetwork()
        with pytest.raises(ValueError, match="duplicate"):
            net.add_node("x", states=["a", "a"])

    def test_add_edge_unknown_node(self):
        net = BayesianNetwork()
        net.add_node("a", states=["t", "f"])
        with pytest.raises(ValueError, match="not found"):
            net.add_edge("a", "b")

    def test_add_edge_self_loop(self):
        net = BayesianNetwork()
        net.add_node("a", states=["t", "f"])
        with pytest.raises(ValueError, match="Self-loop"):
            net.add_edge("a", "a")

    def test_add_edge_cycle_detected(self):
        net = BayesianNetwork()
        net.add_node("a", states=["t", "f"])
        net.add_node("b", states=["t", "f"])
        net.add_edge("a", "b")
        with pytest.raises(ValueError, match="Cycle"):
            net.add_edge("b", "a")

    def test_add_edge_duplicate(self):
        net = BayesianNetwork()
        net.add_node("a", states=["t", "f"])
        net.add_node("b", states=["t", "f"])
        net.add_edge("a", "b")
        with pytest.raises(ValueError, match="already exists"):
            net.add_edge("a", "b")

    def test_topological_order_chain(self):
        net = _chain_net()
        order = net.topological_order()
        assert order.index("economy") < order.index("regime")
        assert order.index("regime") < order.index("return")


class TestCPTValidation:
    def test_set_prior_for_node_with_parents_raises(self):
        net = _two_node_net()
        with pytest.raises(ValueError, match="use set_cpt"):
            net.set_prior("regime", [0.4, 0.3, 0.3])

    def test_set_cpt_for_root_raises(self):
        net = BayesianNetwork()
        net.add_node("a", states=["t", "f"])
        with pytest.raises(ValueError, match="use set_prior"):
            net.set_cpt("a", {})

    def test_prior_wrong_length(self):
        net = BayesianNetwork()
        net.add_node("a", states=["t", "f"])
        with pytest.raises(ValueError, match="expected 2"):
            net.set_prior("a", [0.3, 0.3, 0.4])

    def test_prior_not_summing_to_one(self):
        net = BayesianNetwork()
        net.add_node("a", states=["t", "f"])
        with pytest.raises(ValueError, match="sum to 1"):
            net.set_prior("a", [0.4, 0.4])

    def test_prior_negative_values(self):
        net = BayesianNetwork()
        net.add_node("a", states=["t", "f"])
        with pytest.raises(ValueError, match="non-negative"):
            net.set_prior("a", [-0.1, 1.1])

    def test_cpt_missing_rows_raises(self):
        net = BayesianNetwork()
        net.add_node("a", states=["t", "f"])
        net.add_node("b", states=["x", "y"])
        net.add_edge("a", "b")
        # Only 1 of 2 required rows provided
        with pytest.raises(ValueError, match="2 parent-state combinations"):
            net.set_cpt("b", {("t",): [0.8, 0.2]})

    def test_cpt_row_not_summing_to_one(self):
        net = BayesianNetwork()
        net.add_node("a", states=["t", "f"])
        net.add_node("b", states=["x", "y"])
        net.add_edge("a", "b")
        with pytest.raises(ValueError, match="sum to 1"):
            net.set_cpt("b", {("t",): [0.3, 0.4], ("f",): [0.5, 0.5]})

    def test_cpt_unknown_parent_state(self):
        net = BayesianNetwork()
        net.add_node("a", states=["t", "f"])
        net.add_node("b", states=["x", "y"])
        net.add_edge("a", "b")
        with pytest.raises(ValueError, match="Unknown parent state"):
            net.set_cpt("b", {("INVALID",): [0.5, 0.5], ("f",): [0.5, 0.5]})

    def test_inference_without_cpt_raises(self):
        net = BayesianNetwork()
        net.add_node("a", states=["t", "f"])
        net.add_node("b", states=["x", "y"])
        net.add_edge("a", "b")
        net.set_prior("a", [0.5, 0.5])
        # CPT for b is not set
        with pytest.raises(ValueError, match="CPT not set"):
            net.posterior("b")


# ---------------------------------------------------------------------------
# Evidence validation
# ---------------------------------------------------------------------------


class TestEvidence:
    def test_observe_unknown_node(self):
        net = _two_node_net()
        with pytest.raises(ValueError, match="not found"):
            net.observe("gdp", "high")

    def test_observe_invalid_state(self):
        net = _two_node_net()
        with pytest.raises(ValueError, match="not valid"):
            net.observe("economy", "stagflation")

    def test_reset_evidence(self):
        net = _two_node_net()
        net.observe("economy", "recession")
        net.reset_evidence()
        post = net.posterior("regime")
        # Should equal prior-weighted marginal
        expected_bull = 0.70 * 0.60 + 0.30 * 0.20
        assert post["bull"] == pytest.approx(expected_bull / sum(post.values()), abs=1e-9)


# ---------------------------------------------------------------------------
# Single-node inference (no edges)
# ---------------------------------------------------------------------------


class TestSingleNode:
    def test_prior_equals_posterior_without_evidence(self):
        net = _single_node_net()
        post = net.posterior("rain")
        assert post["yes"] == pytest.approx(0.3, abs=1e-9)
        assert post["no"] == pytest.approx(0.7, abs=1e-9)

    def test_observed_query_returns_degenerate(self):
        net = _single_node_net()
        net.observe("rain", "yes")
        post = net.posterior("rain")
        assert post["yes"] == pytest.approx(1.0, abs=1e-9)
        assert post["no"] == pytest.approx(0.0, abs=1e-9)

    def test_posterior_sums_to_one(self):
        net = _single_node_net()
        post = net.posterior("rain")
        assert sum(post.values()) == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Two-node network
# ---------------------------------------------------------------------------


class TestTwoNode:
    def test_prior_marginal_of_regime(self):
        """P(regime) = Σ_e P(regime|e) P(e)."""
        net = _two_node_net()
        post = net.posterior("regime")

        expected_bull = 0.70 * 0.60 + 0.30 * 0.20  # 0.48
        expected_bear = 0.70 * 0.10 + 0.30 * 0.60  # 0.25
        expected_neutral = 0.70 * 0.30 + 0.30 * 0.20  # 0.27

        assert post["bull"] == pytest.approx(expected_bull, abs=1e-9)
        assert post["bear"] == pytest.approx(expected_bear, abs=1e-9)
        assert post["neutral"] == pytest.approx(expected_neutral, abs=1e-9)

    def test_posterior_sums_to_one(self):
        net = _two_node_net()
        assert sum(net.posterior("regime").values()) == pytest.approx(1.0, abs=1e-9)

    def test_evidence_expansion_exact(self):
        """P(regime | economy=expansion) must equal the CPT row directly."""
        net = _two_node_net()
        net.observe("economy", "expansion")
        post = net.posterior("regime")
        assert post["bull"] == pytest.approx(0.60, abs=1e-9)
        assert post["bear"] == pytest.approx(0.10, abs=1e-9)
        assert post["neutral"] == pytest.approx(0.30, abs=1e-9)

    def test_evidence_recession_exact(self):
        net = _two_node_net()
        net.observe("economy", "recession")
        post = net.posterior("regime")
        assert post["bull"] == pytest.approx(0.20, abs=1e-9)
        assert post["bear"] == pytest.approx(0.60, abs=1e-9)
        assert post["neutral"] == pytest.approx(0.20, abs=1e-9)

    def test_bayes_theorem_reverse_query(self):
        """P(economy=expansion | regime=bull) via Bayes' theorem.

        P(E=exp | R=bull) = P(R=bull | E=exp) P(E=exp) / P(R=bull)
                          = 0.60 * 0.70 / (0.60*0.70 + 0.20*0.30)
                          = 0.42 / 0.48
                          = 0.875
        """
        net = _two_node_net()
        net.observe("regime", "bull")
        post = net.posterior("economy")
        expected = (0.60 * 0.70) / (0.60 * 0.70 + 0.20 * 0.30)
        assert post["expansion"] == pytest.approx(expected, abs=1e-9)

    def test_update_method_preserves_existing_evidence(self):
        net = _two_node_net()
        net.observe("economy", "recession")
        # update() with different evidence should not change the stored evidence
        _ = net.update({"economy": "expansion"}, ["regime"])
        # After update(), original evidence should be restored
        post_after = net.posterior("regime")
        assert post_after["bull"] == pytest.approx(0.20, abs=1e-9)


# ---------------------------------------------------------------------------
# Chain network (3 nodes)
# ---------------------------------------------------------------------------


class TestChain:
    def test_marginal_return_no_evidence(self):
        """P(return=high) = Σ_{e,r} P(return=high|r) P(r|e) P(e).

        P(regime=bull) = 0.70*0.60 + 0.30*0.25 = 0.42 + 0.075 = 0.495
        P(regime=bear) = 0.70*0.40 + 0.30*0.75 = 0.28 + 0.225 = 0.505
        P(return=high) = 0.495*0.80 + 0.505*0.20 = 0.396 + 0.101 = 0.497
        """
        net = _chain_net()
        post = net.posterior("return")
        expected = 0.495 * 0.80 + 0.505 * 0.20
        assert post["high"] == pytest.approx(expected, abs=1e-9)

    def test_evidence_economy_expansion(self):
        """P(return=high | economy=expansion) = 0.60*0.80 + 0.40*0.20 = 0.56."""
        net = _chain_net()
        net.observe("economy", "expansion")
        post = net.posterior("return")
        expected = 0.60 * 0.80 + 0.40 * 0.20
        assert post["high"] == pytest.approx(expected, abs=1e-9)

    def test_evidence_regime_bull_blocks_economy(self):
        """Regime=bull observed → economy becomes d-separated from return.

        P(return=high | regime=bull) = 0.80  (direct CPT lookup).
        """
        net = _chain_net()
        net.observe("regime", "bull")
        post = net.posterior("return")
        assert post["high"] == pytest.approx(0.80, abs=1e-9)

    def test_posterior_sums_to_one(self):
        net = _chain_net()
        net.observe("economy", "expansion")
        post = net.posterior("return")
        assert sum(post.values()) == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Diamond network (4 nodes, non-singly-connected)
# ---------------------------------------------------------------------------


class TestDiamond:
    def test_no_evidence_sums_to_one(self):
        net = _diamond_net()
        for q in ["A", "B", "C", "D"]:
            post = net.posterior(q)
            assert sum(post.values()) == pytest.approx(1.0, abs=1e-9)

    def test_prior_marginal_a(self):
        net = _diamond_net()
        post = net.posterior("A")
        assert post["t"] == pytest.approx(0.6, abs=1e-9)

    def test_d_given_a_true_manual(self):
        """P(D=t | A=t) — computed by hand.

        P(B=t | A=t) = 0.9   P(B=f | A=t) = 0.1
        P(C=t | A=t) = 0.7   P(C=f | A=t) = 0.3

        P(D=t | A=t) = Σ_{b,c} P(D=t|b,c) P(B=b|A=t) P(C=c|A=t)
          = P(D|t,t)*P(B=t)*P(C=t) + P(D|t,f)*P(B=t)*P(C=f)
          + P(D|f,t)*P(B=f)*P(C=t) + P(D|f,f)*P(B=f)*P(C=f)
          = 0.95*0.9*0.7 + 0.70*0.9*0.3 + 0.65*0.1*0.7 + 0.10*0.1*0.3
          = 0.5985 + 0.189 + 0.0455 + 0.003
          = 0.836
        """
        net = _diamond_net()
        net.observe("A", "t")
        post = net.posterior("D")
        expected = 0.95 * 0.9 * 0.7 + 0.70 * 0.9 * 0.3 + 0.65 * 0.1 * 0.7 + 0.10 * 0.1 * 0.3
        assert post["t"] == pytest.approx(expected, abs=1e-9)

    def test_sums_to_one_with_evidence(self):
        net = _diamond_net()
        net.observe("A", "t")
        for q in ["B", "C", "D"]:
            post = net.posterior(q)
            assert sum(post.values()) == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# update() convenience wrapper
# ---------------------------------------------------------------------------


class TestUpdateMethod:
    def test_returns_all_queried_nodes(self):
        net = _chain_net()
        result = net.update(
            evidence={"economy": "expansion"},
            queries=["regime", "return"],
        )
        assert set(result.keys()) == {"regime", "return"}

    def test_values_match_observe_then_posterior(self):
        net = _chain_net()
        batch = net.update({"economy": "expansion"}, ["return"])

        net.observe("economy", "expansion")
        manual = net.posterior("return")

        assert batch["return"]["high"] == pytest.approx(manual["high"], abs=1e-12)

    def test_does_not_mutate_stored_evidence(self):
        net = _chain_net()
        # Establish baseline without evidence
        baseline = net.posterior("return")
        net.update({"economy": "expansion"}, ["return"])
        after_update = net.posterior("return")
        assert after_update["high"] == pytest.approx(baseline["high"], abs=1e-12)
