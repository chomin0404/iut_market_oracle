"""Tests for src/schemas.py — validation behavior under nominal and edge cases."""

import pytest
from pydantic import ValidationError

from schemas import (
    AssumptionSet,
    ClaimTag,
    EdgeMeta,
    Evidence,
    EvidenceKind,
    ExperimentMeta,
    GraphInput,
    NodeMeta,
    PortfolioMetrics,
    PosteriorSummary,
    PriorSpec,
    ScenarioResult,
)

# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------


class TestEvidence:
    def test_valid_minimal(self):
        e = Evidence(source="bloomberg", kind=EvidenceKind.MARKET_DATA, value=0.05)
        assert e.weight == 1.0
        assert e.tag == ClaimTag.EMPIRICAL

    def test_invalid_empty_source(self):
        with pytest.raises(ValidationError):
            Evidence(source="", kind=EvidenceKind.OBSERVATION, value=0.1)

    def test_invalid_inf_value(self):
        with pytest.raises(ValidationError):
            Evidence(source="x", kind=EvidenceKind.OBSERVATION, value=float("inf"))

    def test_invalid_nan_value(self):
        with pytest.raises(ValidationError):
            Evidence(source="x", kind=EvidenceKind.OBSERVATION, value=float("nan"))

    def test_invalid_zero_weight(self):
        with pytest.raises(ValidationError):
            Evidence(source="x", kind=EvidenceKind.OBSERVATION, value=0.0, weight=0.0)

    def test_negative_value_allowed(self):
        e = Evidence(source="x", kind=EvidenceKind.BACKTEST, value=-0.03)
        assert e.value == pytest.approx(-0.03)


# ---------------------------------------------------------------------------
# PriorSpec
# ---------------------------------------------------------------------------


class TestPriorSpec:
    def test_valid(self):
        p = PriorSpec(distribution="beta", params={"alpha": 2.0, "beta": 5.0})
        assert p.distribution == "beta"

    def test_empty_params_rejected(self):
        with pytest.raises(ValidationError):
            PriorSpec(distribution="normal", params={})


# ---------------------------------------------------------------------------
# PosteriorSummary
# ---------------------------------------------------------------------------


class TestPosteriorSummary:
    def test_valid(self):
        ps = PosteriorSummary(
            mean=0.1,
            variance=0.01,
            credible_interval_95=(0.0, 0.2),
            n_evidence=10,
        )
        assert ps.n_evidence == 10

    def test_negative_variance_rejected(self):
        with pytest.raises(ValidationError):
            PosteriorSummary(
                mean=0.0,
                variance=-0.01,
                credible_interval_95=(0.0, 0.1),
                n_evidence=1,
            )

    def test_inverted_interval_rejected(self):
        with pytest.raises(ValidationError):
            PosteriorSummary(
                mean=0.0,
                variance=0.01,
                credible_interval_95=(0.5, 0.1),  # lo > hi
                n_evidence=1,
            )

    def test_negative_n_evidence_rejected(self):
        with pytest.raises(ValidationError):
            PosteriorSummary(
                mean=0.0,
                variance=0.0,
                credible_interval_95=(0.0, 0.0),
                n_evidence=-1,
            )


# ---------------------------------------------------------------------------
# NodeMeta / EdgeMeta / GraphInput
# ---------------------------------------------------------------------------


class TestNodeMeta:
    def test_valid(self):
        n = NodeMeta(node_id="A", label="Alpha")
        assert n.weight == 1.0

    def test_empty_id_rejected(self):
        with pytest.raises(ValidationError):
            NodeMeta(node_id="")

    def test_zero_weight_rejected(self):
        with pytest.raises(ValidationError):
            NodeMeta(node_id="A", weight=0.0)


class TestEdgeMeta:
    def test_valid(self):
        e = EdgeMeta(source="A", target="B", strength=0.8)
        assert e.strength == pytest.approx(0.8)

    def test_self_loop_rejected(self):
        with pytest.raises(ValidationError):
            EdgeMeta(source="A", target="A")

    def test_negative_strength_rejected(self):
        with pytest.raises(ValidationError):
            EdgeMeta(source="A", target="B", strength=-0.1)


class TestGraphInput:
    def _nodes(self):
        return [NodeMeta(node_id="A"), NodeMeta(node_id="B")]

    def test_valid_no_edges(self):
        g = GraphInput(nodes=self._nodes())
        assert len(g.edges) == 0

    def test_valid_with_edges(self):
        g = GraphInput(nodes=self._nodes(), edges=[EdgeMeta(source="A", target="B")])
        assert len(g.edges) == 1

    def test_empty_node_list_rejected(self):
        with pytest.raises(ValidationError):
            GraphInput(nodes=[])

    def test_edge_missing_source_node_rejected(self):
        with pytest.raises(ValidationError):
            GraphInput(
                nodes=[NodeMeta(node_id="A")],
                edges=[EdgeMeta(source="Z", target="A")],  # Z not in nodes
            )

    def test_edge_missing_target_node_rejected(self):
        with pytest.raises(ValidationError):
            GraphInput(
                nodes=[NodeMeta(node_id="A")],
                edges=[EdgeMeta(source="A", target="Z")],
            )


# ---------------------------------------------------------------------------
# PortfolioMetrics
# ---------------------------------------------------------------------------


class TestPortfolioMetrics:
    def test_valid(self):
        m = PortfolioMetrics(
            basis_diversity=0.7,
            dependency_concentration=1.2,
            portfolio_score=0.65,
            node_count=5,
            edge_count=4,
        )
        assert m.node_count == 5

    def test_diversity_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            PortfolioMetrics(
                basis_diversity=1.5,
                dependency_concentration=0.0,
                portfolio_score=0.5,
                node_count=1,
                edge_count=0,
            )

    def test_zero_node_count_rejected(self):
        with pytest.raises(ValidationError):
            PortfolioMetrics(
                basis_diversity=0.5,
                dependency_concentration=0.0,
                portfolio_score=0.5,
                node_count=0,
                edge_count=0,
            )


# ---------------------------------------------------------------------------
# AssumptionSet
# ---------------------------------------------------------------------------


class TestAssumptionSet:
    def test_valid(self):
        a = AssumptionSet(name="base", params={"discount_rate": 0.08, "growth": 0.03})
        assert a.version == "1.0"

    def test_empty_params_rejected(self):
        with pytest.raises(ValidationError):
            AssumptionSet(name="base", params={})

    def test_empty_name_rejected(self):
        with pytest.raises(ValidationError):
            AssumptionSet(name="", params={"x": 1.0})


# ---------------------------------------------------------------------------
# ScenarioResult
# ---------------------------------------------------------------------------


class TestScenarioResult:
    def test_valid(self):
        r = ScenarioResult(
            scenario_name="bull",
            assumption_version="1.0",
            value=1_500_000.0,
            unit="JPY",
        )
        assert r.sensitivity == {}

    def test_empty_name_rejected(self):
        with pytest.raises(ValidationError):
            ScenarioResult(scenario_name="", assumption_version="1.0", value=0.0)


# ---------------------------------------------------------------------------
# ExperimentMeta
# ---------------------------------------------------------------------------


class TestExperimentMeta:
    def test_valid(self):
        m = ExperimentMeta(
            experiment_id="exp-001",
            title="Baseline valuation",
            config_path="configs/scenarios/base.yaml",
        )
        assert m.tags == []

    def test_invalid_id_format(self):
        with pytest.raises(ValidationError):
            ExperimentMeta(
                experiment_id="001",  # missing "exp-" prefix
                title="x",
                config_path="configs/x.yaml",
            )

    def test_invalid_id_non_numeric(self):
        with pytest.raises(ValidationError):
            ExperimentMeta(
                experiment_id="exp-abc",
                title="x",
                config_path="configs/x.yaml",
            )

    def test_empty_title_rejected(self):
        with pytest.raises(ValidationError):
            ExperimentMeta(
                experiment_id="exp-002",
                title="",
                config_path="configs/x.yaml",
            )
