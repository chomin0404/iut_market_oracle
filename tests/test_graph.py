"""Tests for src/graph/builder.py and src/graph/metrics.py.

All metrics are deterministic; expected values are derived analytically
in docstrings next to each test.
"""

from __future__ import annotations

import math

import pytest

from graph.builder import (
    adjacency_list,
    category_weight_map,
    weighted_in_degree,
    weighted_out_degree,
)
from graph.metrics import (
    basis_diversity,
    compute_all,
    dependency_concentration,
    portfolio_score,
)
from schemas import EdgeMeta, GraphInput, NodeMeta

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def node(node_id: str, category: str = "", weight: float = 1.0) -> NodeMeta:
    return NodeMeta(node_id=node_id, category=category, weight=weight)


def edge(src: str, tgt: str, strength: float = 1.0) -> EdgeMeta:
    return EdgeMeta(source=src, target=tgt, strength=strength)


def graph(*nodes: NodeMeta, edges: list[EdgeMeta] | None = None) -> GraphInput:
    return GraphInput(nodes=list(nodes), edges=edges or [])


# Canonical three-node toy graph used across multiple tests:
#
#   A (tech,  w=1) ──1.0──► B (finance, w=1)
#   A (tech,  w=1) ──1.0──► C (tech,    w=1)
#   B (finance,w=1)──1.0──► C (tech,    w=1)
#
# In-degrees:  A=0, B=1, C=2  → total=3
# s_A=0, s_B=1/3, s_C=2/3
# HHI = 0² + (1/3)² + (2/3)² = 0 + 1/9 + 4/9 = 5/9 ≈ 0.5556
#
# Categories: tech={A,C} → w=2, finance={B} → w=1  → total=3
# p_tech=2/3, p_fin=1/3
# H = -(2/3·log2(2/3) + 1/3·log2(1/3)) ≈ 0.9183 bits
# basis_diversity = H / log2(2) = H ≈ 0.9183

_A = node("A", category="tech")
_B = node("B", category="finance")
_C = node("C", category="tech")
_AB = edge("A", "B")
_AC = edge("A", "C")
_BC = edge("B", "C")
TRIANGLE = graph(_A, _B, _C, edges=[_AB, _AC, _BC])


# ---------------------------------------------------------------------------
# builder: adjacency_list
# ---------------------------------------------------------------------------


class TestAdjacencyList:
    def test_isolate_has_empty_adjacency(self):
        g = graph(node("X"))
        adj = adjacency_list(g)
        assert adj == {"X": []}

    def test_single_edge_recorded(self):
        g = graph(node("A"), node("B"), edges=[edge("A", "B", strength=0.5)])
        adj = adjacency_list(g)
        assert adj["A"] == [("B", 0.5)]
        assert adj["B"] == []

    def test_triangle_adjacency(self):
        adj = adjacency_list(TRIANGLE)
        assert set(adj["A"]) == {("B", 1.0), ("C", 1.0)}
        assert adj["B"] == [("C", 1.0)]
        assert adj["C"] == []

    def test_all_nodes_present_as_keys(self):
        adj = adjacency_list(TRIANGLE)
        assert set(adj.keys()) == {"A", "B", "C"}


# ---------------------------------------------------------------------------
# builder: weighted_in_degree / weighted_out_degree
# ---------------------------------------------------------------------------


class TestWeightedDegrees:
    def test_no_edges_all_zero(self):
        g = graph(node("X"), node("Y"))
        assert weighted_in_degree(g) == {"X": 0.0, "Y": 0.0}
        assert weighted_out_degree(g) == {"X": 0.0, "Y": 0.0}

    def test_triangle_in_degree(self):
        indeg = weighted_in_degree(TRIANGLE)
        assert indeg["A"] == pytest.approx(0.0)
        assert indeg["B"] == pytest.approx(1.0)
        assert indeg["C"] == pytest.approx(2.0)

    def test_triangle_out_degree(self):
        outdeg = weighted_out_degree(TRIANGLE)
        assert outdeg["A"] == pytest.approx(2.0)
        assert outdeg["B"] == pytest.approx(1.0)
        assert outdeg["C"] == pytest.approx(0.0)

    def test_strength_accumulates(self):
        g = graph(
            node("A"),
            node("B"),
            edges=[edge("A", "B", 0.4), edge("A", "B", 0.6)],
        )
        assert weighted_in_degree(g)["B"] == pytest.approx(1.0)

    def test_weighted_edge_strength(self):
        g = graph(node("A"), node("B"), edges=[edge("A", "B", strength=3.5)])
        assert weighted_in_degree(g)["B"] == pytest.approx(3.5)
        assert weighted_out_degree(g)["A"] == pytest.approx(3.5)


# ---------------------------------------------------------------------------
# builder: category_weight_map
# ---------------------------------------------------------------------------


class TestCategoryWeightMap:
    def test_single_category(self):
        g = graph(node("A", category="tech"), node("B", category="tech"))
        cw = category_weight_map(g)
        assert set(cw.keys()) == {"tech"}
        assert cw["tech"] == pytest.approx(2.0)

    def test_empty_category_becomes_uncategorized(self):
        g = graph(node("A", category=""), node("B", category=""))
        cw = category_weight_map(g)
        assert "uncategorized" in cw

    def test_triangle_categories(self):
        cw = category_weight_map(TRIANGLE)
        assert cw["tech"] == pytest.approx(2.0)
        assert cw["finance"] == pytest.approx(1.0)

    def test_node_weight_reflected(self):
        g = graph(
            node("A", category="X", weight=3.0),
            node("B", category="X", weight=2.0),
            node("C", category="Y", weight=1.0),
        )
        cw = category_weight_map(g)
        assert cw["X"] == pytest.approx(5.0)
        assert cw["Y"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# metrics: basis_diversity
# ---------------------------------------------------------------------------


class TestBasisDiversity:
    def test_single_category_zero_diversity(self):
        g = graph(node("A", category="tech"), node("B", category="tech"))
        assert basis_diversity(g) == pytest.approx(0.0)

    def test_no_category_zero_diversity(self):
        g = graph(node("A"), node("B"))  # both uncategorized
        assert basis_diversity(g) == pytest.approx(0.0)

    def test_two_equal_categories_max_diversity(self):
        g = graph(node("A", category="X"), node("B", category="Y"))
        # p_X = p_Y = 0.5 → H = 1.0 bit → diversity = 1.0
        assert basis_diversity(g) == pytest.approx(1.0, abs=1e-9)

    def test_triangle_diversity(self):
        # p_tech=2/3, p_fin=1/3 → H = -(2/3 log2(2/3) + 1/3 log2(1/3))
        p = [2 / 3, 1 / 3]
        h = -sum(pi * math.log2(pi) for pi in p)
        expected = h / math.log2(2)
        assert basis_diversity(TRIANGLE) == pytest.approx(expected, rel=1e-9)

    def test_four_equal_categories_unity(self):
        nodes = [node(str(i), category=f"cat{i}") for i in range(4)]
        g = GraphInput(nodes=nodes)
        assert basis_diversity(g) == pytest.approx(1.0, abs=1e-9)

    def test_range_in_unit_interval(self):
        d = basis_diversity(TRIANGLE)
        assert 0.0 <= d <= 1.0

    def test_deterministic(self):
        assert basis_diversity(TRIANGLE) == basis_diversity(TRIANGLE)


# ---------------------------------------------------------------------------
# metrics: dependency_concentration
# ---------------------------------------------------------------------------


class TestDependencyConcentration:
    def test_no_edges_zero_concentration(self):
        g = graph(node("A"), node("B"), node("C"))
        assert dependency_concentration(g) == pytest.approx(0.0)

    def test_triangle_hhi(self):
        # s_A=0, s_B=1/3, s_C=2/3 → HHI = 0 + 1/9 + 4/9 = 5/9
        expected = 5 / 9
        assert dependency_concentration(TRIANGLE) == pytest.approx(expected, rel=1e-9)

    def test_star_graph_high_concentration(self):
        # All edges point to a single hub → HHI approaches 1.0 as n grows
        hub = node("hub")
        spokes = [node(f"s{i}", category=f"c{i}") for i in range(5)]
        edges = [edge(f"s{i}", "hub") for i in range(5)]
        g = GraphInput(nodes=[hub] + spokes, edges=edges)
        # in-degree: hub=5, others=0 → HHI = 1.0
        assert dependency_concentration(g) == pytest.approx(1.0, abs=1e-9)

    def test_balanced_in_degree_low_hhi(self):
        # Two nodes each receive equal in-flow → HHI = 0.5
        g = graph(
            node("A"),
            node("B"),
            node("src"),
            edges=[edge("src", "A", 1.0), edge("src", "B", 1.0)],
        )
        assert dependency_concentration(g) == pytest.approx(0.5, abs=1e-9)

    def test_range(self):
        c = dependency_concentration(TRIANGLE)
        assert 0.0 <= c <= 1.0

    def test_deterministic(self):
        assert dependency_concentration(TRIANGLE) == dependency_concentration(TRIANGLE)

    def test_strength_weighting(self):
        # Edge to A has double strength → A dominates
        g = graph(
            node("A"),
            node("B"),
            node("src"),
            edges=[edge("src", "A", 2.0), edge("src", "B", 1.0)],
        )
        # s_A=2/3, s_B=1/3 → HHI = 4/9 + 1/9 = 5/9
        assert dependency_concentration(g) == pytest.approx(5 / 9, rel=1e-9)


# ---------------------------------------------------------------------------
# metrics: portfolio_score
# ---------------------------------------------------------------------------


class TestPortfolioScore:
    def test_ideal_graph_high_score(self):
        # High diversity + no edges (zero concentration) → score = 0.5*1 + 0.5*1 = 1.0
        nodes = [node(f"n{i}", category=f"c{i}") for i in range(4)]
        g = GraphInput(nodes=nodes)
        assert portfolio_score(g) == pytest.approx(1.0, abs=1e-9)

    def test_worst_case_low_score(self):
        # Single category + all edges to one hub → diversity=0, conc=1 → score=0
        hub = node("hub", category="X")
        spokes = [node(f"s{i}", category="X") for i in range(3)]
        edges = [edge(f"s{i}", "hub") for i in range(3)]
        g = GraphInput(nodes=[hub] + spokes, edges=edges)
        assert portfolio_score(g) == pytest.approx(0.0, abs=1e-9)

    def test_triangle_score(self):
        div = basis_diversity(TRIANGLE)
        conc = dependency_concentration(TRIANGLE)
        expected = 0.5 * div + 0.5 * (1.0 - conc)
        assert portfolio_score(TRIANGLE) == pytest.approx(expected, rel=1e-9)

    def test_range(self):
        s = portfolio_score(TRIANGLE)
        assert 0.0 <= s <= 1.0


# ---------------------------------------------------------------------------
# compute_all
# ---------------------------------------------------------------------------


class TestComputeAll:
    def test_returns_portfolio_metrics(self):
        pm = compute_all(TRIANGLE)
        assert pm.node_count == 3
        assert pm.edge_count == 3

    def test_consistency_with_individual_metrics(self):
        pm = compute_all(TRIANGLE)
        assert pm.basis_diversity == pytest.approx(basis_diversity(TRIANGLE), rel=1e-9)
        assert pm.dependency_concentration == pytest.approx(
            dependency_concentration(TRIANGLE), rel=1e-9
        )
        assert pm.portfolio_score == pytest.approx(portfolio_score(TRIANGLE), rel=1e-9)

    def test_single_node_no_edges(self):
        g = graph(node("solo", category="X"))
        pm = compute_all(g)
        assert pm.node_count == 1
        assert pm.edge_count == 0
        assert pm.basis_diversity == pytest.approx(0.0)
        assert pm.dependency_concentration == pytest.approx(0.0)
        assert pm.portfolio_score == pytest.approx(0.5)  # 0.5*0 + 0.5*(1-0)

    def test_deterministic(self):
        assert compute_all(TRIANGLE).basis_diversity == compute_all(TRIANGLE).basis_diversity
