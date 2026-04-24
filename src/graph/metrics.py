"""Graph-based metrics for basis diversity, dependency concentration, and portfolio score.

Metric definitions
------------------
basis_diversity
    Normalised Shannon entropy of the node-weight distribution across categories.

        H = -Σ_k  p_k · log₂(p_k)
        basis_diversity = H / log₂(K)    (K = number of distinct categories)

    Range: [0, 1].
    0 → all nodes in one category (no diversity).
    1 → node weight perfectly uniform across all categories.
    Returns 0.0 when there is only one category (K = 1) or the graph is empty.

dependency_concentration  (Herfindahl-Hirschman Index on in-degrees)
    Measures how much inbound flow is concentrated on a small set of nodes.

        s_i = weighted_in_degree(i) / Σ weighted_in_degree(j)
        HHI = Σ_i  s_i²

    Range: [1/N_active, 1.0], or 0.0 when there are no edges.
    N_active = number of nodes with in-degree > 0.
    1.0 → all edges point to a single node (maximum concentration).

portfolio_score
    Composite health indicator combining the two metrics.

        portfolio_score = 0.5 · basis_diversity + 0.5 · (1 − min(HHI, 1.0))

    Range: [0, 1].  Higher is better.
"""

from __future__ import annotations

import math

from graph.builder import category_weight_map, weighted_in_degree
from schemas import GraphInput, PortfolioMetrics

# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------


def basis_diversity(graph: GraphInput) -> float:
    """Normalised Shannon entropy of node-weight distribution across categories.

    Returns
    -------
    float
        Value in [0, 1].
    """
    cat_weight = category_weight_map(graph)
    K = len(cat_weight)
    if K <= 1:
        return 0.0

    total = sum(cat_weight.values())
    if total == 0.0:
        return 0.0

    entropy = 0.0
    for w in cat_weight.values():
        p = w / total
        if p > 0.0:
            entropy -= p * math.log2(p)

    return entropy / math.log2(K)


def dependency_concentration(graph: GraphInput) -> float:
    """Herfindahl-Hirschman Index computed on weighted in-degrees.

    Returns
    -------
    float
        Value in [0, 1].  0.0 when there are no edges.
    """
    indeg = weighted_in_degree(graph)
    total = sum(indeg.values())
    if total == 0.0:
        return 0.0

    hhi = sum((w / total) ** 2 for w in indeg.values())
    return float(hhi)


def portfolio_score(graph: GraphInput) -> float:
    """Composite health score combining diversity and low concentration.

    Returns
    -------
    float
        Value in [0, 1].
    """
    div = basis_diversity(graph)
    conc = dependency_concentration(graph)
    return 0.5 * div + 0.5 * (1.0 - min(conc, 1.0))


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------


def compute_all(graph: GraphInput) -> PortfolioMetrics:
    """Compute all graph metrics and return as a validated PortfolioMetrics.

    Parameters
    ----------
    graph:
        A validated GraphInput (nodes + edges).

    Returns
    -------
    PortfolioMetrics
        All three scalar metrics plus structural counts.
    """
    div = basis_diversity(graph)
    conc = dependency_concentration(graph)
    score = 0.5 * div + 0.5 * (1.0 - min(conc, 1.0))

    return PortfolioMetrics(
        basis_diversity=div,
        dependency_concentration=conc,
        portfolio_score=score,
        node_count=len(graph.nodes),
        edge_count=len(graph.edges),
    )
