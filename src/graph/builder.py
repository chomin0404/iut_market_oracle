"""Graph construction utilities for the Dependency / Skill Graph module.

All functions operate on a validated GraphInput and return plain Python
structures so they can be consumed by metrics.py or serialised directly.

Definitions
-----------
weighted_in_degree(v)
    Σ edge.strength  for all edges with target == v.

weighted_out_degree(v)
    Σ edge.strength  for all edges with source == v.

adjacency_list
    Mapping  source_id → [(target_id, strength), ...]
"""

from __future__ import annotations

from collections import defaultdict

from schemas import GraphInput

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def adjacency_list(graph: GraphInput) -> dict[str, list[tuple[str, float]]]:
    """Return directed adjacency list.

    Returns
    -------
    dict[str, list[tuple[str, float]]]
        ``adj[src]`` is a list of ``(target, strength)`` tuples.
        Every node (even isolates) appears as a key.
    """
    adj: dict[str, list[tuple[str, float]]] = {n.node_id: [] for n in graph.nodes}
    for edge in graph.edges:
        adj[edge.source].append((edge.target, edge.strength))
    return adj


def weighted_in_degree(graph: GraphInput) -> dict[str, float]:
    """Return weighted in-degree for every node.

    Parameters
    ----------
    graph:
        Validated GraphInput.

    Returns
    -------
    dict[str, float]
        Mapping node_id → Σ(strength) of all incoming edges.
        Nodes with no incoming edges map to 0.0.
    """
    indeg: dict[str, float] = {n.node_id: 0.0 for n in graph.nodes}
    for edge in graph.edges:
        indeg[edge.target] += edge.strength
    return indeg


def weighted_out_degree(graph: GraphInput) -> dict[str, float]:
    """Return weighted out-degree for every node."""
    outdeg: dict[str, float] = {n.node_id: 0.0 for n in graph.nodes}
    for edge in graph.edges:
        outdeg[edge.source] += edge.strength
    return outdeg


def category_weight_map(graph: GraphInput) -> dict[str, float]:
    """Return total node weight grouped by category.

    Nodes whose ``category`` field is empty are grouped under
    ``"uncategorized"``.
    """
    cat_weight: dict[str, float] = defaultdict(float)
    for node in graph.nodes:
        key = node.category if node.category else "uncategorized"
        cat_weight[key] += node.weight
    return dict(cat_weight)
