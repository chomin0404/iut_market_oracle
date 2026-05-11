"""Linear Structural Causal Model (SCM) — ATE via DFS path enumeration (T1700).

DAG representation:
    edges: list[CausalEdge]   where each edge stores b_{effect←cause}

Average Treatment Effect (linear SCM):
    ATE(cause → effect) = Σ_{directed paths p: cause→…→effect} Π_{(a→b) ∈ p} b_{b←a}

Path enumeration:
    DFS with visited-set cycle guard; depth capped at _MAX_PATH_DEPTH.
"""

from __future__ import annotations

from collections import defaultdict

from schemas import CausalEdge, CausalEffect

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_PATH_DEPTH: int = 20


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_adjacency(
    edges: list[CausalEdge],
) -> dict[str, list[tuple[str, float]]]:
    """Build forward adjacency list: cause → [(effect, coefficient), ...]."""
    adj: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for e in edges:
        adj[e.cause].append((e.effect, e.coefficient))
    return dict(adj)


def _enumerate_paths(
    adj: dict[str, list[tuple[str, float]]],
    source: str,
    target: str,
) -> list[list[tuple[str, str, float]]]:
    """DFS: enumerate all directed paths from source to target.

    Each path is a list of (from, to, coefficient) triples.
    Cycle guard: no node visited twice per path.
    Depth cap: at most _MAX_PATH_DEPTH edges.
    """
    results: list[list[tuple[str, str, float]]] = []
    stack: list[tuple[str, list[tuple[str, str, float]], set[str]]] = [(source, [], {source})]

    while stack:
        node, path, visited = stack.pop()
        if node == target and path:
            results.append(path)
            continue
        if len(path) >= _MAX_PATH_DEPTH:
            continue
        for neighbour, coef in adj.get(node, []):
            if neighbour not in visited:
                stack.append((neighbour, path + [(node, neighbour, coef)], visited | {neighbour}))

    return results


def _path_product(path: list[tuple[str, str, float]]) -> float:
    """Product of edge coefficients along a path."""
    product = 1.0
    for _, _, coef in path:
        product *= coef
    return product


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_ate(
    edges: list[CausalEdge],
    cause: str,
    effect: str,
) -> CausalEffect:
    """Compute total ATE from cause to effect via all directed paths.

    ATE = Σ_paths Π coefficients

    Args:
        edges:  DAG edges.
        cause:  Source variable name.
        effect: Target variable name.

    Returns:
        CausalEffect with total_effect and n_paths.
    """
    adj = _build_adjacency(edges)
    paths = _enumerate_paths(adj, cause, effect)
    total = sum(_path_product(p) for p in paths)
    return CausalEffect(
        cause=cause,
        effect=effect,
        total_effect=total,
        n_paths=len(paths),
    )


def compute_all_effects(
    edges: list[CausalEdge],
    causes: list[str],
    effects: list[str],
) -> list[CausalEffect]:
    """Compute ATE for every (cause, effect) pair.

    Args:
        edges:   DAG edges.
        causes:  Source variables.
        effects: Target variables.

    Returns:
        List of CausalEffect (one per pair, only where cause ≠ effect).
    """
    results: list[CausalEffect] = []
    for cause in causes:
        for effect in effects:
            if cause == effect:
                continue
            results.append(compute_ate(edges, cause, effect))
    return results
