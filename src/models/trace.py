"""Append-only traceability graph for ModelForge (ModelForge T1400+).

Graph structure:
    Nodes — TraceNode (JSONL, one per line, append-only)
    Edges — encoded as parent_ids list on each node

Storage:
    artifacts/modelforge/trace.jsonl   (canonical, never overwritten)

Node ID generation:
    node_id = SHA-256(content_hash + ":" + node_type + ":" + model_id)[:16]
    Short enough to be readable, collision probability negligible for O(1000) nodes.

Usage::

    graph = TraceGraph()
    graph.append(node)
    nodes = graph.load_all()
    chain = graph.ancestry(node_id)   # DAG ancestors
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from pathlib import Path

from schemas import TraceNode, TraceNodeType

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_TRACE_PATH = Path("artifacts") / "modelforge" / "trace.jsonl"
_NODE_ID_LENGTH: int = 16  # hex chars


# ---------------------------------------------------------------------------
# Node ID helper
# ---------------------------------------------------------------------------


def make_node_id(content_hash: str, node_type: TraceNodeType, model_id: str) -> str:
    """Stable 16-hex-char node ID from content hash + type + model.

    Deterministic: same inputs → same ID.
    """
    raw = f"{content_hash}:{node_type.value}:{model_id}".encode()
    return hashlib.sha256(raw).hexdigest()[:_NODE_ID_LENGTH]


# ---------------------------------------------------------------------------
# TraceGraph
# ---------------------------------------------------------------------------


class TraceGraph:
    """Append-only DAG of TraceNodes stored as JSONL.

    Args:
        path: Path to the JSONL file (created if absent).
    """

    def __init__(self, path: Path = _DEFAULT_TRACE_PATH) -> None:
        self._path = path

    # ── Write ────────────────────────────────────────────────────────────

    def append(self, node: TraceNode) -> bool:
        """Append one TraceNode to the JSONL log (idempotent — skips if node_id already present).

        Returns:
            True if node was written, False if it already existed.
        """
        if self._path.exists():
            existing_ids = {n.node_id for n in self.load_all()}
            if node.node_id in existing_ids:
                return False
        self._path.parent.mkdir(parents=True, exist_ok=True)
        line = node.model_dump_json() + "\n"
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(line)
        return True

    # ── Read ─────────────────────────────────────────────────────────────

    def load_all(self) -> list[TraceNode]:
        """Return all nodes in append order."""
        if not self._path.exists():
            return []
        nodes: list[TraceNode] = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                nodes.append(TraceNode.model_validate_json(line))
        return nodes

    def load_model(self, model_id: str) -> list[TraceNode]:
        """Return all nodes for a specific model_id."""
        return [n for n in self.load_all() if n.model_id == model_id]

    def get_node(self, node_id: str) -> TraceNode | None:
        """Look up a node by its node_id (O(n) scan)."""
        for node in self.load_all():
            if node.node_id == node_id:
                return node
        return None

    # ── Graph traversal ──────────────────────────────────────────────────

    def ancestry(self, node_id: str) -> list[TraceNode]:
        """BFS ancestry: all nodes reachable by following parent_ids upstream.

        Returns nodes in BFS order (closest ancestors first).
        Returns empty list if node_id not found.
        """
        all_nodes = {n.node_id: n for n in self.load_all()}
        if node_id not in all_nodes:
            return []

        visited: set[str] = set()
        queue: deque[str] = deque(all_nodes[node_id].parent_ids)
        result: list[TraceNode] = []

        while queue:
            nid = queue.popleft()
            if nid in visited or nid not in all_nodes:
                continue
            visited.add(nid)
            node = all_nodes[nid]
            result.append(node)
            queue.extend(node.parent_ids)

        return result

    def descendants(self, node_id: str) -> list[TraceNode]:
        """BFS descendants: all nodes that have node_id as an ancestor."""
        all_nodes = self.load_all()
        visited: set[str] = set()
        frontier: set[str] = {node_id}
        result: list[TraceNode] = []

        while frontier:
            next_frontier: set[str] = set()
            for node in all_nodes:
                if node.node_id in visited:
                    continue
                if any(pid in frontier for pid in node.parent_ids):
                    visited.add(node.node_id)
                    result.append(node)
                    next_frontier.add(node.node_id)
            frontier = next_frontier

        return result

    # ── Serialisation ────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Export full graph as a dict with nodes and edges lists."""
        nodes = self.load_all()
        edges = [{"from": pid, "to": n.node_id} for n in nodes for pid in n.parent_ids]
        return {
            "nodes": [json.loads(n.model_dump_json()) for n in nodes],
            "edges": edges,
        }
