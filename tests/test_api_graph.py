"""Tests for POST /graph/metrics."""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.app import app

client = TestClient(app, raise_server_exceptions=False)

_URL = "/api/v1/graph/metrics"


# ---------------------------------------------------------------------------
# Success cases
# ---------------------------------------------------------------------------


def test_two_nodes_one_edge_returns_200() -> None:
    payload = {
        "nodes": [{"node_id": "A"}, {"node_id": "B"}],
        "edges": [{"source": "A", "target": "B"}],
    }
    resp = client.post(_URL, json=payload)
    assert resp.status_code == 200


def test_response_fields_present() -> None:
    payload = {
        "nodes": [{"node_id": "A"}, {"node_id": "B"}],
        "edges": [{"source": "A", "target": "B"}],
    }
    body = client.post(_URL, json=payload).json()
    assert "basis_diversity" in body
    assert "dependency_concentration" in body
    assert "portfolio_score" in body
    assert "node_count" in body
    assert "edge_count" in body


def test_node_count_matches_input() -> None:
    payload = {
        "nodes": [{"node_id": "X"}, {"node_id": "Y"}, {"node_id": "Z"}],
        "edges": [],
    }
    body = client.post(_URL, json=payload).json()
    assert body["node_count"] == 3


def test_edge_count_matches_input() -> None:
    payload = {
        "nodes": [{"node_id": "A"}, {"node_id": "B"}, {"node_id": "C"}],
        "edges": [{"source": "A", "target": "B"}, {"source": "B", "target": "C"}],
    }
    body = client.post(_URL, json=payload).json()
    assert body["edge_count"] == 2


def test_no_edges_returns_200() -> None:
    payload = {"nodes": [{"node_id": "solo"}], "edges": []}
    resp = client.post(_URL, json=payload)
    assert resp.status_code == 200


def test_metrics_in_valid_range() -> None:
    payload = {
        "nodes": [{"node_id": "A"}, {"node_id": "B"}],
        "edges": [{"source": "A", "target": "B"}],
    }
    body = client.post(_URL, json=payload).json()
    assert 0.0 <= body["basis_diversity"] <= 1.0
    assert body["dependency_concentration"] >= 0.0
    assert 0.0 <= body["portfolio_score"] <= 1.0


def test_node_with_weight_and_label() -> None:
    payload = {
        "nodes": [
            {"node_id": "skill_1", "label": "Python", "weight": 2.0},
            {"node_id": "skill_2", "label": "Statistics", "weight": 1.5},
        ],
        "edges": [{"source": "skill_1", "target": "skill_2", "strength": 0.8}],
    }
    resp = client.post(_URL, json=payload)
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_self_loop_returns_422() -> None:
    payload = {
        "nodes": [{"node_id": "A"}, {"node_id": "B"}],
        "edges": [{"source": "A", "target": "A"}],
    }
    resp = client.post(_URL, json=payload)
    assert resp.status_code == 422


def test_edge_referencing_missing_node_returns_422() -> None:
    payload = {
        "nodes": [{"node_id": "A"}],
        "edges": [{"source": "A", "target": "MISSING"}],
    }
    resp = client.post(_URL, json=payload)
    assert resp.status_code == 422


def test_empty_node_list_returns_422() -> None:
    payload = {"nodes": [], "edges": []}
    resp = client.post(_URL, json=payload)
    assert resp.status_code == 422


def test_node_id_empty_string_returns_422() -> None:
    payload = {"nodes": [{"node_id": ""}], "edges": []}
    resp = client.post(_URL, json=payload)
    assert resp.status_code == 422
