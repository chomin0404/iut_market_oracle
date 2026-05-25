"""Tests for POST /api/v1/bayesian/network/infer."""

from __future__ import annotations

import math

import pytest
from fastapi.testclient import TestClient

from api.app import app

client = TestClient(app, raise_server_exceptions=False)

_URL = "/api/v1/bayesian/network/infer"

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_ECONOMY_NODE = {"node_id": "economy", "states": ["expansion", "recession"]}
_REGIME_NODE  = {"node_id": "regime",  "states": ["bull", "bear", "neutral"]}
_ECONOMY_CPD  = {"node_id": "economy", "probs": [0.7, 0.3]}
_REGIME_CPD   = {
    "node_id": "regime",
    "rows": [
        {"parent_states": ["expansion"], "probs": [0.6, 0.1, 0.3]},
        {"parent_states": ["recession"], "probs": [0.2, 0.6, 0.2]},
    ],
}

_TWO_NODE_NETWORK = {
    "nodes": [_ECONOMY_NODE, _REGIME_NODE],
    "edges": [{"parent": "economy", "child": "regime"}],
    "cpds": [_ECONOMY_CPD, _REGIME_CPD],
}


def _post(network: dict, evidence: dict, queries: list[str]) -> dict:
    resp = client.post(_URL, json={"network": network, "evidence": evidence, "queries": queries})
    return resp


# ---------------------------------------------------------------------------
# Success — HTTP 200
# ---------------------------------------------------------------------------


def test_returns_200_with_evidence() -> None:
    resp = _post(_TWO_NODE_NETWORK, {"economy": "expansion"}, ["regime"])
    assert resp.status_code == 200


def test_returns_200_no_evidence() -> None:
    resp = _post(_TWO_NODE_NETWORK, {}, ["economy", "regime"])
    assert resp.status_code == 200


def test_response_contains_posteriors_key() -> None:
    body = _post(_TWO_NODE_NETWORK, {}, ["regime"]).json()
    assert "posteriors" in body


def test_queried_nodes_present_in_posteriors() -> None:
    body = _post(_TWO_NODE_NETWORK, {}, ["economy", "regime"]).json()
    assert "economy" in body["posteriors"]
    assert "regime" in body["posteriors"]


def test_posteriors_sum_to_one_with_evidence() -> None:
    body = _post(_TWO_NODE_NETWORK, {"economy": "expansion"}, ["regime"]).json()
    total = sum(body["posteriors"]["regime"].values())
    assert math.isclose(total, 1.0, abs_tol=1e-6)


def test_posteriors_sum_to_one_no_evidence() -> None:
    body = _post(_TWO_NODE_NETWORK, {}, ["economy", "regime"]).json()
    for node, dist in body["posteriors"].items():
        assert math.isclose(sum(dist.values()), 1.0, abs_tol=1e-6), node


def test_posterior_values_with_evidence() -> None:
    # P(regime | economy=expansion) = [0.6, 0.1, 0.3]
    body = _post(_TWO_NODE_NETWORK, {"economy": "expansion"}, ["regime"]).json()
    dist = body["posteriors"]["regime"]
    assert math.isclose(dist["bull"],    0.6, abs_tol=1e-6)
    assert math.isclose(dist["bear"],    0.1, abs_tol=1e-6)
    assert math.isclose(dist["neutral"], 0.3, abs_tol=1e-6)


def test_prior_marginal_no_evidence() -> None:
    # P(regime) = 0.7*[0.6,0.1,0.3] + 0.3*[0.2,0.6,0.2] = [0.48, 0.25, 0.27]
    body = _post(_TWO_NODE_NETWORK, {}, ["regime"]).json()
    dist = body["posteriors"]["regime"]
    assert math.isclose(dist["bull"],    0.48, abs_tol=1e-6)
    assert math.isclose(dist["bear"],    0.25, abs_tol=1e-6)
    assert math.isclose(dist["neutral"], 0.27, abs_tol=1e-6)


def test_observed_node_returns_degenerate_distribution() -> None:
    # Querying an observed node → probability 1 on the observed state
    body = _post(_TWO_NODE_NETWORK, {"economy": "expansion"}, ["economy"]).json()
    dist = body["posteriors"]["economy"]
    assert math.isclose(dist["expansion"], 1.0, abs_tol=1e-6)
    assert math.isclose(dist["recession"], 0.0, abs_tol=1e-6)


def test_root_only_network() -> None:
    network = {
        "nodes": [{"node_id": "x", "states": ["a", "b", "c"]}],
        "edges": [],
        "cpds": [{"node_id": "x", "probs": [0.2, 0.5, 0.3]}],
    }
    body = _post(network, {}, ["x"]).json()
    dist = body["posteriors"]["x"]
    assert math.isclose(dist["a"], 0.2, abs_tol=1e-6)
    assert math.isclose(dist["b"], 0.5, abs_tol=1e-6)
    assert math.isclose(dist["c"], 0.3, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# Evidence errors — HTTP 400
# ---------------------------------------------------------------------------


def test_unknown_evidence_node_returns_400() -> None:
    resp = _post(_TWO_NODE_NETWORK, {"unknown": "expansion"}, ["regime"])
    assert resp.status_code == 400


def test_invalid_evidence_state_returns_400() -> None:
    resp = _post(_TWO_NODE_NETWORK, {"economy": "nonexistent_state"}, ["regime"])
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Network spec validation errors — HTTP 422
# ---------------------------------------------------------------------------


def test_missing_queries_returns_422() -> None:
    resp = client.post(_URL, json={"network": _TWO_NODE_NETWORK, "evidence": {}, "queries": []})
    assert resp.status_code == 422


def test_invalid_network_cycle_returns_422() -> None:
    network = {
        "nodes": [
            {"node_id": "a", "states": ["0", "1"]},
            {"node_id": "b", "states": ["0", "1"]},
        ],
        "edges": [{"parent": "a", "child": "b"}, {"parent": "b", "child": "a"}],
        "cpds": [
            {"node_id": "a", "rows": [
                {"parent_states": ["0"], "probs": [0.5, 0.5]},
                {"parent_states": ["1"], "probs": [0.4, 0.6]},
            ]},
            {"node_id": "b", "rows": [
                {"parent_states": ["0"], "probs": [0.5, 0.5]},
                {"parent_states": ["1"], "probs": [0.4, 0.6]},
            ]},
        ],
    }
    resp = _post(network, {}, ["a"])
    assert resp.status_code == 422


def test_missing_cpd_returns_422() -> None:
    network = {
        "nodes": [_ECONOMY_NODE, _REGIME_NODE],
        "edges": [{"parent": "economy", "child": "regime"}],
        "cpds": [_ECONOMY_CPD],  # regime CPD missing
    }
    resp = _post(network, {}, ["regime"])
    assert resp.status_code == 422


def test_invalid_cpd_probs_not_summing_to_one_returns_422() -> None:
    network = {
        "nodes": [{"node_id": "x", "states": ["a", "b"]}],
        "edges": [],
        "cpds": [{"node_id": "x", "probs": [0.3, 0.3]}],  # sums to 0.6
    }
    resp = _post(network, {}, ["x"])
    assert resp.status_code == 422
