"""Tests for POST /bayesian/update."""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.app import app

client = TestClient(app, raise_server_exceptions=False)

_URL = "/api/v1/bayesian/update"

_BETA_PRIOR = {"distribution": "beta", "params": {"alpha": 2.0, "beta": 18.0}}
_NORMAL_PRIOR = {"distribution": "normal", "params": {"mu": 0.05, "sigma": 0.02}}
_EVIDENCE = [{"source": "obs_1", "kind": "observation", "value": 0.7, "weight": 1.0}]


# ---------------------------------------------------------------------------
# Success — Beta prior
# ---------------------------------------------------------------------------


def test_beta_update_returns_200() -> None:
    resp = client.post(_URL, json={"prior": _BETA_PRIOR, "evidence": _EVIDENCE})
    assert resp.status_code == 200


def test_beta_update_response_fields() -> None:
    body = client.post(_URL, json={"prior": _BETA_PRIOR, "evidence": _EVIDENCE}).json()
    assert "mean" in body
    assert "variance" in body
    assert "credible_interval_95" in body
    assert "n_evidence" in body


def test_beta_update_mean_in_unit_interval() -> None:
    body = client.post(_URL, json={"prior": _BETA_PRIOR, "evidence": _EVIDENCE}).json()
    assert 0.0 < body["mean"] < 1.0


def test_beta_update_variance_non_negative() -> None:
    body = client.post(_URL, json={"prior": _BETA_PRIOR, "evidence": _EVIDENCE}).json()
    assert body["variance"] >= 0.0


def test_beta_update_n_evidence_matches_input() -> None:
    body = client.post(_URL, json={"prior": _BETA_PRIOR, "evidence": _EVIDENCE}).json()
    assert body["n_evidence"] == len(_EVIDENCE)


def test_beta_update_multiple_evidence() -> None:
    evidence = [
        {"source": "obs_1", "kind": "observation", "value": 0.6, "weight": 1.0},
        {"source": "obs_2", "kind": "observation", "value": 0.8, "weight": 1.0},
    ]
    resp = client.post(_URL, json={"prior": _BETA_PRIOR, "evidence": evidence})
    assert resp.status_code == 200
    assert resp.json()["n_evidence"] == 2


# ---------------------------------------------------------------------------
# Success — Normal prior
# ---------------------------------------------------------------------------


def test_normal_update_returns_200() -> None:
    resp = client.post(_URL, json={"prior": _NORMAL_PRIOR, "evidence": _EVIDENCE})
    assert resp.status_code == 200


def test_normal_update_response_fields() -> None:
    body = client.post(_URL, json={"prior": _NORMAL_PRIOR, "evidence": _EVIDENCE}).json()
    assert "mean" in body
    assert "variance" in body
    assert "credible_interval_95" in body


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_missing_prior_returns_422() -> None:
    resp = client.post(_URL, json={"evidence": _EVIDENCE})
    assert resp.status_code == 422


def test_missing_evidence_returns_422() -> None:
    resp = client.post(_URL, json={"prior": _BETA_PRIOR})
    assert resp.status_code == 422


def test_empty_params_returns_422() -> None:
    prior = {"distribution": "beta", "params": {}}
    resp = client.post(_URL, json={"prior": prior, "evidence": _EVIDENCE})
    assert resp.status_code == 422


def test_invalid_distribution_returns_400() -> None:
    prior = {"distribution": "unsupported_dist", "params": {"x": 1.0}}
    resp = client.post(_URL, json={"prior": prior, "evidence": _EVIDENCE})
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# /infer — network Variable Elimination inference
# ---------------------------------------------------------------------------

_INFER_URL = "/api/v1/bayesian/infer"

_TWO_NODE_NET = {
    "nodes": [
        {"id": "economy", "states": ["expansion", "recession"]},
        {"id": "regime", "states": ["bull", "bear", "neutral"]},
    ],
    "edges": [{"parent": "economy", "child": "regime"}],
    "priors": [{"node": "economy", "probs": [0.70, 0.30]}],
    "cpts": [
        {
            "node": "regime",
            "rows": [
                {"parents": ["expansion"], "probs": [0.60, 0.10, 0.30]},
                {"parents": ["recession"], "probs": [0.20, 0.60, 0.20]},
            ],
        }
    ],
    "query": "regime",
    "evidence": {"economy": "expansion"},
}

_CHAIN_NET = {
    "nodes": [
        {"id": "economy", "states": ["expansion", "recession"]},
        {"id": "regime", "states": ["bull", "bear"]},
        {"id": "ret", "states": ["high", "low"]},
    ],
    "edges": [
        {"parent": "economy", "child": "regime"},
        {"parent": "regime", "child": "ret"},
    ],
    "priors": [{"node": "economy", "probs": [0.70, 0.30]}],
    "cpts": [
        {
            "node": "regime",
            "rows": [
                {"parents": ["expansion"], "probs": [0.60, 0.40]},
                {"parents": ["recession"], "probs": [0.25, 0.75]},
            ],
        },
        {
            "node": "ret",
            "rows": [
                {"parents": ["bull"], "probs": [0.80, 0.20]},
                {"parents": ["bear"], "probs": [0.20, 0.80]},
            ],
        },
    ],
    "query": "ret",
    "evidence": {"economy": "expansion"},
}


# --- Success cases ---


def test_infer_returns_200() -> None:
    resp = client.post(_INFER_URL, json=_TWO_NODE_NET)
    assert resp.status_code == 200


def test_infer_response_has_required_fields() -> None:
    body = client.post(_INFER_URL, json=_TWO_NODE_NET).json()
    assert "query" in body
    assert "evidence" in body
    assert "posterior" in body


def test_infer_posterior_sums_to_one() -> None:
    body = client.post(_INFER_URL, json=_TWO_NODE_NET).json()
    total = sum(body["posterior"].values())
    assert abs(total - 1.0) < 1e-6


def test_infer_posterior_state_keys_match_query_node() -> None:
    body = client.post(_INFER_URL, json=_TWO_NODE_NET).json()
    assert set(body["posterior"].keys()) == {"bull", "bear", "neutral"}


def test_infer_expansion_evidence_bull_dominant() -> None:
    body = client.post(_INFER_URL, json=_TWO_NODE_NET).json()
    # P(bull | expansion) = 0.60 > P(bear | expansion) = 0.10
    assert body["posterior"]["bull"] > body["posterior"]["bear"]


def test_infer_recession_evidence_bear_dominant() -> None:
    payload = {**_TWO_NODE_NET, "evidence": {"economy": "recession"}}
    body = client.post(_INFER_URL, json=payload).json()
    assert body["posterior"]["bear"] > body["posterior"]["bull"]


def test_infer_no_evidence_returns_prior_marginal() -> None:
    payload = {**_TWO_NODE_NET, "evidence": {}}
    body = client.post(_INFER_URL, json=payload).json()
    # P(bull) = 0.70*0.60 + 0.30*0.20 = 0.48
    assert abs(body["posterior"]["bull"] - 0.48) < 1e-6


def test_infer_query_root_node() -> None:
    payload = {**_TWO_NODE_NET, "query": "economy", "evidence": {}}
    body = client.post(_INFER_URL, json=payload).json()
    assert abs(body["posterior"]["expansion"] - 0.70) < 1e-6


def test_infer_chain_network_returns_200() -> None:
    resp = client.post(_INFER_URL, json=_CHAIN_NET)
    assert resp.status_code == 200


def test_infer_chain_network_posterior_sums_to_one() -> None:
    body = client.post(_INFER_URL, json=_CHAIN_NET).json()
    total = sum(body["posterior"].values())
    assert abs(total - 1.0) < 1e-6


def test_infer_evidence_echoed_in_response() -> None:
    body = client.post(_INFER_URL, json=_TWO_NODE_NET).json()
    assert body["evidence"] == {"economy": "expansion"}
    assert body["query"] == "regime"


# --- Validation / error cases ---


def test_infer_missing_nodes_returns_422() -> None:
    payload = {k: v for k, v in _TWO_NODE_NET.items() if k != "nodes"}
    resp = client.post(_INFER_URL, json=payload)
    assert resp.status_code == 422


def test_infer_missing_query_returns_422() -> None:
    payload = {k: v for k, v in _TWO_NODE_NET.items() if k != "query"}
    resp = client.post(_INFER_URL, json=payload)
    assert resp.status_code == 422


def test_infer_unknown_query_node_returns_400() -> None:
    payload = {**_TWO_NODE_NET, "query": "nonexistent"}
    resp = client.post(_INFER_URL, json=payload)
    assert resp.status_code == 400


def test_infer_unknown_evidence_state_returns_400() -> None:
    payload = {**_TWO_NODE_NET, "evidence": {"economy": "unknown_state"}}
    resp = client.post(_INFER_URL, json=payload)
    assert resp.status_code == 400


def test_infer_cycle_in_graph_returns_400() -> None:
    payload = {
        **_TWO_NODE_NET,
        "edges": [
            {"parent": "economy", "child": "regime"},
            {"parent": "regime", "child": "economy"},  # cycle
        ],
    }
    resp = client.post(_INFER_URL, json=payload)
    assert resp.status_code == 400


def test_infer_node_with_single_state_returns_422() -> None:
    # states must have >= 2 values — caught by Pydantic min_length
    payload = {
        **_TWO_NODE_NET,
        "nodes": [
            {"id": "economy", "states": ["expansion"]},  # only 1 state
            {"id": "regime", "states": ["bull", "bear", "neutral"]},
        ],
    }
    resp = client.post(_INFER_URL, json=payload)
    assert resp.status_code in {400, 422}


def test_infer_incomplete_cpt_returns_400() -> None:
    payload = {
        **_TWO_NODE_NET,
        "cpts": [
            {
                "node": "regime",
                "rows": [
                    # missing "recession" row
                    {"parents": ["expansion"], "probs": [0.60, 0.10, 0.30]},
                ],
            }
        ],
    }
    resp = client.post(_INFER_URL, json=payload)
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Named network registry — GET /networks  and  POST /networks/{name}/infer
# ---------------------------------------------------------------------------

_NETWORKS_URL = "/api/v1/bayesian/networks"
_WATER_INFER_URL = f"{_NETWORKS_URL}/fukuoka_water_demand/infer"


def test_list_networks_returns_200() -> None:
    resp = client.get(_NETWORKS_URL)
    assert resp.status_code == 200


def test_list_networks_is_list() -> None:
    body = client.get(_NETWORKS_URL).json()
    assert isinstance(body, list)


def test_list_networks_contains_fukuoka() -> None:
    body = client.get(_NETWORKS_URL).json()
    names = [entry["name"] for entry in body]
    assert "fukuoka_water_demand" in names


def test_list_networks_entry_has_description() -> None:
    body = client.get(_NETWORKS_URL).json()
    entry = next(e for e in body if e["name"] == "fukuoka_water_demand")
    assert "description" in entry and entry["description"]


def test_named_infer_returns_200() -> None:
    resp = client.post(_WATER_INFER_URL, json={"query": "demand_level", "evidence": {}})
    assert resp.status_code == 200


def test_named_infer_posterior_sums_to_one() -> None:
    body = client.post(_WATER_INFER_URL, json={"query": "demand_level", "evidence": {}}).json()
    total = sum(body["posterior"].values())
    assert abs(total - 1.0) < 1e-6


def test_named_infer_state_keys_match_demand_level() -> None:
    body = client.post(_WATER_INFER_URL, json={"query": "demand_level", "evidence": {}}).json()
    assert set(body["posterior"].keys()) == {"low", "normal", "high"}


def test_named_infer_summer_holiday_shifts_demand_high() -> None:
    body_summer = client.post(
        _WATER_INFER_URL,
        json={"query": "demand_level", "evidence": {"season": "summer", "day_type": "holiday"}},
    ).json()
    body_winter = client.post(
        _WATER_INFER_URL,
        json={"query": "demand_level", "evidence": {"season": "winter", "day_type": "weekday"}},
    ).json()
    assert body_summer["posterior"]["high"] > body_winter["posterior"]["high"]


def test_named_infer_evidence_echoed() -> None:
    evidence = {"season": "summer"}
    body = client.post(
        _WATER_INFER_URL, json={"query": "demand_level", "evidence": evidence}
    ).json()
    assert body["evidence"] == evidence
    assert body["query"] == "demand_level"


def test_named_infer_unknown_network_returns_404() -> None:
    resp = client.post(
        f"{_NETWORKS_URL}/nonexistent_network/infer",
        json={"query": "demand_level", "evidence": {}},
    )
    assert resp.status_code == 404


def test_named_infer_unknown_query_node_returns_400() -> None:
    resp = client.post(_WATER_INFER_URL, json={"query": "nonexistent_node", "evidence": {}})
    assert resp.status_code == 400


def test_named_infer_missing_query_returns_422() -> None:
    resp = client.post(_WATER_INFER_URL, json={"evidence": {}})
    assert resp.status_code == 422
