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
