"""Tests for /entropy/* endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.app import app

client = TestClient(app, raise_server_exceptions=False)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_NORMAL_PRIOR = {"distribution": "normal", "params": {"mean": 0.0, "std": 1.0}}
_BETA_PRIOR = {"distribution": "beta", "params": {"alpha": 2.0, "beta": 18.0}}

# Normal posterior: variance=0.04 > 0, within (0, 1) mean not required for normal
_NORMAL_POSTERIOR = {
    "mean": 0.5,
    "variance": 0.04,
    "credible_interval_95": [0.1, 0.9],
    "n_evidence": 10,
}

# Beta posterior: mean=0.1, variance=0.002 < mean*(1-mean)=0.09
_BETA_POSTERIOR = {
    "mean": 0.1,
    "variance": 0.002,
    "credible_interval_95": [0.02, 0.20],
    "n_evidence": 20,
}


# ---------------------------------------------------------------------------
# POST /entropy/entropy
# ---------------------------------------------------------------------------


def test_entropy_normal_returns_200() -> None:
    resp = client.post(
        "/api/v1/entropy/entropy",
        json={"posterior": _NORMAL_POSTERIOR, "prior": _NORMAL_PRIOR},
    )
    assert resp.status_code == 200


def test_entropy_normal_response_field() -> None:
    body = client.post(
        "/api/v1/entropy/entropy",
        json={"posterior": _NORMAL_POSTERIOR, "prior": _NORMAL_PRIOR},
    ).json()
    assert "entropy" in body
    assert isinstance(body["entropy"], float)


def test_entropy_beta_returns_200() -> None:
    resp = client.post(
        "/api/v1/entropy/entropy",
        json={"posterior": _BETA_POSTERIOR, "prior": _BETA_PRIOR},
    )
    assert resp.status_code == 200


def test_entropy_unsupported_distribution_returns_400() -> None:
    prior = {"distribution": "poisson", "params": {"lambda": 3.0}}
    resp = client.post(
        "/api/v1/entropy/entropy",
        json={"posterior": _NORMAL_POSTERIOR, "prior": prior},
    )
    assert resp.status_code == 400


def test_entropy_zero_variance_returns_400() -> None:
    posterior = {**_NORMAL_POSTERIOR, "variance": 0.0}
    resp = client.post(
        "/api/v1/entropy/entropy",
        json={"posterior": posterior, "prior": _NORMAL_PRIOR},
    )
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# POST /entropy/kl
# ---------------------------------------------------------------------------


def test_kl_normal_returns_200() -> None:
    resp = client.post(
        "/api/v1/entropy/kl",
        json={"posterior": _NORMAL_POSTERIOR, "prior": _NORMAL_PRIOR},
    )
    assert resp.status_code == 200


def test_kl_normal_response_field() -> None:
    body = client.post(
        "/api/v1/entropy/kl",
        json={"posterior": _NORMAL_POSTERIOR, "prior": _NORMAL_PRIOR},
    ).json()
    assert "kl_divergence" in body
    assert body["kl_divergence"] >= 0.0


def test_kl_beta_returns_200() -> None:
    resp = client.post(
        "/api/v1/entropy/kl",
        json={"posterior": _BETA_POSTERIOR, "prior": _BETA_PRIOR},
    )
    assert resp.status_code == 200


def test_kl_missing_std_param_returns_400() -> None:
    prior_no_std = {"distribution": "normal", "params": {"mean": 0.0}}
    resp = client.post(
        "/api/v1/entropy/kl",
        json={"posterior": _NORMAL_POSTERIOR, "prior": prior_no_std},
    )
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# POST /entropy/detect
# ---------------------------------------------------------------------------


def test_detect_returns_200() -> None:
    body = {
        "posteriors": [_NORMAL_POSTERIOR, _NORMAL_POSTERIOR, _NORMAL_POSTERIOR],
        "prior": _NORMAL_PRIOR,
        "experiment_id": "exp-001",
    }
    resp = client.post("/api/v1/entropy/detect", json=body)
    assert resp.status_code == 200


def test_detect_response_fields() -> None:
    body = {
        "posteriors": [_NORMAL_POSTERIOR, _NORMAL_POSTERIOR],
        "prior": _NORMAL_PRIOR,
        "experiment_id": "exp-042",
    }
    data = client.post("/api/v1/entropy/detect", json=body).json()
    assert "experiment_id" in data
    assert "entropy_series" in data
    assert "kl_series" in data
    assert "alerts" in data


def test_detect_entropy_series_length_matches_posteriors() -> None:
    posteriors = [_NORMAL_POSTERIOR] * 5
    body = {
        "posteriors": posteriors,
        "prior": _NORMAL_PRIOR,
        "experiment_id": "exp-001",
    }
    data = client.post("/api/v1/entropy/detect", json=body).json()
    assert len(data["entropy_series"]) == 5


def test_detect_no_alerts_for_identical_posteriors() -> None:
    posteriors = [_NORMAL_POSTERIOR] * 4
    body = {
        "posteriors": posteriors,
        "prior": _NORMAL_PRIOR,
        "experiment_id": "exp-001",
        "kl_threshold": 100.0,
        "entropy_gradient_threshold": 100.0,
    }
    data = client.post("/api/v1/entropy/detect", json=body).json()
    assert data["alerts"] == []


def test_detect_empty_posteriors_returns_422() -> None:
    body = {
        "posteriors": [],
        "prior": _NORMAL_PRIOR,
        "experiment_id": "exp-001",
    }
    resp = client.post("/api/v1/entropy/detect", json=body)
    assert resp.status_code == 422
