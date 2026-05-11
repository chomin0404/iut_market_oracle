"""Tests for POST /yield-twin/recommend and POST /yield-twin/report (T1600 API)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.app import app

client = TestClient(app, raise_server_exceptions=False)

# ---------------------------------------------------------------------------
# Shared payloads
# ---------------------------------------------------------------------------

_FACTOR_SPECS = [
    {"name": "temp", "low": 150.0, "high": 250.0},
    {"name": "pressure", "low": 1.0, "high": 5.0},
]

_OBSERVATIONS = [
    {"factors": {"temp": 150.0, "pressure": 1.0}, "yield_obs": 0.60},
    {"factors": {"temp": 200.0, "pressure": 3.0}, "yield_obs": 0.88},
    {"factors": {"temp": 250.0, "pressure": 5.0}, "yield_obs": 0.75},
]

_BASE_PAYLOAD: dict = {
    "factor_specs": _FACTOR_SPECS,
    "observations": _OBSERVATIONS,
    "random_seed": 42,
}


# ---------------------------------------------------------------------------
# POST /yield-twin/recommend
# ---------------------------------------------------------------------------


def test_recommend_status_ok() -> None:
    resp = client.post("/yield-twin/recommend", json=_BASE_PAYLOAD)
    assert resp.status_code == 200


def test_recommend_returns_doe_recommendation_fields() -> None:
    body = client.post("/yield-twin/recommend", json=_BASE_PAYLOAD).json()
    assert "factors" in body
    assert "expected_improvement" in body
    assert "d_leverage" in body
    assert "fusion_score" in body
    assert "predicted_yield" in body
    assert "predicted_std" in body
    assert "acquisition_mode" in body
    assert "n_observations" in body


def test_recommend_factors_within_bounds() -> None:
    body = client.post("/yield-twin/recommend", json=_BASE_PAYLOAD).json()
    factors = body["factors"]
    assert 150.0 <= factors["temp"] <= 250.0
    assert 1.0 <= factors["pressure"] <= 5.0


def test_recommend_predicted_yield_in_unit_interval() -> None:
    body = client.post("/yield-twin/recommend", json=_BASE_PAYLOAD).json()
    assert 0.0 <= body["predicted_yield"] <= 1.0


def test_recommend_n_observations_matches_input() -> None:
    body = client.post("/yield-twin/recommend", json=_BASE_PAYLOAD).json()
    assert body["n_observations"] == len(_OBSERVATIONS)


def test_recommend_no_observations_doe_explore_mode() -> None:
    payload = {"factor_specs": _FACTOR_SPECS, "random_seed": 0}
    body = client.post("/yield-twin/recommend", json=payload).json()
    assert body["acquisition_mode"] == "doe_explore"
    assert body["n_observations"] == 0


def test_recommend_reproducible_with_same_seed() -> None:
    r1 = client.post("/yield-twin/recommend", json=_BASE_PAYLOAD).json()
    r2 = client.post("/yield-twin/recommend", json=_BASE_PAYLOAD).json()
    assert r1["factors"] == pytest.approx(r2["factors"], rel=1e-9)


def test_recommend_different_seed_may_differ() -> None:
    payload_b = {**_BASE_PAYLOAD, "random_seed": 999}
    r1 = client.post("/yield-twin/recommend", json=_BASE_PAYLOAD).json()
    r2 = client.post("/yield-twin/recommend", json=payload_b).json()
    # Not guaranteed to differ, but with high probability the LHS draws differ
    # — we just check that both responses are valid
    assert r1["factors"] is not None
    assert r2["factors"] is not None


# ---------------------------------------------------------------------------
# POST /yield-twin/report
# ---------------------------------------------------------------------------


def test_report_status_ok() -> None:
    resp = client.post("/yield-twin/report", json=_BASE_PAYLOAD)
    assert resp.status_code == 200


def test_report_fields() -> None:
    body = client.post("/yield-twin/report", json=_BASE_PAYLOAD).json()
    assert "n_observations" in body
    assert "best_yield_observed" in body
    assert "best_factors" in body
    assert "surrogate_loocv_r2" in body
    assert "recommendation" in body
    assert "gp_hyperparams" in body
    assert "factor_specs" in body


def test_report_best_yield_correct() -> None:
    body = client.post("/yield-twin/report", json=_BASE_PAYLOAD).json()
    # Best yield in _OBSERVATIONS is 0.88
    assert abs(body["best_yield_observed"] - 0.88) < 1e-9


def test_report_factor_specs_echo() -> None:
    body = client.post("/yield-twin/report", json=_BASE_PAYLOAD).json()
    returned_names = {fs["name"] for fs in body["factor_specs"]}
    assert returned_names == {"temp", "pressure"}


def test_report_gp_hyperparams_populated_when_enough_obs() -> None:
    body = client.post("/yield-twin/report", json=_BASE_PAYLOAD).json()
    # 3 observations ≥ 2 → GP is fitted → hyperparams non-empty
    assert len(body["gp_hyperparams"]) > 0


def test_report_single_observation_no_loocv() -> None:
    payload = {
        "factor_specs": _FACTOR_SPECS,
        "observations": [{"factors": {"temp": 200.0, "pressure": 3.0}, "yield_obs": 0.80}],
        "random_seed": 42,
    }
    body = client.post("/yield-twin/report", json=payload).json()
    assert body["n_observations"] == 1
    assert body["surrogate_loocv_r2"] is None


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_missing_factor_specs_returns_422() -> None:
    resp = client.post("/yield-twin/recommend", json={"observations": []})
    assert resp.status_code == 422


def test_duplicate_factor_names_returns_422() -> None:
    payload = {
        "factor_specs": [
            {"name": "temp", "low": 150.0, "high": 250.0},
            {"name": "temp", "low": 200.0, "high": 300.0},
        ],
    }
    resp = client.post("/yield-twin/recommend", json=payload)
    assert resp.status_code == 422


def test_unknown_factor_in_observations_returns_422() -> None:
    payload = {
        "factor_specs": _FACTOR_SPECS,
        "observations": [{"factors": {"temp": 200.0, "unknown_factor": 1.0}, "yield_obs": 0.5}],
    }
    resp = client.post("/yield-twin/recommend", json=payload)
    assert resp.status_code == 422


def test_low_ge_high_factor_spec_returns_422() -> None:
    payload = {
        "factor_specs": [{"name": "temp", "low": 250.0, "high": 150.0}],
    }
    resp = client.post("/yield-twin/recommend", json=payload)
    assert resp.status_code == 422


def test_n_candidates_too_small_returns_422() -> None:
    payload = {**_BASE_PAYLOAD, "n_candidates": 5}
    resp = client.post("/yield-twin/recommend", json=payload)
    assert resp.status_code == 422
