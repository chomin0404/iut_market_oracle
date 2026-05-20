"""Tests for Monte Carlo risk API endpoints.

Covers:
    POST /api/v1/simulate
    POST /api/v1/risk/boundary
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.app import app

client = TestClient(app)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_BASE_SIMULATE_BODY = {
    "n_vars": 2,
    "n_samples": 2000,
    "distributions": [
        {"name": "normal", "params": {"loc": 0.0, "scale": 1.0}},
        {"name": "lognormal", "params": {"s": 0.8, "loc": 0.0, "scale": 1.0}},
    ],
    "copula": {"type": "gaussian", "corr_matrix": [[1, 0.6], [0.6, 1]]},
    "seed": 42,
}


@pytest.fixture(scope="module")
def simulation_id() -> str:
    resp = client.post("/api/v1/simulate", json=_BASE_SIMULATE_BODY)
    assert resp.status_code == 200
    return resp.json()["simulation_id"]


# ---------------------------------------------------------------------------
# POST /api/v1/simulate
# ---------------------------------------------------------------------------


class TestSimulate:
    def test_success_returns_200(self) -> None:
        resp = client.post("/api/v1/simulate", json=_BASE_SIMULATE_BODY)
        assert resp.status_code == 200

    def test_response_schema(self) -> None:
        resp = client.post("/api/v1/simulate", json=_BASE_SIMULATE_BODY)
        data = resp.json()
        assert "simulation_id" in data
        assert data["n_samples"] == _BASE_SIMULATE_BODY["n_samples"]
        assert "summary" in data
        for key in ("mean", "std", "var_95", "es_95"):
            assert key in data["summary"], f"missing {key}"
            assert isinstance(data["summary"][key], float)

    def test_simulation_id_is_uuid(self) -> None:
        import uuid

        resp = client.post("/api/v1/simulate", json=_BASE_SIMULATE_BODY)
        uid = resp.json()["simulation_id"]
        uuid.UUID(uid)  # raises ValueError if not valid UUID

    def test_deterministic_with_seed(self) -> None:
        r1 = client.post("/api/v1/simulate", json=_BASE_SIMULATE_BODY).json()["summary"]
        r2 = client.post("/api/v1/simulate", json=_BASE_SIMULATE_BODY).json()["summary"]
        assert math.isclose(r1["mean"], r2["mean"], rel_tol=1e-9)
        assert math.isclose(r1["var_95"], r2["var_95"], rel_tol=1e-9)

    def test_var_95_less_than_es_95(self) -> None:
        resp = client.post("/api/v1/simulate", json=_BASE_SIMULATE_BODY)
        summary = resp.json()["summary"]
        assert summary["es_95"] >= summary["var_95"]

    def test_distributions_count_mismatch_returns_422(self) -> None:
        body = {**_BASE_SIMULATE_BODY, "n_vars": 3}
        resp = client.post("/api/v1/simulate", json=body)
        assert resp.status_code == 422

    def test_unsupported_copula_returns_400(self) -> None:
        body = {
            **_BASE_SIMULATE_BODY,
            "copula": {"type": "clayton", "corr_matrix": [[1, 0.5], [0.5, 1]]},
        }
        resp = client.post("/api/v1/simulate", json=body)
        assert resp.status_code == 400

    def test_unsupported_distribution_returns_400(self) -> None:
        body = {
            **_BASE_SIMULATE_BODY,
            "distributions": [
                {"name": "cauchy_unknown", "params": {}},
                {"name": "normal", "params": {"loc": 0.0, "scale": 1.0}},
            ],
        }
        resp = client.post("/api/v1/simulate", json=body)
        assert resp.status_code == 400

    def test_n_samples_too_large_returns_422(self) -> None:
        body = {**_BASE_SIMULATE_BODY, "n_samples": 200_000}
        resp = client.post("/api/v1/simulate", json=body)
        assert resp.status_code == 422

    def test_gev_distribution_works(self) -> None:
        body = {
            "n_vars": 1,
            "n_samples": 1000,
            "distributions": [{"name": "gev", "params": {"c": 0.2, "loc": 10.0, "scale": 5.0}}],
            "copula": {"type": "gaussian", "corr_matrix": [[1.0]]},
            "seed": 1,
        }
        resp = client.post("/api/v1/simulate", json=body)
        assert resp.status_code == 200
        assert resp.json()["summary"]["mean"] > 0


# ---------------------------------------------------------------------------
# POST /api/v1/risk/boundary
# ---------------------------------------------------------------------------


class TestRiskBoundary:
    def test_success_from_simulation_id(self, simulation_id: str) -> None:
        body = {
            "simulation_id": simulation_id,
            "target_variable_index": 0,
            "thresholds": [1.0, 2.0, 3.0],
            "confidence_level": 0.95,
            "bootstrap_n": 100,
        }
        resp = client.post("/api/v1/risk/boundary", json=body)
        assert resp.status_code == 200

    def test_response_schema(self, simulation_id: str) -> None:
        body = {
            "simulation_id": simulation_id,
            "thresholds": [1.0, 2.0, 3.0, 4.0, 5.0],
            "bootstrap_n": 100,
        }
        resp = client.post("/api/v1/risk/boundary", json=body)
        data = resp.json()
        assert data["thresholds"] == [1.0, 2.0, 3.0, 4.0, 5.0]
        assert len(data["exceedance_probs"]) == 5
        assert len(data["confidence_band"]["lower"]) == 5
        assert len(data["confidence_band"]["upper"]) == 5
        assert "var_95" in data
        assert "es_95" in data

    def test_exceedance_probs_monotone_decreasing(self, simulation_id: str) -> None:
        body = {
            "simulation_id": simulation_id,
            "thresholds": [0.5, 1.0, 1.5, 2.0, 3.0],
            "bootstrap_n": 50,
        }
        probs = client.post("/api/v1/risk/boundary", json=body).json()["exceedance_probs"]
        for p_prev, p_next in zip(probs, probs[1:]):
            assert p_prev >= p_next - 1e-9, "exceedance_probs must be non-increasing"

    def test_confidence_band_contains_exceedance(self, simulation_id: str) -> None:
        body = {
            "simulation_id": simulation_id,
            "thresholds": [1.0, 2.0],
            "bootstrap_n": 200,
        }
        data = client.post("/api/v1/risk/boundary", json=body).json()
        for i in range(2):
            p = data["exceedance_probs"][i]
            lo = data["confidence_band"]["lower"][i]
            hi = data["confidence_band"]["upper"][i]
            # Central estimate should lie within (or very near) the CI
            assert lo <= p + 0.05
            assert hi >= p - 0.05

    def test_from_raw_samples(self) -> None:
        samples = [0.5, 1.2, 2.3, 0.8, 3.1, 1.5, 0.3, 4.0, 0.9, 1.7]
        body = {
            "samples": samples,
            "thresholds": [1.0, 2.0, 3.0],
            "bootstrap_n": 50,
        }
        resp = client.post("/api/v1/risk/boundary", json=body)
        assert resp.status_code == 200
        probs = resp.json()["exceedance_probs"]
        # 6 of 10 samples exceed 1.0
        assert math.isclose(probs[0], 0.6, abs_tol=1e-9)

    def test_unknown_simulation_id_returns_404(self) -> None:
        body = {
            "simulation_id": "00000000-0000-0000-0000-000000000000",
            "thresholds": [1.0],
        }
        resp = client.post("/api/v1/risk/boundary", json=body)
        assert resp.status_code == 404

    def test_out_of_range_variable_index_returns_400(self, simulation_id: str) -> None:
        body = {
            "simulation_id": simulation_id,
            "target_variable_index": 99,
            "thresholds": [1.0],
        }
        resp = client.post("/api/v1/risk/boundary", json=body)
        assert resp.status_code == 400

    def test_no_source_returns_422(self) -> None:
        body = {"thresholds": [1.0]}
        resp = client.post("/api/v1/risk/boundary", json=body)
        assert resp.status_code == 422

    def test_es_greater_than_or_equal_var(self, simulation_id: str) -> None:
        body = {
            "simulation_id": simulation_id,
            "thresholds": [1.0],
            "bootstrap_n": 50,
        }
        data = client.post("/api/v1/risk/boundary", json=body).json()
        assert data["es_95"] >= data["var_95"]


# ---------------------------------------------------------------------------
# risk_metrics unit tests
# ---------------------------------------------------------------------------


class TestRiskMetricsUnit:
    def test_compute_var(self) -> None:
        from core.risk_metrics import compute_var

        rng = np.random.default_rng(0)
        s = rng.standard_normal(10_000)
        var = compute_var(s, 0.95)
        # Theoretical 95th percentile of N(0,1) ≈ 1.645
        assert 1.55 < var < 1.75

    def test_compute_es_greater_than_var(self) -> None:
        from core.risk_metrics import compute_es, compute_var

        rng = np.random.default_rng(1)
        s = rng.standard_normal(5000)
        var = compute_var(s, 0.95)
        es = compute_es(s, 0.95)
        assert es >= var

    def test_exceedance_curve_monotone(self) -> None:
        from core.risk_metrics import compute_exceedance_curve

        rng = np.random.default_rng(2)
        s = rng.standard_normal(5000)
        thresholds = [-1.0, 0.0, 1.0, 2.0, 3.0]
        probs = compute_exceedance_curve(s, thresholds)
        for p1, p2 in zip(probs, probs[1:]):
            assert p1 >= p2 - 1e-9

    def test_confidence_band_shapes(self) -> None:
        from core.risk_metrics import compute_confidence_band

        rng = np.random.default_rng(3)
        s = rng.standard_normal(1000)
        thresholds = [0.0, 1.0, 2.0]
        band = compute_confidence_band(s, thresholds, bootstrap_n=100)
        assert len(band["lower"]) == 3
        assert len(band["upper"]) == 3
        for lo, hi in zip(band["lower"], band["upper"]):
            assert lo <= hi + 1e-9
