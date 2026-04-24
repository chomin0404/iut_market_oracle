"""HTTP endpoint tests for T1200 matroid log-concavity API."""

from __future__ import annotations

import math

import pytest
from fastapi.testclient import TestClient

from api.app import app

client = TestClient(app)

# ---------------------------------------------------------------------------
# Shared payload
# ---------------------------------------------------------------------------

_PAYLOAD = {
    "n_assets": 10,
    "rank_weight": 0.8,
    "corank_weight": 1.2,
}


# ---------------------------------------------------------------------------
# POST /matroid/log-concavity — status and schema
# ---------------------------------------------------------------------------


class TestLogConcavityStatus:
    def test_status_200(self):
        r = client.post("/matroid/log-concavity", json=_PAYLOAD)
        assert r.status_code == 200

    def test_response_has_required_fields(self):
        r = client.post("/matroid/log-concavity", json=_PAYLOAD)
        body = r.json()
        for field in (
            "n_assets",
            "rank_weight",
            "corank_weight",
            "subset_sizes",
            "probability_mass",
            "log_probability",
            "log_concavity_checks",
            "is_log_concave",
        ):
            assert field in body

    def test_n_assets_above_max_returns_422(self):
        r = client.post("/matroid/log-concavity", json={**_PAYLOAD, "n_assets": 201})
        assert r.status_code == 422

    def test_n_assets_zero_returns_422(self):
        r = client.post("/matroid/log-concavity", json={**_PAYLOAD, "n_assets": 0})
        assert r.status_code == 422

    def test_rank_weight_zero_returns_422(self):
        r = client.post("/matroid/log-concavity", json={**_PAYLOAD, "rank_weight": 0.0})
        assert r.status_code == 422

    def test_corank_weight_negative_returns_422(self):
        r = client.post("/matroid/log-concavity", json={**_PAYLOAD, "corank_weight": -1.0})
        assert r.status_code == 422

    def test_default_weights_applied(self):
        r = client.post("/matroid/log-concavity", json={"n_assets": 5})
        assert r.status_code == 200
        body = r.json()
        assert body["rank_weight"] == pytest.approx(0.8)
        assert body["corank_weight"] == pytest.approx(1.2)


# ---------------------------------------------------------------------------
# POST /matroid/log-concavity — structural invariants
# ---------------------------------------------------------------------------


class TestLogConcavityInvariants:
    def test_subset_sizes_range(self):
        r = client.post("/matroid/log-concavity", json=_PAYLOAD)
        body = r.json()
        assert body["subset_sizes"] == list(range(body["n_assets"] + 1))

    def test_all_series_length_n_plus_one(self):
        r = client.post("/matroid/log-concavity", json=_PAYLOAD)
        body = r.json()
        n = body["n_assets"]
        for field in ("subset_sizes", "probability_mass", "log_probability"):
            assert len(body[field]) == n + 1

    def test_log_concavity_checks_length_n_minus_one(self):
        r = client.post("/matroid/log-concavity", json=_PAYLOAD)
        body = r.json()
        assert len(body["log_concavity_checks"]) == body["n_assets"] - 1

    def test_probability_mass_sums_to_one(self):
        r = client.post("/matroid/log-concavity", json=_PAYLOAD)
        total = sum(r.json()["probability_mass"])
        assert math.isclose(total, 1.0, rel_tol=1e-9)

    def test_probability_mass_all_positive(self):
        r = client.post("/matroid/log-concavity", json=_PAYLOAD)
        assert all(p > 0 for p in r.json()["probability_mass"])

    def test_is_log_concave_true_for_standard_params(self):
        r = client.post("/matroid/log-concavity", json=_PAYLOAD)
        assert r.json()["is_log_concave"] is True

    def test_all_checks_true_for_standard_params(self):
        r = client.post("/matroid/log-concavity", json=_PAYLOAD)
        assert all(r.json()["log_concavity_checks"])

    def test_log_probability_consistent_with_mass(self):
        r = client.post("/matroid/log-concavity", json=_PAYLOAD)
        body = r.json()
        for p, lp in zip(body["probability_mass"], body["log_probability"]):
            assert lp == pytest.approx(math.log(p), rel=1e-6)


# ---------------------------------------------------------------------------
# POST /matroid/log-concavity — parametric behaviour
# ---------------------------------------------------------------------------


class TestLogConcavityParametric:
    def test_symmetric_weights_give_symmetric_pmf(self):
        r = client.post(
            "/matroid/log-concavity",
            json={"n_assets": 10, "rank_weight": 1.0, "corank_weight": 1.0},
        )
        p = r.json()["probability_mass"]
        n = 10
        for k in range(n + 1):
            assert p[k] == pytest.approx(p[n - k], rel=1e-9)

    def test_high_rank_weight_shifts_mode_right(self):
        low = client.post(
            "/matroid/log-concavity",
            json={"n_assets": 10, "rank_weight": 0.1, "corank_weight": 1.9},
        ).json()["probability_mass"]
        high = client.post(
            "/matroid/log-concavity",
            json={"n_assets": 10, "rank_weight": 1.9, "corank_weight": 0.1},
        ).json()["probability_mass"]
        mode_low = max(range(11), key=lambda k: low[k])
        mode_high = max(range(11), key=lambda k: high[k])
        assert mode_high > mode_low

    def test_n_assets_1_edge_case(self):
        r = client.post("/matroid/log-concavity", json={"n_assets": 1})
        body = r.json()
        assert body["n_assets"] == 1
        assert len(body["subset_sizes"]) == 2
        assert body["log_concavity_checks"] == []
        assert body["is_log_concave"] is True

    def test_large_n_no_overflow(self):
        r = client.post(
            "/matroid/log-concavity",
            json={"n_assets": 200, "rank_weight": 0.8, "corank_weight": 1.2},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["is_log_concave"] is True
        assert math.isclose(sum(body["probability_mass"]), 1.0, rel_tol=1e-9)

    def test_idempotent(self):
        """Same parameters always return identical results (no randomness)."""
        r1 = client.post("/matroid/log-concavity", json=_PAYLOAD)
        r2 = client.post("/matroid/log-concavity", json=_PAYLOAD)
        assert r1.json()["probability_mass"] == r2.json()["probability_mass"]
