"""HTTP endpoint tests for T1100 regime-switching and market-evolution API."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.app import app

client = TestClient(app)

# ---------------------------------------------------------------------------
# Shared payloads
# ---------------------------------------------------------------------------

_REGIME_PAYLOAD = {
    "n_steps": 100,
    "initial_price": 100.0,
    "p_stay_normal": 0.95,
    "p_stay_volatile": 0.90,
    "random_seed": 42,
}

_MARKET_PAYLOAD = {
    "n_steps": 100,
    "gamma_alpha": 2.0,
    "gamma_beta": 1.0,
    "random_seed": 42,
}


# ---------------------------------------------------------------------------
# POST /twin/regime-simulate — status and schema
# ---------------------------------------------------------------------------


class TestRegimeSimulate:
    def test_status_200(self):
        r = client.post("/twin/regime-simulate", json=_REGIME_PAYLOAD)
        assert r.status_code == 200

    def test_response_has_required_fields(self):
        r = client.post("/twin/regime-simulate", json=_REGIME_PAYLOAD)
        body = r.json()
        for field in ("n_steps", "prices", "regimes", "p_stay_normal", "p_stay_volatile"):
            assert field in body

    def test_prices_length_equals_n_steps(self):
        r = client.post("/twin/regime-simulate", json=_REGIME_PAYLOAD)
        body = r.json()
        assert len(body["prices"]) == body["n_steps"]

    def test_regimes_length_equals_n_steps(self):
        r = client.post("/twin/regime-simulate", json=_REGIME_PAYLOAD)
        body = r.json()
        assert len(body["regimes"]) == body["n_steps"]

    def test_initial_price_preserved(self):
        r = client.post("/twin/regime-simulate", json=_REGIME_PAYLOAD)
        assert r.json()["prices"][0] == pytest.approx(100.0)

    def test_regimes_binary(self):
        r = client.post("/twin/regime-simulate", json=_REGIME_PAYLOAD)
        assert all(reg in (0, 1) for reg in r.json()["regimes"])

    def test_prices_all_positive(self):
        r = client.post("/twin/regime-simulate", json=_REGIME_PAYLOAD)
        assert all(p > 0 for p in r.json()["prices"])

    def test_reproducibility(self):
        r1 = client.post("/twin/regime-simulate", json=_REGIME_PAYLOAD)
        r2 = client.post("/twin/regime-simulate", json=_REGIME_PAYLOAD)
        assert r1.json()["prices"] == r2.json()["prices"]

    def test_different_seeds_differ(self):
        r1 = client.post("/twin/regime-simulate", json={**_REGIME_PAYLOAD, "random_seed": 1})
        r2 = client.post("/twin/regime-simulate", json={**_REGIME_PAYLOAD, "random_seed": 2})
        assert r1.json()["prices"] != r2.json()["prices"]

    def test_n_steps_above_max_returns_422(self):
        r = client.post("/twin/regime-simulate", json={**_REGIME_PAYLOAD, "n_steps": 9999})
        assert r.status_code == 422

    def test_missing_random_seed_returns_422(self):
        payload = {k: v for k, v in _REGIME_PAYLOAD.items() if k != "random_seed"}
        r = client.post("/twin/regime-simulate", json=payload)
        assert r.status_code == 422

    def test_p_stay_normal_zero_returns_422(self):
        r = client.post("/twin/regime-simulate", json={**_REGIME_PAYLOAD, "p_stay_normal": 0.0})
        assert r.status_code == 422

    def test_initial_price_zero_returns_422(self):
        r = client.post("/twin/regime-simulate", json={**_REGIME_PAYLOAD, "initial_price": 0.0})
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# POST /twin/regime-simulate/summary
# ---------------------------------------------------------------------------


class TestRegimeSimulateSummary:
    def test_status_200(self):
        r = client.post("/twin/regime-simulate/summary", json=_REGIME_PAYLOAD)
        assert r.status_code == 200

    def test_response_has_required_fields(self):
        r = client.post("/twin/regime-simulate/summary", json=_REGIME_PAYLOAD)
        body = r.json()
        for field in (
            "n_steps",
            "final_price",
            "min_price",
            "max_price",
            "regime_0_fraction",
            "regime_1_fraction",
            "regime_switch_count",
        ):
            assert field in body

    def test_no_full_price_series(self):
        r = client.post("/twin/regime-simulate/summary", json=_REGIME_PAYLOAD)
        assert "prices" not in r.json()

    def test_fractions_sum_to_one(self):
        r = client.post("/twin/regime-simulate/summary", json=_REGIME_PAYLOAD)
        body = r.json()
        total = body["regime_0_fraction"] + body["regime_1_fraction"]
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_min_le_final_le_max(self):
        r = client.post("/twin/regime-simulate/summary", json=_REGIME_PAYLOAD)
        body = r.json()
        assert body["min_price"] <= body["final_price"] <= body["max_price"]

    def test_switch_count_non_negative(self):
        r = client.post("/twin/regime-simulate/summary", json=_REGIME_PAYLOAD)
        assert r.json()["regime_switch_count"] >= 0

    def test_switch_count_less_than_n_steps(self):
        r = client.post("/twin/regime-simulate/summary", json=_REGIME_PAYLOAD)
        body = r.json()
        assert body["regime_switch_count"] < body["n_steps"]

    def test_summary_consistent_with_full(self):
        """final_price in summary must equal last element of full price series."""
        full = client.post("/twin/regime-simulate", json=_REGIME_PAYLOAD).json()
        summary = client.post("/twin/regime-simulate/summary", json=_REGIME_PAYLOAD).json()
        assert summary["final_price"] == pytest.approx(full["prices"][-1], rel=1e-9)
        assert summary["min_price"] == pytest.approx(min(full["prices"]), rel=1e-9)
        assert summary["max_price"] == pytest.approx(max(full["prices"]), rel=1e-9)

    def test_reproducibility(self):
        r1 = client.post("/twin/regime-simulate/summary", json=_REGIME_PAYLOAD)
        r2 = client.post("/twin/regime-simulate/summary", json=_REGIME_PAYLOAD)
        assert r1.json() == r2.json()


# ---------------------------------------------------------------------------
# POST /twin/market-evolve — status and schema
# ---------------------------------------------------------------------------


class TestMarketEvolve:
    def test_status_200(self):
        r = client.post("/twin/market-evolve", json=_MARKET_PAYLOAD)
        assert r.status_code == 200

    def test_response_has_required_fields(self):
        r = client.post("/twin/market-evolve", json=_MARKET_PAYLOAD)
        body = r.json()
        for field in (
            "n_steps",
            "new_customers",
            "cumulative_base",
            "sigmoid_factor",
            "market_capture",
            "gamma_alpha",
            "gamma_beta",
        ):
            assert field in body

    def test_all_series_length_equals_n_steps(self):
        r = client.post("/twin/market-evolve", json=_MARKET_PAYLOAD)
        body = r.json()
        n = body["n_steps"]
        for field in ("new_customers", "cumulative_base", "sigmoid_factor", "market_capture"):
            assert len(body[field]) == n

    def test_new_customers_non_negative(self):
        r = client.post("/twin/market-evolve", json=_MARKET_PAYLOAD)
        assert all(k >= 0 for k in r.json()["new_customers"])

    def test_sigmoid_factor_in_unit_interval(self):
        r = client.post("/twin/market-evolve", json=_MARKET_PAYLOAD)
        assert all(0.0 < s < 1.0 for s in r.json()["sigmoid_factor"])

    def test_reproducibility(self):
        r1 = client.post("/twin/market-evolve", json=_MARKET_PAYLOAD)
        r2 = client.post("/twin/market-evolve", json=_MARKET_PAYLOAD)
        assert r1.json()["new_customers"] == r2.json()["new_customers"]

    def test_different_seeds_differ(self):
        r1 = client.post("/twin/market-evolve", json={**_MARKET_PAYLOAD, "random_seed": 1})
        r2 = client.post("/twin/market-evolve", json={**_MARKET_PAYLOAD, "random_seed": 2})
        assert r1.json()["new_customers"] != r2.json()["new_customers"]

    def test_n_steps_above_max_returns_422(self):
        r = client.post("/twin/market-evolve", json={**_MARKET_PAYLOAD, "n_steps": 9999})
        assert r.status_code == 422

    def test_missing_random_seed_returns_422(self):
        payload = {k: v for k, v in _MARKET_PAYLOAD.items() if k != "random_seed"}
        r = client.post("/twin/market-evolve", json=payload)
        assert r.status_code == 422

    def test_gamma_alpha_zero_returns_422(self):
        r = client.post("/twin/market-evolve", json={**_MARKET_PAYLOAD, "gamma_alpha": 0.0})
        assert r.status_code == 422
