"""HTTP endpoint tests for /twin/simulate and /twin/calibrate (T800),
plus ValueError error-path coverage for regime/market endpoints."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from api.app import app

client = TestClient(app)

# ---------------------------------------------------------------------------
# Shared payloads
# ---------------------------------------------------------------------------

_STATE = {
    "experiment_id": "exp-001",
    "state_vector": [5.0, 0.10, -2.3],
    "state_labels": ["log_revenue", "growth_rate", "log_volatility"],
}

_SIMULATE_PAYLOAD = {
    "initial_state": _STATE,
    "horizon": 5,
    "n_samples": 3,
    "process_noise_std": 0.01,
    "random_seed": 42,
}

_CALIBRATE_PAYLOAD = {
    "observations": [0.08, 0.12, 0.09, 0.11],
    "prior": {"distribution": "normal", "params": {"mu": 0.10, "sigma": 0.05}},
    "experiment_id": "exp-001",
    "obs_precision": 1.0,
}

_REGIME_PAYLOAD = {
    "n_steps": 50,
    "initial_price": 100.0,
    "p_stay_normal": 0.95,
    "p_stay_volatile": 0.90,
    "random_seed": 42,
}

_MARKET_PAYLOAD = {
    "n_steps": 50,
    "gamma_alpha": 2.0,
    "gamma_beta": 1.0,
    "random_seed": 42,
}


# ---------------------------------------------------------------------------
# POST /twin/simulate
# ---------------------------------------------------------------------------


class TestSimulateEndpoint:
    def test_status_200(self) -> None:
        r = client.post("/api/v1/twin/simulate", json=_SIMULATE_PAYLOAD)
        assert r.status_code == 200

    def test_response_schema(self) -> None:
        body = client.post("/api/v1/twin/simulate", json=_SIMULATE_PAYLOAD).json()
        for field in ("experiment_id", "trajectories", "n_samples", "horizon", "state_labels"):
            assert field in body

    def test_trajectories_count_equals_n_samples(self) -> None:
        body = client.post("/api/v1/twin/simulate", json=_SIMULATE_PAYLOAD).json()
        assert len(body["trajectories"]) == _SIMULATE_PAYLOAD["n_samples"]

    def test_with_transition_matrix(self) -> None:
        payload = {
            **_SIMULATE_PAYLOAD,
            "transition_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        }
        r = client.post("/api/v1/twin/simulate", json=payload)
        assert r.status_code == 200

    def test_reproducibility(self) -> None:
        r1 = client.post("/api/v1/twin/simulate", json=_SIMULATE_PAYLOAD)
        r2 = client.post("/api/v1/twin/simulate", json=_SIMULATE_PAYLOAD)
        assert r1.json()["trajectories"] == r2.json()["trajectories"]

    def test_different_seeds_differ(self) -> None:
        r1 = client.post("/api/v1/twin/simulate", json={**_SIMULATE_PAYLOAD, "random_seed": 1})
        r2 = client.post("/api/v1/twin/simulate", json={**_SIMULATE_PAYLOAD, "random_seed": 2})
        assert r1.json()["trajectories"] != r2.json()["trajectories"]

    def test_value_error_returns_400(self) -> None:
        with patch("api.routers.twin.simulate", side_effect=ValueError("bad input")):
            r = client.post("/api/v1/twin/simulate", json=_SIMULATE_PAYLOAD)
        assert r.status_code == 400

    def test_missing_random_seed_returns_422(self) -> None:
        payload = {k: v for k, v in _SIMULATE_PAYLOAD.items() if k != "random_seed"}
        assert client.post("/api/v1/twin/simulate", json=payload).status_code == 422

    def test_n_samples_zero_returns_422(self) -> None:
        assert (
            client.post(
                "/api/v1/twin/simulate", json={**_SIMULATE_PAYLOAD, "n_samples": 0}
            ).status_code
            == 422
        )


# ---------------------------------------------------------------------------
# POST /twin/calibrate
# ---------------------------------------------------------------------------


class TestCalibrateEndpoint:
    def test_status_200(self) -> None:
        r = client.post("/api/v1/twin/calibrate", json=_CALIBRATE_PAYLOAD)
        assert r.status_code == 200

    def test_response_schema(self) -> None:
        body = client.post("/api/v1/twin/calibrate", json=_CALIBRATE_PAYLOAD).json()
        assert "posterior" in body
        assert "state" in body

    def test_posterior_has_mean_and_credible_interval(self) -> None:
        posterior = client.post("/api/v1/twin/calibrate", json=_CALIBRATE_PAYLOAD).json()[
            "posterior"
        ]
        assert "mean" in posterior
        assert "credible_interval_95" in posterior

    def test_state_has_state_vector(self) -> None:
        state = client.post("/api/v1/twin/calibrate", json=_CALIBRATE_PAYLOAD).json()["state"]
        assert "state_vector" in state

    def test_value_error_returns_400(self) -> None:
        with patch("api.routers.twin.calibrate", side_effect=ValueError("calibration failed")):
            r = client.post("/api/v1/twin/calibrate", json=_CALIBRATE_PAYLOAD)
        assert r.status_code == 400

    def test_missing_observations_returns_422(self) -> None:
        payload = {k: v for k, v in _CALIBRATE_PAYLOAD.items() if k != "observations"}
        assert client.post("/api/v1/twin/calibrate", json=payload).status_code == 422


# ---------------------------------------------------------------------------
# Error paths — regime / market (ValueError → 400)
# ---------------------------------------------------------------------------


class TestRegimeMarketErrorPaths:
    def test_regime_simulate_value_error_returns_400(self) -> None:
        with patch(
            "api.routers.twin.simulate_regime_switching", side_effect=ValueError("regime err")
        ):
            r = client.post("/api/v1/twin/regime-simulate", json=_REGIME_PAYLOAD)
        assert r.status_code == 400

    def test_regime_simulate_summary_value_error_returns_400(self) -> None:
        with patch(
            "api.routers.twin.simulate_regime_switching", side_effect=ValueError("regime err")
        ):
            r = client.post("/api/v1/twin/regime-simulate/summary", json=_REGIME_PAYLOAD)
        assert r.status_code == 400

    def test_market_evolve_value_error_returns_400(self) -> None:
        with patch(
            "api.routers.twin.simulate_market_evolution", side_effect=ValueError("market err")
        ):
            r = client.post("/api/v1/twin/market-evolve", json=_MARKET_PAYLOAD)
        assert r.status_code == 400
