"""HTTP endpoint tests for the valuation router.

Covers:
  POST /api/v1/valuation/dcf               — DCFRequest → DCFResponse
  POST /api/v1/valuation/dcf/reverse       — ReverseDCFRequest → ReverseDCFResponse
  POST /api/v1/valuation/scenario          — AssumptionSet → ScenarioResult
  POST /api/v1/valuation/scenarios/run-all — RunAllRequest → list[ScenarioResult]
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.app import app

client = TestClient(app)

# ---------------------------------------------------------------------------
# Shared payloads
# ---------------------------------------------------------------------------

_DCF_BASE = {
    "initial_fcf": 10.0,
    "growth_rate": 0.05,
    "discount_rate": 0.10,
    "forecast_years": 5,
    "terminal_growth_rate": 0.03,
}

_REVERSE_DCF_BASE = {
    "target_enterprise_value": 200.0,
    "initial_fcf": 10.0,
    "discount_rate": 0.10,
    "forecast_years": 5,
    "terminal_growth_rate": 0.03,
}

_ASSUMPTION_BASE = {
    "name": "test_scenario",
    "version": "1.0",
    "params": {
        "initial_revenue": 10_000.0,
        "revenue_growth": 0.05,
        "ebit_margin": 0.12,
        "tax_rate": 0.30,
        "capex_rate": 0.30,
        "discount_rate": 0.10,
        "terminal_growth_rate": 0.02,
        "forecast_years": 5,
    },
}


# ---------------------------------------------------------------------------
# POST /valuation/dcf
# ---------------------------------------------------------------------------


class TestDCFEndpoint:
    def test_status_200(self) -> None:
        resp = client.post("/api/v1/valuation/dcf", json=_DCF_BASE)
        assert resp.status_code == 200

    def test_response_schema(self) -> None:
        body = client.post("/api/v1/valuation/dcf", json=_DCF_BASE).json()
        for field in (
            "projected_fcfs",
            "discounted_fcfs",
            "terminal_value",
            "discounted_terminal_value",
            "enterprise_value",
        ):
            assert field in body

    def test_projected_fcfs_length_equals_forecast_years(self) -> None:
        body = client.post("/api/v1/valuation/dcf", json=_DCF_BASE).json()
        assert len(body["projected_fcfs"]) == _DCF_BASE["forecast_years"]

    def test_enterprise_value_positive(self) -> None:
        body = client.post("/api/v1/valuation/dcf", json=_DCF_BASE).json()
        assert body["enterprise_value"] > 0.0

    def test_higher_growth_yields_higher_ev(self) -> None:
        low = client.post("/api/v1/valuation/dcf", json={**_DCF_BASE, "growth_rate": 0.01}).json()[
            "enterprise_value"
        ]
        high = client.post("/api/v1/valuation/dcf", json={**_DCF_BASE, "growth_rate": 0.20}).json()[
            "enterprise_value"
        ]
        assert high > low

    def test_negative_initial_fcf_returns_422(self) -> None:
        resp = client.post("/api/v1/valuation/dcf", json={**_DCF_BASE, "initial_fcf": -1.0})
        assert resp.status_code == 422

    def test_terminal_growth_gte_discount_rate_returns_400(self) -> None:
        # discount_rate <= terminal_growth_rate → DCF raises ValueError → 400
        resp = client.post(
            "/api/v1/valuation/dcf",
            json={**_DCF_BASE, "terminal_growth_rate": 0.15, "discount_rate": 0.10},
        )
        assert resp.status_code == 400

    def test_single_forecast_year(self) -> None:
        resp = client.post("/api/v1/valuation/dcf", json={**_DCF_BASE, "forecast_years": 1})
        assert resp.status_code == 200
        assert len(resp.json()["projected_fcfs"]) == 1


# ---------------------------------------------------------------------------
# POST /valuation/dcf/reverse
# ---------------------------------------------------------------------------


class TestReverseDCFEndpoint:
    def test_status_200(self) -> None:
        resp = client.post("/api/v1/valuation/dcf/reverse", json=_REVERSE_DCF_BASE)
        assert resp.status_code == 200

    def test_response_contains_implied_growth_rate(self) -> None:
        body = client.post("/api/v1/valuation/dcf/reverse", json=_REVERSE_DCF_BASE).json()
        assert "implied_growth_rate" in body

    def test_implied_growth_is_float(self) -> None:
        body = client.post("/api/v1/valuation/dcf/reverse", json=_REVERSE_DCF_BASE).json()
        assert isinstance(body["implied_growth_rate"], float)

    def test_higher_target_ev_implies_higher_growth(self) -> None:
        g_low = client.post(
            "/api/v1/valuation/dcf/reverse",
            json={**_REVERSE_DCF_BASE, "target_enterprise_value": 100.0},
        ).json()["implied_growth_rate"]
        g_high = client.post(
            "/api/v1/valuation/dcf/reverse",
            json={**_REVERSE_DCF_BASE, "target_enterprise_value": 500.0},
        ).json()["implied_growth_rate"]
        assert g_high > g_low

    def test_zero_initial_fcf_returns_422(self) -> None:
        resp = client.post(
            "/api/v1/valuation/dcf/reverse",
            json={**_REVERSE_DCF_BASE, "initial_fcf": 0.0},
        )
        assert resp.status_code == 422

    def test_negative_target_ev_returns_422(self) -> None:
        resp = client.post(
            "/api/v1/valuation/dcf/reverse",
            json={**_REVERSE_DCF_BASE, "target_enterprise_value": -1.0},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /valuation/scenario
# ---------------------------------------------------------------------------


class TestScenarioEndpoint:
    def test_status_200(self) -> None:
        resp = client.post("/api/v1/valuation/scenario", json=_ASSUMPTION_BASE)
        assert resp.status_code == 200

    def test_response_schema(self) -> None:
        body = client.post("/api/v1/valuation/scenario", json=_ASSUMPTION_BASE).json()
        for field in ("scenario_name", "assumption_version", "value", "sensitivity"):
            assert field in body

    def test_scenario_name_matches_input(self) -> None:
        body = client.post("/api/v1/valuation/scenario", json=_ASSUMPTION_BASE).json()
        assert body["scenario_name"] == _ASSUMPTION_BASE["name"]

    def test_sensitivity_is_dict(self) -> None:
        body = client.post("/api/v1/valuation/scenario", json=_ASSUMPTION_BASE).json()
        assert isinstance(body["sensitivity"], dict)

    def test_empty_params_returns_422(self) -> None:
        payload = {**_ASSUMPTION_BASE, "params": {}}
        resp = client.post("/api/v1/valuation/scenario", json=payload)
        assert resp.status_code == 422

    def test_discount_rate_lte_terminal_growth_returns_400(self) -> None:
        payload = {
            **_ASSUMPTION_BASE,
            "params": {
                **_ASSUMPTION_BASE["params"],
                "discount_rate": 0.01,
                "terminal_growth_rate": 0.05,
            },
        }
        resp = client.post("/api/v1/valuation/scenario", json=payload)
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# POST /valuation/scenarios/run-all
# ---------------------------------------------------------------------------


class TestRunAllEndpoint:
    def test_status_200_with_default_dir(self) -> None:
        resp = client.post(
            "/api/v1/valuation/scenarios/run-all",
            json={"scenario_dir": "configs/scenarios"},
        )
        assert resp.status_code == 200

    def test_returns_list(self) -> None:
        body = client.post(
            "/api/v1/valuation/scenarios/run-all",
            json={"scenario_dir": "configs/scenarios"},
        ).json()
        assert isinstance(body, list)

    def test_list_has_three_scenarios(self) -> None:
        """configs/scenarios contains base.yaml, bear.yaml, bull.yaml."""
        body = client.post(
            "/api/v1/valuation/scenarios/run-all",
            json={"scenario_dir": "configs/scenarios"},
        ).json()
        assert len(body) == 3

    def test_each_result_has_scenario_name(self) -> None:
        body = client.post(
            "/api/v1/valuation/scenarios/run-all",
            json={"scenario_dir": "configs/scenarios"},
        ).json()
        for entry in body:
            assert "scenario_name" in entry
            assert "value" in entry

    def test_nonexistent_dir_returns_empty_list(self) -> None:
        """Non-existent directory silently yields an empty result list."""
        resp = client.post(
            "/api/v1/valuation/scenarios/run-all",
            json={"scenario_dir": "configs/nonexistent_xyz"},
        )
        assert resp.status_code == 200
        assert resp.json() == []
