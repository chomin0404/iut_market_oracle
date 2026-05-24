"""Tests for /exit/* endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.app import app

client = TestClient(app, raise_server_exceptions=False)

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

_OPTION = {
    "name": "IPO exit",
    "exit_type": "ipo",
    "timing_earliest": 1.0,
    "timing_expected": 3.0,
    "timing_latest": 5.0,
    "value_by_scenario": {"bear": 80.0, "base": 150.0, "bull": 280.0},
    "floor_value": 0.0,
    "discount_rate": 0.10,
}

_OPTION_B = {
    "name": "M&A exit",
    "exit_type": "ma",
    "timing_earliest": 0.5,
    "timing_expected": 2.0,
    "timing_latest": 4.0,
    "value_by_scenario": {"bear": 100.0, "base": 200.0, "bull": 350.0},
    "floor_value": 50.0,
    "discount_rate": 0.08,
}

_SCENARIO_PROBS = {"bear": 0.2, "base": 0.5, "bull": 0.3}


# ---------------------------------------------------------------------------
# POST /exit/price
# ---------------------------------------------------------------------------


def test_price_returns_200() -> None:
    resp = client.post("/api/v1/exit/price", json={"option": _OPTION})
    assert resp.status_code == 200


def test_price_response_fields() -> None:
    body = client.post("/api/v1/exit/price", json={"option": _OPTION}).json()
    assert "option_name" in body
    assert "exit_type" in body
    assert "scenario_payoffs" in body
    assert "scenario_pvs" in body
    assert "expected_value" in body


def test_price_option_name_matches() -> None:
    body = client.post("/api/v1/exit/price", json={"option": _OPTION}).json()
    assert body["option_name"] == "IPO exit"


def test_price_with_scenario_probs() -> None:
    resp = client.post(
        "/api/v1/exit/price",
        json={"option": _OPTION, "scenario_probs": _SCENARIO_PROBS},
    )
    assert resp.status_code == 200
    assert resp.json()["expected_value"] > 0.0


def test_price_invalid_timing_returns_422() -> None:
    bad_option = {**_OPTION, "timing_earliest": 5.0, "timing_expected": 3.0}
    resp = client.post("/api/v1/exit/price", json={"option": bad_option})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /exit/price-all
# ---------------------------------------------------------------------------


def test_price_all_returns_200() -> None:
    resp = client.post("/api/v1/exit/price-all", json={"options": [_OPTION, _OPTION_B]})
    assert resp.status_code == 200


def test_price_all_returns_list() -> None:
    body = client.post("/api/v1/exit/price-all", json={"options": [_OPTION, _OPTION_B]}).json()
    assert isinstance(body, list)
    assert len(body) == 2


def test_price_all_sorted_descending() -> None:
    body = client.post(
        "/api/v1/exit/price-all",
        json={"options": [_OPTION, _OPTION_B], "scenario_probs": _SCENARIO_PROBS},
    ).json()
    evs = [item["expected_value"] for item in body]
    assert evs == sorted(evs, reverse=True)


# ---------------------------------------------------------------------------
# POST /exit/timing-map
# ---------------------------------------------------------------------------


def test_timing_map_returns_200() -> None:
    resp = client.post("/api/v1/exit/timing-map", json={"option": _OPTION})
    assert resp.status_code == 200


def test_timing_map_response_fields() -> None:
    body = client.post("/api/v1/exit/timing-map", json={"option": _OPTION}).json()
    assert "option_name" in body
    assert "time_steps" in body
    assert "probabilities" in body
    assert "expected_timing" in body


def test_timing_map_probs_sum_to_one() -> None:
    body = client.post("/api/v1/exit/timing-map", json={"option": _OPTION}).json()
    assert abs(sum(body["probabilities"]) - 1.0) < 1e-3


def test_timing_map_custom_n_steps() -> None:
    body = client.post("/api/v1/exit/timing-map", json={"option": _OPTION, "n_steps": 20}).json()
    assert len(body["time_steps"]) == 20


# ---------------------------------------------------------------------------
# POST /exit/price-with-timing
# ---------------------------------------------------------------------------


def test_price_with_timing_returns_200() -> None:
    timing_resp = client.post("/api/v1/exit/timing-map", json={"option": _OPTION}).json()
    resp = client.post(
        "/api/v1/exit/price-with-timing",
        json={"option": _OPTION, "timing": timing_resp},
    )
    assert resp.status_code == 200


def test_price_with_timing_expected_value_field() -> None:
    timing_resp = client.post("/api/v1/exit/timing-map", json={"option": _OPTION}).json()
    body = client.post(
        "/api/v1/exit/price-with-timing",
        json={"option": _OPTION, "timing": timing_resp},
    ).json()
    assert "expected_value" in body
    assert isinstance(body["expected_value"], float)


# ---------------------------------------------------------------------------
# POST /exit/compare
# ---------------------------------------------------------------------------


def test_compare_returns_200() -> None:
    resp = client.post("/api/v1/exit/compare", json={"options": [_OPTION, _OPTION_B]})
    assert resp.status_code == 200


def test_compare_returns_list_of_timing_distributions() -> None:
    body = client.post("/api/v1/exit/compare", json={"options": [_OPTION, _OPTION_B]}).json()
    assert isinstance(body, list)
    assert len(body) == 2
    for item in body:
        assert "option_name" in item
        assert "expected_timing" in item
