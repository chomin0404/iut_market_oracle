"""Tests for Monte Carlo risk API endpoints.

Covers:
    POST /risk/simulate
    POST /risk/boundary
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
    resp = client.post("/api/v1/risk/simulate", json=_BASE_SIMULATE_BODY)
    assert resp.status_code == 200
    return resp.json()["simulation_id"]


# ---------------------------------------------------------------------------
# POST /risk/simulate
# ---------------------------------------------------------------------------


class TestSimulate:
    def test_success_returns_200(self) -> None:
        resp = client.post("/api/v1/risk/simulate", json=_BASE_SIMULATE_BODY)
        assert resp.status_code == 200

    def test_response_schema(self) -> None:
        resp = client.post("/api/v1/risk/simulate", json=_BASE_SIMULATE_BODY)
        data = resp.json()
        assert "simulation_id" in data
        assert data["n_samples"] == _BASE_SIMULATE_BODY["n_samples"]
        assert "summary" in data
        for key in ("mean", "std", "var_95", "es_95"):
            assert key in data["summary"], f"missing {key}"
            assert isinstance(data["summary"][key], float)

    def test_simulation_id_is_uuid(self) -> None:
        import uuid

        resp = client.post("/api/v1/risk/simulate", json=_BASE_SIMULATE_BODY)
        uid = resp.json()["simulation_id"]
        uuid.UUID(uid)  # raises ValueError if not valid UUID

    def test_deterministic_with_seed(self) -> None:
        r1 = client.post("/api/v1/risk/simulate", json=_BASE_SIMULATE_BODY).json()["summary"]
        r2 = client.post("/api/v1/risk/simulate", json=_BASE_SIMULATE_BODY).json()["summary"]
        assert math.isclose(r1["mean"], r2["mean"], rel_tol=1e-9)
        assert math.isclose(r1["var_95"], r2["var_95"], rel_tol=1e-9)

    def test_var_95_less_than_es_95(self) -> None:
        resp = client.post("/api/v1/risk/simulate", json=_BASE_SIMULATE_BODY)
        summary = resp.json()["summary"]
        assert summary["es_95"] >= summary["var_95"]

    def test_distributions_count_mismatch_returns_422(self) -> None:
        body = {**_BASE_SIMULATE_BODY, "n_vars": 3}
        resp = client.post("/api/v1/risk/simulate", json=body)
        assert resp.status_code == 422

    def test_unsupported_copula_returns_400(self) -> None:
        body = {
            **_BASE_SIMULATE_BODY,
            "copula": {"type": "frank"},
        }
        resp = client.post("/api/v1/risk/simulate", json=body)
        assert resp.status_code == 400

    def test_student_t_copula_returns_200(self) -> None:
        body = {
            **_BASE_SIMULATE_BODY,
            "copula": {"type": "student_t", "corr_matrix": [[1, 0.6], [0.6, 1]], "df": 5.0},
        }
        resp = client.post("/api/v1/risk/simulate", json=body)
        assert resp.status_code == 200

    def test_clayton_copula_returns_200(self) -> None:
        body = {
            "n_vars": 2,
            "n_samples": 1000,
            "distributions": [
                {"name": "normal", "params": {"loc": 0.0, "scale": 1.0}},
                {"name": "normal", "params": {"loc": 0.0, "scale": 1.0}},
            ],
            "copula": {"type": "clayton", "theta": 2.0},
            "seed": 7,
        }
        resp = client.post("/api/v1/risk/simulate", json=body)
        assert resp.status_code == 200

    def test_independent_copula_returns_200(self) -> None:
        body = {
            "n_vars": 2,
            "n_samples": 1000,
            "distributions": [
                {"name": "normal", "params": {"loc": 0.0, "scale": 1.0}},
                {"name": "normal", "params": {"loc": 0.0, "scale": 1.0}},
            ],
            "copula": {"type": "independent"},
            "seed": 8,
        }
        resp = client.post("/api/v1/risk/simulate", json=body)
        assert resp.status_code == 200

    def test_student_t_missing_df_returns_422(self) -> None:
        body = {
            **_BASE_SIMULATE_BODY,
            "copula": {"type": "student_t", "corr_matrix": [[1, 0.6], [0.6, 1]]},
        }
        resp = client.post("/api/v1/risk/simulate", json=body)
        assert resp.status_code == 422

    def test_clayton_missing_theta_returns_422(self) -> None:
        body = {
            "n_vars": 2,
            "n_samples": 1000,
            "distributions": [
                {"name": "normal", "params": {"loc": 0.0, "scale": 1.0}},
                {"name": "normal", "params": {"loc": 0.0, "scale": 1.0}},
            ],
            "copula": {"type": "clayton"},
            "seed": 0,
        }
        resp = client.post("/api/v1/risk/simulate", json=body)
        assert resp.status_code == 422

    def test_unsupported_distribution_returns_400(self) -> None:
        body = {
            **_BASE_SIMULATE_BODY,
            "distributions": [
                {"name": "cauchy_unknown", "params": {}},
                {"name": "normal", "params": {"loc": 0.0, "scale": 1.0}},
            ],
        }
        resp = client.post("/api/v1/risk/simulate", json=body)
        assert resp.status_code == 400

    def test_n_samples_too_large_returns_422(self) -> None:
        body = {**_BASE_SIMULATE_BODY, "n_samples": 200_000}
        resp = client.post("/api/v1/risk/simulate", json=body)
        assert resp.status_code == 422

    def test_gev_distribution_works(self) -> None:
        body = {
            "n_vars": 1,
            "n_samples": 1000,
            "distributions": [{"name": "gev", "params": {"c": 0.2, "loc": 10.0, "scale": 5.0}}],
            "copula": {"type": "gaussian", "corr_matrix": [[1.0]]},
            "seed": 1,
        }
        resp = client.post("/api/v1/risk/simulate", json=body)
        assert resp.status_code == 200
        assert resp.json()["summary"]["mean"] > 0


# ---------------------------------------------------------------------------
# POST /risk/boundary
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

    def test_bootstrap_reproducible_with_seed(self, simulation_id: str) -> None:
        body = {
            "simulation_id": simulation_id,
            "thresholds": [1.0, 2.0],
            "bootstrap_n": 200,
            "bootstrap_seed": 42,
        }
        r1 = client.post("/api/v1/risk/boundary", json=body).json()
        r2 = client.post("/api/v1/risk/boundary", json=body).json()
        assert r1["confidence_band"]["lower"] == r2["confidence_band"]["lower"]
        assert r1["confidence_band"]["upper"] == r2["confidence_band"]["upper"]


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


# ---------------------------------------------------------------------------
# POST /risk/tail
# ---------------------------------------------------------------------------

_ALPHAS = [0.90, 0.95, 0.99]


class TestTailEndpoint:
    def test_success_from_simulation_id(self, simulation_id: str) -> None:
        body = {
            "simulation_id": simulation_id,
            "target_variable_index": 0,
            "alphas": _ALPHAS,
        }
        resp = client.post("/api/v1/risk/tail", json=body)
        assert resp.status_code == 200

    def test_response_schema(self, simulation_id: str) -> None:
        body = {"simulation_id": simulation_id, "alphas": _ALPHAS}
        data = client.post("/api/v1/risk/tail", json=body).json()
        assert data["n_samples"] == _BASE_SIMULATE_BODY["n_samples"]
        assert len(data["tail_stats"]) == len(_ALPHAS)
        for entry in data["tail_stats"]:
            assert "alpha" in entry
            assert "var" in entry
            assert "es" in entry

    def test_sorted_by_alpha(self, simulation_id: str) -> None:
        body = {"simulation_id": simulation_id, "alphas": [0.99, 0.90, 0.95]}
        data = client.post("/api/v1/risk/tail", json=body).json()
        returned_alphas = [e["alpha"] for e in data["tail_stats"]]
        assert returned_alphas == sorted(returned_alphas)

    def test_es_ge_var_at_each_alpha(self, simulation_id: str) -> None:
        body = {"simulation_id": simulation_id, "alphas": _ALPHAS}
        data = client.post("/api/v1/risk/tail", json=body).json()
        for entry in data["tail_stats"]:
            assert entry["es"] >= entry["var"], f"ES < VaR at alpha={entry['alpha']}"

    def test_var_non_decreasing_with_alpha(self, simulation_id: str) -> None:
        body = {"simulation_id": simulation_id, "alphas": [0.80, 0.90, 0.95, 0.99]}
        data = client.post("/api/v1/risk/tail", json=body).json()
        vars_ = [e["var"] for e in data["tail_stats"]]
        for v1, v2 in zip(vars_, vars_[1:]):
            assert v1 <= v2 + 1e-9, "VaR must be non-decreasing with increasing alpha"

    def test_from_raw_samples(self) -> None:
        samples = list(range(1, 101))  # [1, 2, ..., 100]
        body = {"samples": samples, "alphas": [0.90, 0.95]}
        resp = client.post("/api/v1/risk/tail", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert data["n_samples"] == 100
        # 90th percentile of [1..100] ≈ 90
        assert math.isclose(data["tail_stats"][0]["var"], 90.1, rel_tol=0.05)

    def test_unknown_simulation_id_returns_404(self) -> None:
        body = {
            "simulation_id": "00000000-0000-0000-0000-000000000000",
            "alphas": [0.95],
        }
        resp = client.post("/api/v1/risk/tail", json=body)
        assert resp.status_code == 404

    def test_out_of_range_variable_index_returns_400(self, simulation_id: str) -> None:
        body = {
            "simulation_id": simulation_id,
            "target_variable_index": 99,
            "alphas": [0.95],
        }
        resp = client.post("/api/v1/risk/tail", json=body)
        assert resp.status_code == 400

    def test_no_source_returns_422(self) -> None:
        resp = client.post("/api/v1/risk/tail", json={"alphas": [0.95]})
        assert resp.status_code == 422

    def test_alpha_out_of_range_returns_422(self, simulation_id: str) -> None:
        body = {"simulation_id": simulation_id, "alphas": [1.5]}
        resp = client.post("/api/v1/risk/tail", json=body)
        assert resp.status_code == 422

    def test_error_response_has_error_field(self) -> None:
        body = {
            "simulation_id": "00000000-0000-0000-0000-000000000000",
            "alphas": [0.95],
        }
        data = client.post("/api/v1/risk/tail", json=body).json()
        assert "error" in data
        assert "detail" in data


# ---------------------------------------------------------------------------
# POST /risk/gnss_scenario
# ---------------------------------------------------------------------------

_GNSS_BODY = {
    "scenario_name": "multipath_and_spoofing",
    "variables": [
        {
            "name": "position_error_m",
            "distribution": {"name": "lognormal", "params": {"s": 0.8, "loc": 0.0, "scale": 2.0}},
        },
        {
            "name": "cn0_degradation_db",
            "distribution": {"name": "normal", "params": {"loc": 3.0, "scale": 1.5}},
        },
        {
            "name": "time_error_ns",
            "distribution": {"name": "gev", "params": {"c": 0.2, "loc": 10.0, "scale": 5.0}},
        },
    ],
    "copula": {
        "type": "student_t",
        "corr_matrix": [[1.0, 0.7, 0.4], [0.7, 1.0, 0.3], [0.4, 0.3, 1.0]],
        "df": 4.0,
    },
    "n_samples": 3000,
    "seed": 42,
    "alphas": [0.90, 0.95, 0.99],
}


class TestGnssScenario:
    def test_success_returns_200(self) -> None:
        resp = client.post("/api/v1/risk/gnss_scenario", json=_GNSS_BODY)
        assert resp.status_code == 200

    def test_response_schema(self) -> None:
        data = client.post("/api/v1/risk/gnss_scenario", json=_GNSS_BODY).json()
        assert data["scenario_name"] == _GNSS_BODY["scenario_name"]
        assert data["n_samples"] == _GNSS_BODY["n_samples"]
        assert len(data["per_variable"]) == len(_GNSS_BODY["variables"])
        for entry in data["per_variable"]:
            assert "name" in entry
            assert "mean" in entry
            assert "std" in entry
            assert "tail_stats" in entry
            assert len(entry["tail_stats"]) == len(_GNSS_BODY["alphas"])

    def test_variable_names_preserved(self) -> None:
        data = client.post("/api/v1/risk/gnss_scenario", json=_GNSS_BODY).json()
        names = [v["name"] for v in data["per_variable"]]
        expected = [v["name"] for v in _GNSS_BODY["variables"]]
        assert names == expected

    def test_tail_stats_sorted_by_alpha(self) -> None:
        body = {**_GNSS_BODY, "alphas": [0.99, 0.90, 0.95]}
        data = client.post("/api/v1/risk/gnss_scenario", json=body).json()
        for entry in data["per_variable"]:
            alphas = [s["alpha"] for s in entry["tail_stats"]]
            assert alphas == sorted(alphas)

    def test_es_ge_var_all_variables(self) -> None:
        data = client.post("/api/v1/risk/gnss_scenario", json=_GNSS_BODY).json()
        for entry in data["per_variable"]:
            for stat in entry["tail_stats"]:
                assert stat["es"] >= stat["var"], (
                    f"ES < VaR for {entry['name']} at alpha={stat['alpha']}"
                )

    def test_var_non_decreasing_with_alpha(self) -> None:
        body = {**_GNSS_BODY, "alphas": [0.80, 0.90, 0.95, 0.99]}
        data = client.post("/api/v1/risk/gnss_scenario", json=body).json()
        for entry in data["per_variable"]:
            vars_ = [s["var"] for s in entry["tail_stats"]]
            for v1, v2 in zip(vars_, vars_[1:]):
                assert v1 <= v2 + 1e-9, f"VaR not non-decreasing for {entry['name']}"

    def test_deterministic_with_seed(self) -> None:
        r1 = client.post("/api/v1/risk/gnss_scenario", json=_GNSS_BODY).json()
        r2 = client.post("/api/v1/risk/gnss_scenario", json=_GNSS_BODY).json()
        for e1, e2 in zip(r1["per_variable"], r2["per_variable"]):
            assert math.isclose(e1["mean"], e2["mean"], rel_tol=1e-9)
            for s1, s2 in zip(e1["tail_stats"], e2["tail_stats"]):
                assert math.isclose(s1["var"], s2["var"], rel_tol=1e-9)

    def test_independent_copula_works(self) -> None:
        body = {
            "scenario_name": "nominal_baseline",
            "variables": [
                {
                    "name": "position_error_m",
                    "distribution": {
                        "name": "lognormal",
                        "params": {"s": 0.3, "loc": 0.0, "scale": 1.0},
                    },
                },
            ],
            "copula": {"type": "independent"},
            "n_samples": 1000,
            "seed": 0,
            "alphas": [0.95],
        }
        resp = client.post("/api/v1/risk/gnss_scenario", json=body)
        assert resp.status_code == 200
        assert resp.json()["per_variable"][0]["name"] == "position_error_m"

    def test_default_alphas_used_when_omitted(self) -> None:
        body = {k: v for k, v in _GNSS_BODY.items() if k != "alphas"}
        data = client.post("/api/v1/risk/gnss_scenario", json=body).json()
        # Default alphas: [0.90, 0.95, 0.99]
        for entry in data["per_variable"]:
            assert len(entry["tail_stats"]) == 3
            assert [s["alpha"] for s in entry["tail_stats"]] == [0.90, 0.95, 0.99]

    def test_corr_matrix_row_mismatch_returns_422(self) -> None:
        body = {
            **_GNSS_BODY,
            "copula": {
                "type": "student_t",
                "corr_matrix": [[1.0, 0.5], [0.5, 1.0]],  # 2×2 for 3 variables
                "df": 4.0,
            },
        }
        resp = client.post("/api/v1/risk/gnss_scenario", json=body)
        assert resp.status_code == 422

    def test_unsupported_distribution_returns_400(self) -> None:
        body = {
            **_GNSS_BODY,
            "variables": [
                {
                    "name": "position_error_m",
                    "distribution": {"name": "unknown_dist", "params": {}},
                },
                *_GNSS_BODY["variables"][1:],
            ],
        }
        resp = client.post("/api/v1/risk/gnss_scenario", json=body)
        assert resp.status_code == 400

    def test_alpha_out_of_range_returns_422(self) -> None:
        body = {**_GNSS_BODY, "alphas": [0.95, 1.5]}
        resp = client.post("/api/v1/risk/gnss_scenario", json=body)
        assert resp.status_code == 422

    def test_n_samples_too_large_returns_422(self) -> None:
        body = {**_GNSS_BODY, "n_samples": 200_000}
        resp = client.post("/api/v1/risk/gnss_scenario", json=body)
        assert resp.status_code == 422

    def test_lognormal_position_error_positive_mean(self) -> None:
        """lognormal は正値のみ取るため mean > 0 であることを確認。"""
        data = client.post("/api/v1/risk/gnss_scenario", json=_GNSS_BODY).json()
        pos = next(v for v in data["per_variable"] if v["name"] == "position_error_m")
        assert pos["mean"] > 0.0
        assert pos["tail_stats"][-1]["var"] > 0.0  # 99th percentile positive


# ---------------------------------------------------------------------------
# POST /risk/sensitivity
# ---------------------------------------------------------------------------

# Base body: sweep distribution_0.scale from 1→3 (VaR is guaranteed to increase)
_SENSITIVITY_SCALE_BODY = {
    "base_config": {
        "n_vars": 2,
        "n_samples": 2000,
        "distributions": [
            {"name": "normal", "params": {"loc": 0.0, "scale": 1.0}},
            {"name": "lognormal", "params": {"s": 0.8, "loc": 0.0, "scale": 1.0}},
        ],
        "copula": {"type": "gaussian", "corr_matrix": [[1, 0.6], [0.6, 1]]},
        "seed": 42,
    },
    "sweep_parameter": {
        "target": "distribution_0",
        "param_name": "scale",
        "values": [0.5, 1.0, 1.5, 2.0, 3.0],
    },
    "risk_metric": "var_95",
    "target_variable_index": 0,
}

# Base body: sweep corr_matrix_off_diagonal (response validity check)
_SENSITIVITY_CORR_BODY = {
    "base_config": {
        "n_vars": 2,
        "n_samples": 2000,
        "distributions": [
            {"name": "normal", "params": {"loc": 0.0, "scale": 1.0}},
            {"name": "lognormal", "params": {"s": 0.8, "loc": 0.0, "scale": 1.0}},
        ],
        "copula": {"type": "gaussian", "corr_matrix": [[1, 0.0], [0.0, 1]]},
        "seed": 42,
    },
    "sweep_parameter": {
        "target": "copula",
        "param_name": "corr_matrix_off_diagonal",
        "values": [0.0, 0.3, 0.6, 0.9],
    },
    "risk_metric": "var_95",
    "target_variable_index": 0,
}


class TestRiskSensitivity:
    def test_success_returns_200(self) -> None:
        resp = client.post("/api/v1/risk/sensitivity", json=_SENSITIVITY_SCALE_BODY)
        assert resp.status_code == 200

    def test_response_schema(self) -> None:
        resp = client.post("/api/v1/risk/sensitivity", json=_SENSITIVITY_SCALE_BODY)
        data = resp.json()
        for key in ("parameter_values", "risk_values", "sensitivity_index", "most_sensitive_at"):
            assert key in data, f"missing key: {key}"

    def test_lengths_match(self) -> None:
        resp = client.post("/api/v1/risk/sensitivity", json=_SENSITIVITY_SCALE_BODY)
        data = resp.json()
        n = len(_SENSITIVITY_SCALE_BODY["sweep_parameter"]["values"])
        assert len(data["parameter_values"]) == n
        assert len(data["risk_values"]) == n

    def test_scale_sweep_var_increases(self) -> None:
        """VaR of normal(0, scale) must increase as scale grows (mathematically guaranteed)."""
        resp = client.post("/api/v1/risk/sensitivity", json=_SENSITIVITY_SCALE_BODY)
        risk = resp.json()["risk_values"]
        for prev, nxt in zip(risk, risk[1:]):
            assert prev < nxt + 1e-9, "VaR should be non-decreasing with scale"

    def test_sensitivity_index_non_negative(self) -> None:
        resp = client.post("/api/v1/risk/sensitivity", json=_SENSITIVITY_SCALE_BODY)
        assert resp.json()["sensitivity_index"] >= 0.0

    def test_most_sensitive_at_in_values(self) -> None:
        resp = client.post("/api/v1/risk/sensitivity", json=_SENSITIVITY_SCALE_BODY)
        data = resp.json()
        assert data["most_sensitive_at"] in data["parameter_values"]

    def test_corr_sweep_response_valid(self) -> None:
        """corr_matrix_off_diagonal sweep must return valid response with correct length."""
        resp = client.post("/api/v1/risk/sensitivity", json=_SENSITIVITY_CORR_BODY)
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["parameter_values"]) == len(data["risk_values"]) == 4
        for v in data["risk_values"]:
            assert math.isfinite(v)

    def test_es_metric_works(self) -> None:
        body = {**_SENSITIVITY_SCALE_BODY, "risk_metric": "es_95"}
        resp = client.post("/api/v1/risk/sensitivity", json=body)
        assert resp.status_code == 200

    def test_var_99_metric_works(self) -> None:
        body = {**_SENSITIVITY_SCALE_BODY, "risk_metric": "var_99"}
        resp = client.post("/api/v1/risk/sensitivity", json=body)
        assert resp.status_code == 200

    def test_invalid_risk_metric_returns_422(self) -> None:
        body = {**_SENSITIVITY_SCALE_BODY, "risk_metric": "cvar_90"}
        resp = client.post("/api/v1/risk/sensitivity", json=body)
        assert resp.status_code == 422

    def test_invalid_copula_target_returns_400(self) -> None:
        body = {
            **_SENSITIVITY_CORR_BODY,
            "sweep_parameter": {
                "target": "copula",
                "param_name": "unknown_param",
                "values": [0.5, 1.0],
            },
        }
        resp = client.post("/api/v1/risk/sensitivity", json=body)
        assert resp.status_code == 400

    def test_invalid_distribution_target_returns_400(self) -> None:
        body = {
            **_SENSITIVITY_SCALE_BODY,
            "sweep_parameter": {
                "target": "completely_wrong",
                "param_name": "scale",
                "values": [1.0, 2.0],
            },
        }
        resp = client.post("/api/v1/risk/sensitivity", json=body)
        assert resp.status_code == 400

    def test_out_of_range_variable_index_returns_400(self) -> None:
        body = {**_SENSITIVITY_SCALE_BODY, "target_variable_index": 99}
        resp = client.post("/api/v1/risk/sensitivity", json=body)
        assert resp.status_code == 400
