"""Tests for POST /api/v1/bayesian/mh/sample and /hmc/sample."""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.app import app

client = TestClient(app, raise_server_exceptions=False)

_MH_URL = "/api/v1/bayesian/mh/sample"
_HMC_URL = "/api/v1/bayesian/hmc/sample"

_MH_BASE = {
    "target": {"type": "normal", "mu": [0.0], "sigma": 1.0},
    "step_size": 0.5,
    "initial": [0.0],
    "n_samples": 50,
    "seed": 0,
}

_HMC_BASE = {
    "target": {"type": "normal", "mu": [0.0], "sigma": 1.0},
    "step_size": 0.3,
    "n_leapfrog": 5,
    "initial": [0.0],
    "n_samples": 50,
    "seed": 0,
}


# ---------------------------------------------------------------------------
# MH — success cases
# ---------------------------------------------------------------------------


def test_mh_returns_200() -> None:
    resp = client.post(_MH_URL, json=_MH_BASE)
    assert resp.status_code == 200


def test_mh_response_shape() -> None:
    resp = client.post(_MH_URL, json=_MH_BASE)
    body = resp.json()
    assert len(body["samples"]) == _MH_BASE["n_samples"]
    assert len(body["samples"][0]) == len(_MH_BASE["target"]["mu"])


def test_mh_response_fields_present() -> None:
    resp = client.post(_MH_URL, json=_MH_BASE)
    body = resp.json()
    for key in ("samples", "acceptance_rate", "n_accepted", "n_total", "diagnostics"):
        assert key in body
    diag = body["diagnostics"]
    for key in ("ess", "r_hat", "trace_summary"):
        assert key in diag


def test_mh_diagnostics_shape() -> None:
    resp = client.post(_MH_URL, json=_MH_BASE)
    diag = resp.json()["diagnostics"]
    dim = len(_MH_BASE["target"]["mu"])
    assert len(diag["ess"]) == dim
    assert all(v > 0 for v in diag["ess"])
    assert len(diag["r_hat"]) == dim
    assert all(v > 0 for v in diag["r_hat"])
    ts = diag["trace_summary"]
    for key in ("mean", "std", "q2_5", "q25", "q50", "q75", "q97_5"):
        assert len(ts[key]) == dim


def test_mh_acceptance_rate_in_unit_interval() -> None:
    resp = client.post(_MH_URL, json=_MH_BASE)
    rate = resp.json()["acceptance_rate"]
    assert 0.0 <= rate <= 1.0


def test_mh_2d_gaussian() -> None:
    payload = {
        "target": {"type": "normal", "mu": [0.0, 0.0], "sigma": 1.0},
        "step_size": 0.5,
        "initial": [0.0, 0.0],
        "n_samples": 30,
        "seed": 1,
    }
    resp = client.post(_MH_URL, json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["samples"][0]) == 2


def test_mh_burn_in_and_thin() -> None:
    payload = {**_MH_BASE, "burn_in": 20, "thin": 3, "n_samples": 10}
    resp = client.post(_MH_URL, json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["samples"]) == 10
    assert body["n_total"] == 20 + 10 * 3


def test_mh_reproducible_with_same_seed() -> None:
    r1 = client.post(_MH_URL, json=_MH_BASE).json()["samples"]
    r2 = client.post(_MH_URL, json=_MH_BASE).json()["samples"]
    assert r1 == r2


# ---------------------------------------------------------------------------
# MH — validation errors
# ---------------------------------------------------------------------------


def test_mh_dim_mismatch_returns_422() -> None:
    payload = {**_MH_BASE, "initial": [0.0, 0.0]}  # mu is 1D, initial is 2D
    resp = client.post(_MH_URL, json=payload)
    assert resp.status_code == 422


def test_mh_nonpositive_step_size_returns_422() -> None:
    payload = {**_MH_BASE, "step_size": 0.0}
    resp = client.post(_MH_URL, json=payload)
    assert resp.status_code == 422


def test_mh_n_samples_zero_returns_422() -> None:
    payload = {**_MH_BASE, "n_samples": 0}
    resp = client.post(_MH_URL, json=payload)
    assert resp.status_code == 422


def test_mh_n_samples_over_limit_returns_422() -> None:
    payload = {**_MH_BASE, "n_samples": 10_001}
    resp = client.post(_MH_URL, json=payload)
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# HMC — success cases
# ---------------------------------------------------------------------------


def test_hmc_returns_200() -> None:
    resp = client.post(_HMC_URL, json=_HMC_BASE)
    assert resp.status_code == 200


def test_hmc_response_shape() -> None:
    resp = client.post(_HMC_URL, json=_HMC_BASE)
    body = resp.json()
    assert len(body["samples"]) == _HMC_BASE["n_samples"]
    assert len(body["samples"][0]) == len(_HMC_BASE["target"]["mu"])


def test_hmc_response_fields_present() -> None:
    resp = client.post(_HMC_URL, json=_HMC_BASE)
    body = resp.json()
    for key in ("samples", "acceptance_rate", "n_accepted", "n_total", "diagnostics"):
        assert key in body
    diag = body["diagnostics"]
    for key in ("ess", "r_hat", "trace_summary"):
        assert key in diag


def test_hmc_diagnostics_shape() -> None:
    resp = client.post(_HMC_URL, json=_HMC_BASE)
    diag = resp.json()["diagnostics"]
    dim = len(_HMC_BASE["target"]["mu"])
    assert len(diag["ess"]) == dim
    assert all(v > 0 for v in diag["ess"])
    assert len(diag["r_hat"]) == dim
    assert all(v > 0 for v in diag["r_hat"])
    ts = diag["trace_summary"]
    for key in ("mean", "std", "q2_5", "q25", "q50", "q75", "q97_5"):
        assert len(ts[key]) == dim


def test_hmc_acceptance_rate_in_unit_interval() -> None:
    resp = client.post(_HMC_URL, json=_HMC_BASE)
    rate = resp.json()["acceptance_rate"]
    assert 0.0 <= rate <= 1.0


def test_hmc_2d_with_mass() -> None:
    payload = {
        "target": {"type": "normal", "mu": [0.0, 0.0], "sigma": 1.0},
        "step_size": 0.2,
        "n_leapfrog": 10,
        "initial": [0.0, 0.0],
        "n_samples": 30,
        "seed": 2,
        "mass": [1.0, 2.0],
    }
    resp = client.post(_HMC_URL, json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["samples"][0]) == 2


def test_hmc_burn_in_and_thin() -> None:
    payload = {**_HMC_BASE, "burn_in": 10, "thin": 2, "n_samples": 10}
    resp = client.post(_HMC_URL, json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["samples"]) == 10
    assert body["n_total"] == 10 + 10 * 2


def test_hmc_reproducible_with_same_seed() -> None:
    r1 = client.post(_HMC_URL, json=_HMC_BASE).json()["samples"]
    r2 = client.post(_HMC_URL, json=_HMC_BASE).json()["samples"]
    assert r1 == r2


# ---------------------------------------------------------------------------
# HMC — validation errors
# ---------------------------------------------------------------------------


def test_hmc_dim_mismatch_returns_422() -> None:
    payload = {**_HMC_BASE, "initial": [0.0, 0.0]}
    resp = client.post(_HMC_URL, json=payload)
    assert resp.status_code == 422


def test_hmc_mass_dim_mismatch_returns_422() -> None:
    payload = {
        **_HMC_BASE,
        "target": {"type": "normal", "mu": [0.0, 0.0], "sigma": 1.0},
        "initial": [0.0, 0.0],
        "mass": [1.0, 1.0, 1.0],  # dim=3 but target is 2D
    }
    resp = client.post(_HMC_URL, json=payload)
    assert resp.status_code == 422


def test_hmc_nonpositive_step_size_returns_422() -> None:
    payload = {**_HMC_BASE, "step_size": -0.1}
    resp = client.post(_HMC_URL, json=payload)
    assert resp.status_code == 422


def test_hmc_zero_n_leapfrog_returns_422() -> None:
    payload = {**_HMC_BASE, "n_leapfrog": 0}
    resp = client.post(_HMC_URL, json=payload)
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# multivariate_normal target — MH
# ---------------------------------------------------------------------------


def test_mh_multivariate_normal_returns_200() -> None:
    payload = {
        "target": {
            "type": "multivariate_normal",
            "mu": [0.0, 0.0],
            "cov": [[2.0, 1.0], [1.0, 2.0]],
        },
        "step_size": 0.8,
        "initial": [0.0, 0.0],
        "n_samples": 50,
        "seed": 0,
    }
    resp = client.post(_MH_URL, json=payload)
    assert resp.status_code == 200


def test_mh_multivariate_normal_shape() -> None:
    payload = {
        "target": {
            "type": "multivariate_normal",
            "mu": [1.0, 2.0, 3.0],
            "cov": [[3.0, 0.5, 0.1], [0.5, 2.0, 0.2], [0.1, 0.2, 1.0]],
        },
        "step_size": 0.5,
        "initial": [0.0, 0.0, 0.0],
        "n_samples": 20,
        "seed": 1,
    }
    resp = client.post(_MH_URL, json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["samples"]) == 20
    assert len(body["samples"][0]) == 3


def test_mh_multivariate_normal_nonsymmetric_cov_returns_422() -> None:
    payload = {
        "target": {
            "type": "multivariate_normal",
            "mu": [0.0, 0.0],
            "cov": [[1.0, 0.5], [0.9, 1.0]],  # not symmetric
        },
        "step_size": 0.5,
        "initial": [0.0, 0.0],
        "n_samples": 10,
        "seed": 0,
    }
    resp = client.post(_MH_URL, json=payload)
    assert resp.status_code == 422


def test_mh_multivariate_normal_indefinite_cov_returns_422() -> None:
    payload = {
        "target": {
            "type": "multivariate_normal",
            "mu": [0.0, 0.0],
            "cov": [[1.0, 2.0], [2.0, 1.0]],  # not positive definite
        },
        "step_size": 0.5,
        "initial": [0.0, 0.0],
        "n_samples": 10,
        "seed": 0,
    }
    resp = client.post(_MH_URL, json=payload)
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# multivariate_normal target — HMC
# ---------------------------------------------------------------------------


def test_hmc_multivariate_normal_returns_200() -> None:
    payload = {
        "target": {
            "type": "multivariate_normal",
            "mu": [1.0, 2.0],
            "cov": [[2.0, 1.0], [1.0, 2.0]],
        },
        "step_size": 0.3,
        "n_leapfrog": 15,
        "initial": [0.0, 0.0],
        "n_samples": 50,
        "seed": 0,
    }
    resp = client.post(_HMC_URL, json=payload)
    assert resp.status_code == 200


def test_hmc_multivariate_normal_shape() -> None:
    payload = {
        "target": {
            "type": "multivariate_normal",
            "mu": [0.0, 0.0],
            "cov": [[2.0, 1.0], [1.0, 2.0]],
        },
        "step_size": 0.3,
        "n_leapfrog": 10,
        "initial": [0.0, 0.0],
        "n_samples": 30,
        "seed": 5,
    }
    resp = client.post(_HMC_URL, json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["samples"]) == 30
    assert len(body["samples"][0]) == 2
