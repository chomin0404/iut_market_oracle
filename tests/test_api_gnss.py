"""HTTP endpoint tests for GNSS Resilience Twin API."""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.app import app

client = TestClient(app)

# ---------------------------------------------------------------------------
# Minimal payloads (small MC counts for speed)
# ---------------------------------------------------------------------------

_RESILIENCE_PAYLOAD = {
    "n_mc": 8,
    "n_epochs": 20,
    "n_sats": 6,
    "doppler_noise_std": 0.30,
    "spoof_bias_std": 2.50,
    "spoof_diff_std": 0.80,
    "graph_sigma": 1.50,
    "dirichlet_alpha": 2.0,
    "random_seed": 0,
}

_SPOOF_SIM_PAYLOAD = {
    "n_mc": 10,
    "n_epochs": 30,
    "n_sats": 6,
    "random_seed": 0,
}

_MULTI_SENSOR_PAYLOAD = {
    "T": 40,
    "attack_start": 15,
    "attack_end": 30,
    "n_nominal": 5,
    "n_attack": 5,
    "random_seed": 0,
}


# ---------------------------------------------------------------------------
# POST /gnss/resilience-sim  (T1500 flagship)
# ---------------------------------------------------------------------------


class TestResilienceSim:
    def test_status_200(self) -> None:
        r = client.post("/api/v1/gnss/resilience-sim", json=_RESILIENCE_PAYLOAD)
        assert r.status_code == 200

    def test_response_schema(self) -> None:
        r = client.post("/api/v1/gnss/resilience-sim", json=_RESILIENCE_PAYLOAD)
        body = r.json()
        for field in (
            "p_detection",
            "p_false_alarm",
            "auc",
            "per_class_accuracy",
            "confusion_matrix",
            "mean_confidence",
            "n_mc",
            "n_mc_per_class",
        ):
            assert field in body, f"missing field: {field}"

    def test_confusion_matrix_is_4x4(self) -> None:
        r = client.post("/api/v1/gnss/resilience-sim", json=_RESILIENCE_PAYLOAD)
        cm = r.json()["confusion_matrix"]
        assert len(cm) == 4
        assert all(len(row) == 4 for row in cm)

    def test_per_class_accuracy_keys(self) -> None:
        r = client.post("/api/v1/gnss/resilience-sim", json=_RESILIENCE_PAYLOAD)
        keys = set(r.json()["per_class_accuracy"].keys())
        assert keys == {"nominal", "multipath", "hardware_fault", "spoofing"}

    def test_metrics_in_range(self) -> None:
        r = client.post("/api/v1/gnss/resilience-sim", json=_RESILIENCE_PAYLOAD)
        body = r.json()
        assert 0.0 <= body["p_detection"] <= 1.0
        assert 0.0 <= body["p_false_alarm"] <= 1.0
        assert 0.0 <= body["auc"] <= 1.0
        assert 0.0 <= body["mean_confidence"] <= 1.0

    def test_n_mc_reflected(self) -> None:
        r = client.post("/api/v1/gnss/resilience-sim", json=_RESILIENCE_PAYLOAD)
        assert r.json()["n_mc"] == _RESILIENCE_PAYLOAD["n_mc"]

    def test_n_mc_below_minimum_returns_422(self) -> None:
        payload = {**_RESILIENCE_PAYLOAD, "n_mc": 2}
        r = client.post("/api/v1/gnss/resilience-sim", json=payload)
        assert r.status_code == 422

    def test_n_mc_above_maximum_returns_422(self) -> None:
        payload = {**_RESILIENCE_PAYLOAD, "n_mc": 9999}
        r = client.post("/api/v1/gnss/resilience-sim", json=payload)
        assert r.status_code == 422

    def test_invalid_doppler_noise_returns_422(self) -> None:
        payload = {**_RESILIENCE_PAYLOAD, "doppler_noise_std": -1.0}
        r = client.post("/api/v1/gnss/resilience-sim", json=payload)
        assert r.status_code == 422

    def test_reproducibility(self) -> None:
        r1 = client.post("/api/v1/gnss/resilience-sim", json=_RESILIENCE_PAYLOAD)
        r2 = client.post("/api/v1/gnss/resilience-sim", json=_RESILIENCE_PAYLOAD)
        assert r1.json()["auc"] == r2.json()["auc"]
        assert r1.json()["confusion_matrix"] == r2.json()["confusion_matrix"]


# ---------------------------------------------------------------------------
# POST /gnss/spoof-sim  (T1300)
# ---------------------------------------------------------------------------


class TestSpoofSim:
    def test_status_200(self) -> None:
        r = client.post("/api/v1/gnss/spoof-sim", json=_SPOOF_SIM_PAYLOAD)
        assert r.status_code == 200

    def test_response_has_auc(self) -> None:
        r = client.post("/api/v1/gnss/spoof-sim", json=_SPOOF_SIM_PAYLOAD)
        assert "auc" in r.json()


# ---------------------------------------------------------------------------
# POST /gnss/multi-sensor-sim  (T1350)
# ---------------------------------------------------------------------------


class TestMultiSensorSim:
    def test_status_200(self) -> None:
        r = client.post("/api/v1/gnss/multi-sensor-sim", json=_MULTI_SENSOR_PAYLOAD)
        assert r.status_code == 200

    def test_response_has_auc(self) -> None:
        r = client.post("/api/v1/gnss/multi-sensor-sim", json=_MULTI_SENSOR_PAYLOAD)
        assert "auc" in r.json()


# ---------------------------------------------------------------------------
# POST /gnss/twin/run  (T1500 probabilistic digital twin)
# ---------------------------------------------------------------------------

_N_SATS = 6
_N_EPOCHS = 10


def _make_nominal_obs(n_epochs: int = _N_EPOCHS, n_sats: int = _N_SATS) -> list[dict]:
    """Generate nominal (zero-mean Gaussian) Doppler observations."""
    rng = __import__("numpy").random.default_rng(99)
    return [
        {
            "epoch": t,
            "doppler_residuals": rng.normal(0.0, 0.30, size=n_sats).tolist(),
        }
        for t in range(n_epochs)
    ]


def _make_spoofed_obs(n_epochs: int = _N_EPOCHS, n_sats: int = _N_SATS) -> list[dict]:
    """Generate observations with a large common meaconing bias (spoofing)."""
    rng = __import__("numpy").random.default_rng(7)
    obs = []
    for t in range(n_epochs):
        bias = 5.0  # large common bias [Hz]
        residuals = (rng.normal(0.0, 0.30, size=n_sats) + bias).tolist()
        obs.append({"epoch": t, "doppler_residuals": residuals})
    return obs


_TWIN_PAYLOAD = {
    "observations": _make_nominal_obs(),
    "n_sats": _N_SATS,
}

_TWIN_SPOOF_PAYLOAD = {
    "observations": _make_spoofed_obs(),
    "n_sats": _N_SATS,
}


class TestTwinRun:
    def test_status_200(self) -> None:
        r = client.post("/api/v1/gnss/twin/run", json=_TWIN_PAYLOAD)
        assert r.status_code == 200

    def test_response_top_level_fields(self) -> None:
        r = client.post("/api/v1/gnss/twin/run", json=_TWIN_PAYLOAD)
        body = r.json()
        for field in (
            "epoch_reports",
            "n_epochs",
            "n_sats",
            "dominant_diagnosis",
            "mean_authenticity_genuine",
            "mean_integrity_nominal",
            "alert_epochs",
            "spoofing_window",
            "worst_action",
        ):
            assert field in body, f"missing top-level field: {field}"

    def test_epoch_reports_count(self) -> None:
        r = client.post("/api/v1/gnss/twin/run", json=_TWIN_PAYLOAD)
        body = r.json()
        assert body["n_epochs"] == _N_EPOCHS
        assert len(body["epoch_reports"]) == _N_EPOCHS

    def test_epoch_report_fields(self) -> None:
        r = client.post("/api/v1/gnss/twin/run", json=_TWIN_PAYLOAD)
        report = r.json()["epoch_reports"][0]
        for field in (
            "epoch",
            "authenticity",
            "integrity",
            "fault_posterior",
            "diagnosis",
            "confidence",
            "recommended_action",
            "action_reason",
            "entropy_alert",
            "gmm_n_fault",
            "imm_spoof_weight",
            "spectral_fiedler_ratio",
        ):
            assert field in report, f"missing epoch_report field: {field}"

    def test_authenticity_sums_to_one(self) -> None:
        r = client.post("/api/v1/gnss/twin/run", json=_TWIN_PAYLOAD)
        for rep in r.json()["epoch_reports"]:
            total = rep["authenticity"]["genuine"] + rep["authenticity"]["spoofed"]
            assert abs(total - 1.0) < 1e-6, f"authenticity does not sum to 1: {total}"

    def test_integrity_sums_to_one(self) -> None:
        r = client.post("/api/v1/gnss/twin/run", json=_TWIN_PAYLOAD)
        for rep in r.json()["epoch_reports"]:
            total = rep["integrity"]["nominal"] + rep["integrity"]["degraded"]
            assert abs(total - 1.0) < 1e-6, f"integrity does not sum to 1: {total}"

    def test_fault_posterior_sums_to_one(self) -> None:
        r = client.post("/api/v1/gnss/twin/run", json=_TWIN_PAYLOAD)
        for rep in r.json()["epoch_reports"]:
            total = sum(rep["fault_posterior"].values())
            assert abs(total - 1.0) < 1e-5, f"fault_posterior does not sum to 1: {total}"

    def test_fault_posterior_keys(self) -> None:
        r = client.post("/api/v1/gnss/twin/run", json=_TWIN_PAYLOAD)
        keys = set(r.json()["epoch_reports"][0]["fault_posterior"].keys())
        assert keys == {"nominal", "multipath", "hardware_fault", "spoofing"}

    def test_recommended_action_values(self) -> None:
        valid = {"nominal", "monitor", "reduce_trust", "switch_source", "ground_immediately"}
        r = client.post("/api/v1/gnss/twin/run", json=_TWIN_PAYLOAD)
        for rep in r.json()["epoch_reports"]:
            assert rep["recommended_action"] in valid

    def test_nominal_signal_dominant_diagnosis(self) -> None:
        r = client.post("/api/v1/gnss/twin/run", json=_TWIN_PAYLOAD)
        assert r.json()["dominant_diagnosis"] == "nominal"

    def test_spoofed_signal_raises_severity(self) -> None:
        r = client.post("/api/v1/gnss/twin/run", json=_TWIN_SPOOF_PAYLOAD)
        body = r.json()
        # Under heavy spoofing bias the worst action must escalate beyond NOMINAL
        assert body["worst_action"] != "nominal"

    def test_no_spoofing_window_for_nominal(self) -> None:
        r = client.post("/api/v1/gnss/twin/run", json=_TWIN_PAYLOAD)
        assert r.json()["spoofing_window"] is None

    def test_custom_los_vectors(self) -> None:
        import math

        n = _N_SATS
        los = []
        for i in range(n):
            theta = math.pi * i / n
            los.append([math.sin(theta), 0.0, math.cos(theta)])
        payload = {**_TWIN_PAYLOAD, "los_vectors": los}
        r = client.post("/api/v1/gnss/twin/run", json=payload)
        assert r.status_code == 200

    def test_elevations_deg_accepted(self) -> None:
        obs = [
            {
                "epoch": t,
                "doppler_residuals": [0.0] * _N_SATS,
                "elevations_deg": [30.0 + 5 * i for i in range(_N_SATS)],
            }
            for t in range(_N_EPOCHS)
        ]
        payload = {"observations": obs, "n_sats": _N_SATS}
        r = client.post("/api/v1/gnss/twin/run", json=payload)
        assert r.status_code == 200

    def test_wrong_n_sats_returns_422(self) -> None:
        payload = {**_TWIN_PAYLOAD, "n_sats": _N_SATS + 1}
        r = client.post("/api/v1/gnss/twin/run", json=payload)
        assert r.status_code == 422

    def test_too_few_observations_returns_422(self) -> None:
        payload = {**_TWIN_PAYLOAD, "observations": _TWIN_PAYLOAD["observations"][:1]}
        r = client.post("/api/v1/gnss/twin/run", json=payload)
        assert r.status_code == 422

    def test_reproducibility(self) -> None:
        r1 = client.post("/api/v1/gnss/twin/run", json=_TWIN_PAYLOAD)
        r2 = client.post("/api/v1/gnss/twin/run", json=_TWIN_PAYLOAD)
        assert r1.json()["epoch_reports"] == r2.json()["epoch_reports"]
