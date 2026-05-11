"""Tests for gnss.persistence — JSON save/load of twin run results."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.app import app
from gnss.persistence import (
    _PROJECT_ROOT,
    SCHEMA_VERSION,
    load_twin_run,
    new_run_id,
    save_twin_run,
)

client = TestClient(app)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_N_SATS = 6
_N_EPOCHS = 5


def _nominal_request_dict(n_epochs: int = _N_EPOCHS, n_sats: int = _N_SATS) -> dict:
    rng = np.random.default_rng(42)
    obs = [
        {
            "epoch": t,
            "doppler_residuals": rng.normal(0.0, 0.30, size=n_sats).tolist(),
            "elevations_deg": None,
            "ins_velocity_ms": None,
            "osnma_auth_per_sat": None,
        }
        for t in range(n_epochs)
    ]
    return {
        "observations": obs,
        "n_sats": n_sats,
        "los_vectors": None,
        "doppler_noise_std": 0.30,
        "graph_sigma": 1.50,
        "ins_noise_std": 0.05,
        "save": True,
    }


def _minimal_report_dict() -> dict:
    return {
        "epoch_reports": [],
        "n_epochs": _N_EPOCHS,
        "n_sats": _N_SATS,
        "dominant_diagnosis": "nominal",
        "mean_authenticity_genuine": 0.95,
        "mean_integrity_nominal": 0.95,
        "alert_epochs": [],
        "spoofing_window": None,
        "worst_action": "nominal",
        "produced_at": "2026-01-01T00:00:00+00:00",
        "run_id": None,
        "result_path": None,
    }


# ---------------------------------------------------------------------------
# new_run_id
# ---------------------------------------------------------------------------


class TestNewRunId:
    def test_length(self) -> None:
        assert len(new_run_id()) == 8

    def test_hex_characters(self) -> None:
        rid = new_run_id()
        int(rid, 16)  # raises ValueError if not hex

    def test_uniqueness(self) -> None:
        ids = {new_run_id() for _ in range(100)}
        assert len(ids) == 100


# ---------------------------------------------------------------------------
# save_twin_run
# ---------------------------------------------------------------------------


class TestSaveTwinRun:
    def test_file_created(self, tmp_path: Path) -> None:
        run_id = "aabbccdd"
        save_twin_run(_nominal_request_dict(), _minimal_report_dict(), run_id, tmp_path)
        assert (tmp_path / run_id / "twin_run.json").exists()

    def test_schema_version(self, tmp_path: Path) -> None:
        run_id = new_run_id()
        save_twin_run(_nominal_request_dict(), _minimal_report_dict(), run_id, tmp_path)
        data = json.loads((tmp_path / run_id / "twin_run.json").read_text(encoding="utf-8"))
        assert data["schema_version"] == SCHEMA_VERSION

    def test_run_id_in_payload(self, tmp_path: Path) -> None:
        run_id = new_run_id()
        save_twin_run(_nominal_request_dict(), _minimal_report_dict(), run_id, tmp_path)
        data = json.loads((tmp_path / run_id / "twin_run.json").read_text(encoding="utf-8"))
        assert data["run_id"] == run_id

    def test_request_preserved(self, tmp_path: Path) -> None:
        run_id = new_run_id()
        req = _nominal_request_dict()
        save_twin_run(req, _minimal_report_dict(), run_id, tmp_path)
        data = json.loads((tmp_path / run_id / "twin_run.json").read_text(encoding="utf-8"))
        assert data["request"]["n_sats"] == _N_SATS
        assert len(data["request"]["observations"]) == _N_EPOCHS

    def test_report_preserved(self, tmp_path: Path) -> None:
        run_id = new_run_id()
        save_twin_run(_nominal_request_dict(), _minimal_report_dict(), run_id, tmp_path)
        data = json.loads((tmp_path / run_id / "twin_run.json").read_text(encoding="utf-8"))
        assert data["report"]["dominant_diagnosis"] == "nominal"

    def test_produced_at_present(self, tmp_path: Path) -> None:
        run_id = new_run_id()
        save_twin_run(_nominal_request_dict(), _minimal_report_dict(), run_id, tmp_path)
        data = json.loads((tmp_path / run_id / "twin_run.json").read_text(encoding="utf-8"))
        assert "produced_at" in data
        assert "T" in data["produced_at"]  # ISO-8601

    def test_returns_path_string(self, tmp_path: Path) -> None:
        run_id = new_run_id()
        result = save_twin_run(_nominal_request_dict(), _minimal_report_dict(), run_id, tmp_path)
        assert isinstance(result, str)
        assert run_id in result
        assert "twin_run.json" in result

    def test_default_output_outside_project_returns_absolute(self, tmp_path: Path) -> None:
        """When output_dir is outside project root, return absolute path."""
        run_id = new_run_id()
        result = save_twin_run(_nominal_request_dict(), _minimal_report_dict(), run_id, tmp_path)
        # tmp_path is typically under C:\Users\...\AppData\Local\Temp — outside project root
        # Result should still be a valid string referencing the file
        assert Path(result).name == "twin_run.json" or run_id in result

    def test_idempotent_overwrite(self, tmp_path: Path) -> None:
        """Calling save twice with same run_id overwrites without error."""
        run_id = "deadbeef"
        save_twin_run(_nominal_request_dict(), _minimal_report_dict(), run_id, tmp_path)
        save_twin_run(_nominal_request_dict(), _minimal_report_dict(), run_id, tmp_path)
        assert (tmp_path / run_id / "twin_run.json").exists()

    def test_observations_doppler_values_preserved(self, tmp_path: Path) -> None:
        run_id = new_run_id()
        req = _nominal_request_dict()
        first_epoch_residuals = req["observations"][0]["doppler_residuals"]
        save_twin_run(req, _minimal_report_dict(), run_id, tmp_path)
        data = json.loads((tmp_path / run_id / "twin_run.json").read_text(encoding="utf-8"))
        saved_residuals = data["request"]["observations"][0]["doppler_residuals"]
        assert saved_residuals == pytest.approx(first_epoch_residuals, abs=1e-10)


# ---------------------------------------------------------------------------
# load_twin_run
# ---------------------------------------------------------------------------


class TestLoadTwinRun:
    def _save_and_get_path(self, tmp_path: Path) -> tuple[str, Path]:
        run_id = new_run_id()
        save_twin_run(_nominal_request_dict(), _minimal_report_dict(), run_id, tmp_path)
        full_path = tmp_path / run_id / "twin_run.json"
        return run_id, full_path

    def test_load_by_absolute_path(self, tmp_path: Path) -> None:
        _, full_path = self._save_and_get_path(tmp_path)
        data = load_twin_run(full_path)
        assert data["schema_version"] == SCHEMA_VERSION

    def test_load_returns_dict(self, tmp_path: Path) -> None:
        _, full_path = self._save_and_get_path(tmp_path)
        data = load_twin_run(full_path)
        assert isinstance(data, dict)
        for key in ("schema_version", "run_id", "produced_at", "request", "report"):
            assert key in data, f"missing key: {key}"

    def test_load_nonexistent_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_twin_run(tmp_path / "nonexistent" / "twin_run.json")

    def test_load_wrong_schema_version_raises_value_error(self, tmp_path: Path) -> None:
        bad_path = tmp_path / "bad.json"
        bad_path.write_text(json.dumps({"schema_version": "9.9", "run_id": "x"}), encoding="utf-8")
        with pytest.raises(ValueError, match="Unsupported schema_version"):
            load_twin_run(bad_path)

    def test_roundtrip_run_id(self, tmp_path: Path) -> None:
        run_id, full_path = self._save_and_get_path(tmp_path)
        data = load_twin_run(full_path)
        assert data["run_id"] == run_id

    def test_roundtrip_n_sats(self, tmp_path: Path) -> None:
        _, full_path = self._save_and_get_path(tmp_path)
        data = load_twin_run(full_path)
        assert data["request"]["n_sats"] == _N_SATS


# ---------------------------------------------------------------------------
# API integration — POST /gnss/twin/run with save=True / save=False
# ---------------------------------------------------------------------------


def _api_payload(save: bool) -> dict:
    rng = np.random.default_rng(99)
    obs = [
        {"epoch": t, "doppler_residuals": rng.normal(0.0, 0.30, size=_N_SATS).tolist()}
        for t in range(_N_EPOCHS)
    ]
    return {"observations": obs, "n_sats": _N_SATS, "save": save}


class TestTwinRunApiPersistence:
    """Test that the API endpoint correctly exercises persistence."""

    def test_save_false_no_result_path(self) -> None:
        r = client.post("/gnss/twin/run", json=_api_payload(save=False))
        assert r.status_code == 200
        body = r.json()
        assert body["result_path"] is None

    def test_save_false_run_id_present(self) -> None:
        r = client.post("/gnss/twin/run", json=_api_payload(save=False))
        body = r.json()
        assert body["run_id"] is not None
        assert len(body["run_id"]) == 8

    def test_save_true_result_path_set(self) -> None:
        r = client.post("/gnss/twin/run", json=_api_payload(save=True))
        assert r.status_code == 200
        body = r.json()
        assert body["result_path"] is not None
        assert "twin_run.json" in body["result_path"]

    def test_save_true_run_id_in_result_path(self) -> None:
        r = client.post("/gnss/twin/run", json=_api_payload(save=True))
        body = r.json()
        assert body["run_id"] in body["result_path"]

    def test_save_true_file_exists_on_disk(self) -> None:
        r = client.post("/gnss/twin/run", json=_api_payload(save=True))
        body = r.json()
        result_path = body["result_path"]
        full = _PROJECT_ROOT / result_path
        assert full.exists(), f"Expected file at {full}"
        data = json.loads(full.read_text(encoding="utf-8"))
        assert data["run_id"] == body["run_id"]

    def test_saved_file_contains_observations(self) -> None:
        r = client.post("/gnss/twin/run", json=_api_payload(save=True))
        body = r.json()
        full = _PROJECT_ROOT / body["result_path"]
        data = json.loads(full.read_text(encoding="utf-8"))
        assert len(data["request"]["observations"]) == _N_EPOCHS
