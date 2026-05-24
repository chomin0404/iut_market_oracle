"""Tests for /experiments/* endpoints (filesystem operations via tmp_path)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import api.routers.experiments as _exp_router
from api.app import app

client = TestClient(app, raise_server_exceptions=False)

_URL = "/api/v1/experiments"


# ---------------------------------------------------------------------------
# Fixture: redirect EXPERIMENTS_ROOT to an isolated tmp directory per test
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_experiments_root(
    tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:  # noqa: E501
    monkeypatch.setattr(_exp_router, "_EXPERIMENTS_ROOT", str(tmp_path))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_payload(title: str = "Test experiment") -> dict:
    return {
        "title": title,
        "config_path": "configs/test.yaml",
        "random_seed": 42,
        "tags": ["test"],
        "summary": "Unit test run.",
    }


# ---------------------------------------------------------------------------
# POST /experiments
# ---------------------------------------------------------------------------


def test_create_returns_201() -> None:
    resp = client.post(_URL, json=_create_payload())
    assert resp.status_code == 201


def test_create_response_fields() -> None:
    body = client.post(_URL, json=_create_payload()).json()
    assert "experiment_id" in body
    assert "title" in body
    assert "config_path" in body


def test_create_experiment_id_format() -> None:
    body = client.post(_URL, json=_create_payload()).json()
    assert body["experiment_id"].startswith("exp-")


def test_create_missing_title_returns_422() -> None:
    payload = _create_payload()
    del payload["title"]
    resp = client.post(_URL, json=payload)
    assert resp.status_code == 422


def test_create_missing_config_path_returns_422() -> None:
    payload = _create_payload()
    del payload["config_path"]
    resp = client.post(_URL, json=payload)
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /experiments
# ---------------------------------------------------------------------------


def test_list_returns_200() -> None:
    resp = client.get(_URL)
    assert resp.status_code == 200


def test_list_empty_dir_returns_empty_list() -> None:
    body = client.get(_URL).json()
    assert body == []


def test_list_after_create_returns_one() -> None:
    client.post(_URL, json=_create_payload())
    body = client.get(_URL).json()
    assert len(body) == 1


def test_list_multiple_experiments() -> None:
    client.post(_URL, json=_create_payload("Exp A"))
    client.post(_URL, json=_create_payload("Exp B"))
    body = client.get(_URL).json()
    assert len(body) == 2


# ---------------------------------------------------------------------------
# GET /experiments/{exp_id}
# ---------------------------------------------------------------------------


def test_get_existing_experiment() -> None:
    created = client.post(_URL, json=_create_payload()).json()
    exp_id = created["experiment_id"]
    resp = client.get(f"{_URL}/{exp_id}")
    assert resp.status_code == 200
    assert resp.json()["experiment_id"] == exp_id


def test_get_nonexistent_returns_404() -> None:
    resp = client.get(f"{_URL}/exp-999")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# PATCH /experiments/{exp_id}
# ---------------------------------------------------------------------------


def test_patch_summary_returns_200() -> None:
    created = client.post(_URL, json=_create_payload()).json()
    exp_id = created["experiment_id"]
    resp = client.patch(f"{_URL}/{exp_id}", json={"summary": "Updated summary."})
    assert resp.status_code == 200


def test_patch_updates_summary_field() -> None:
    created = client.post(_URL, json=_create_payload()).json()
    exp_id = created["experiment_id"]
    new_summary = "New summary text."
    body = client.patch(f"{_URL}/{exp_id}", json={"summary": new_summary}).json()
    assert body["summary"] == new_summary


def test_patch_nonexistent_returns_400() -> None:
    resp = client.patch(f"{_URL}/exp-999", json={"summary": "x"})
    assert resp.status_code == 400
