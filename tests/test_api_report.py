"""Tests for POST /report/run (filesystem operations mocked)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from api.app import app

client = TestClient(app, raise_server_exceptions=False)

_URL = "/api/v1/report/run"


# ---------------------------------------------------------------------------
# Mocked success
# ---------------------------------------------------------------------------


def test_report_run_returns_200() -> None:
    mock_artifacts = {
        "dcf_chart": Path("reports/exp-001/dcf_chart.png"),
        "summary_md": Path("reports/exp-001/summary.md"),
    }
    with patch("api.routers.report.run_report", return_value=mock_artifacts):
        resp = client.post(_URL, json={})
    assert resp.status_code == 200


def test_report_run_artifacts_field() -> None:
    mock_artifacts = {"chart": Path("reports/exp-001/chart.png")}
    with patch("api.routers.report.run_report", return_value=mock_artifacts):
        body = client.post(_URL, json={}).json()
    assert "artifacts" in body
    assert isinstance(body["artifacts"], dict)


def test_report_run_artifacts_values_are_strings() -> None:
    mock_artifacts = {"a": Path("reports/x.png"), "b": Path("reports/y.md")}
    with patch("api.routers.report.run_report", return_value=mock_artifacts):
        body = client.post(_URL, json={}).json()
    for v in body["artifacts"].values():
        assert isinstance(v, str)


def test_report_run_custom_dirs() -> None:
    mock_artifacts = {"chart": Path("my_reports/chart.png")}
    with patch("api.routers.report.run_report", return_value=mock_artifacts) as mock_fn:
        client.post(
            _URL,
            json={
                "scenario_dir": "my_configs",
                "reports_dir": "my_reports",
                "experiments_root": "my_experiments",
            },
        )
    mock_fn.assert_called_once_with(
        scenario_dir="my_configs",
        reports_dir="my_reports",
        experiments_root="my_experiments",
    )


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_report_run_file_not_found_returns_404() -> None:
    with patch("api.routers.report.run_report", side_effect=FileNotFoundError("no such dir")):
        resp = client.post(_URL, json={})
    assert resp.status_code == 404


def test_report_run_value_error_returns_400() -> None:
    with patch("api.routers.report.run_report", side_effect=ValueError("bad config")):
        resp = client.post(_URL, json={})
    assert resp.status_code == 400
