"""Tests for POST /ideas/parse (LLM calls mocked)."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from api.app import app
from schemas import ParsedIdeaResponse, ProblemStructure

client = TestClient(app, raise_server_exceptions=False)

_URL = "/api/v1/ideas/parse"

_VALID_IDEA = {
    "title": "GPS spoofing detection",
    "description": "Detect GNSS spoofing attacks using pseudorange and Doppler anomalies.",
    "goal_type": "anomaly_detection",
    "time_horizon": "sequential",
    "data_regime": "medium",
    "uncertainty_level": "high",
}

_MOCK_RESPONSE = ParsedIdeaResponse(
    problem_structure=ProblemStructure(
        is_sequential=True,
        has_latent_state=True,
        has_decision_variables=False,
        has_physical_constraints=True,
        is_high_uncertainty=True,
        is_data_scarce=False,
    ),
    candidate_families=["kalman_filter", "bayesian_hypothesis_test"],
    missing_information=["satellite geometry", "receiver clock model"],
)


# ---------------------------------------------------------------------------
# Success (no API key required)
# ---------------------------------------------------------------------------


def test_parse_idea_returns_200() -> None:
    with patch("models.formalizer.parse_idea", new=AsyncMock(return_value=_MOCK_RESPONSE)):
        resp = client.post(_URL, json=_VALID_IDEA)
    assert resp.status_code == 200


def test_parse_idea_response_fields() -> None:
    with patch("models.formalizer.parse_idea", new=AsyncMock(return_value=_MOCK_RESPONSE)):
        body = client.post(_URL, json=_VALID_IDEA).json()
    assert "problem_structure" in body
    assert "candidate_families" in body
    assert "missing_information" in body


def test_parse_idea_problem_structure_fields() -> None:
    with patch("models.formalizer.parse_idea", new=AsyncMock(return_value=_MOCK_RESPONSE)):
        body = client.post(_URL, json=_VALID_IDEA).json()
    ps = body["problem_structure"]
    for field in (
        "is_sequential",
        "has_latent_state",
        "has_decision_variables",
        "has_physical_constraints",
        "is_high_uncertainty",
        "is_data_scarce",
    ):
        assert field in ps


def test_parse_idea_candidate_families_is_list() -> None:
    with patch("models.formalizer.parse_idea", new=AsyncMock(return_value=_MOCK_RESPONSE)):
        body = client.post(_URL, json=_VALID_IDEA).json()
    assert isinstance(body["candidate_families"], list)


# ---------------------------------------------------------------------------
# Validation errors (no mock needed — Pydantic rejects before reaching handler)
# ---------------------------------------------------------------------------


def test_missing_title_returns_422() -> None:
    idea = {**_VALID_IDEA}
    del idea["title"]
    resp = client.post(_URL, json=idea)
    assert resp.status_code == 422


def test_missing_description_returns_422() -> None:
    idea = {**_VALID_IDEA}
    del idea["description"]
    resp = client.post(_URL, json=idea)
    assert resp.status_code == 422


def test_title_too_short_returns_422() -> None:
    idea = {**_VALID_IDEA, "title": "ab"}
    resp = client.post(_URL, json=idea)
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


def test_no_auth_required_when_env_unset() -> None:
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("IDEAS_API_KEY", None)
        with patch("models.formalizer.parse_idea", new=AsyncMock(return_value=_MOCK_RESPONSE)):
            resp = client.post(_URL, json=_VALID_IDEA)
    assert resp.status_code == 200


def test_wrong_api_key_returns_403() -> None:
    with patch.dict(os.environ, {"IDEAS_API_KEY": "secret"}):
        resp = client.post(_URL, json=_VALID_IDEA, headers={"X-Ideas-API-Key": "wrong"})
    assert resp.status_code == 403


def test_correct_api_key_returns_200() -> None:
    with patch.dict(os.environ, {"IDEAS_API_KEY": "secret"}):
        with patch("models.formalizer.parse_idea", new=AsyncMock(return_value=_MOCK_RESPONSE)):
            resp = client.post(_URL, json=_VALID_IDEA, headers={"X-Ideas-API-Key": "secret"})
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Backend errors
# ---------------------------------------------------------------------------


def test_missing_anthropic_key_returns_503() -> None:
    with patch(
        "models.formalizer.parse_idea",
        new=AsyncMock(side_effect=KeyError("ANTHROPIC_API_KEY")),
    ):
        resp = client.post(_URL, json=_VALID_IDEA)
    assert resp.status_code == 503


def test_llm_value_error_returns_502() -> None:
    with patch(
        "models.formalizer.parse_idea",
        new=AsyncMock(side_effect=ValueError("parse failed")),
    ):
        resp = client.post(_URL, json=_VALID_IDEA)
    assert resp.status_code == 502
