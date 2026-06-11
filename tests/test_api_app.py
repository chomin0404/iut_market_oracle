"""Tests for system-level API endpoints and middleware behaviour."""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.app import app

client = TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------


def test_root_status() -> None:
    resp = client.get("/")
    assert resp.status_code == 200


def test_root_fields() -> None:
    body = client.get("/").json()
    assert body["title"] == "IUT Market Oracle API"
    assert body["version"] == "0.1.0"
    assert body["api_prefix"] == "/api/v1"
    assert body["docs"] == "/docs"
    assert body["redoc"] == "/redoc"
    assert body["health"] == "/health"
    assert body["openapi"] == "/openapi.json"


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


def test_health_status() -> None:
    resp = client.get("/health")
    assert resp.status_code == 200


def test_health_body() -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["version"] == "0.1.0"
    assert isinstance(body["uptime_seconds"], float)
    assert body["uptime_seconds"] >= 0.0


def test_health_features_field() -> None:
    body = client.get("/health").json()
    assert "features" in body
    assert isinstance(body["features"], dict)


def test_health_features_ideas_available_is_bool() -> None:
    body = client.get("/health").json()
    assert isinstance(body["features"]["ideas_available"], bool)


def test_health_features_cors_restricted_is_bool() -> None:
    body = client.get("/health").json()
    assert isinstance(body["features"]["cors_restricted"], bool)


# ---------------------------------------------------------------------------
# CORS headers
# ---------------------------------------------------------------------------


def test_cors_preflight_allowed_origin() -> None:
    """OPTIONS preflight from an allowed origin should return ACAO header."""
    resp = client.options(
        "/health",
        headers={
            "Origin": "http://localhost:8000",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert resp.status_code in (200, 204)
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:8000"


def test_cors_preflight_unknown_origin_rejected() -> None:
    """OPTIONS preflight from an unknown origin must not return ACAO header."""
    resp = client.options(
        "/health",
        headers={
            "Origin": "https://attacker.example.com",
            "Access-Control-Request-Method": "GET",
        },
    )
    # Starlette returns 400 for disallowed origins on preflight
    assert resp.status_code in (400, 200, 204)
    assert resp.headers.get("access-control-allow-origin") != "https://attacker.example.com"


def test_cors_header_on_get() -> None:
    resp = client.get("/health", headers={"Origin": "http://localhost:8000"})
    assert resp.status_code == 200
    assert "access-control-allow-origin" in resp.headers


# ---------------------------------------------------------------------------
# Unified error response shape
# ---------------------------------------------------------------------------


def test_404_error_shape() -> None:
    resp = client.get("/does-not-exist")
    assert resp.status_code == 404
    body = resp.json()
    assert body["status_code"] == 404
    assert body["error"] == "not_found"
    assert "detail" in body


def test_405_error_shape() -> None:
    # POST to a GET-only endpoint
    resp = client.post("/health")
    assert resp.status_code == 405
    body = resp.json()
    assert body["status_code"] == 405
    assert body["error"] == "method_not_allowed"


def test_400_error_shape() -> None:
    # /valuation/dcf with invalid payload triggers 422 (validation)
    resp = client.post("/api/v1/valuation/dcf", json={"initial_fcf": -1.0})
    assert resp.status_code == 422
    body = resp.json()
    assert body["status_code"] == 422
    assert body["error"] == "unprocessable_entity"


# ---------------------------------------------------------------------------
# OpenAPI schema sanity
# ---------------------------------------------------------------------------


def test_openapi_schema_reachable() -> None:
    resp = client.get("/openapi.json")
    assert resp.status_code == 200


def test_openapi_has_tags() -> None:
    schema = client.get("/openapi.json").json()
    tag_names = {t["name"] for t in schema.get("tags", [])}
    for expected in ("system", "valuation", "gnss", "forge", "bayesian"):
        assert expected in tag_names, f"tag '{expected}' missing from OpenAPI schema"


def test_openapi_tag_descriptions() -> None:
    schema = client.get("/openapi.json").json()
    for tag in schema.get("tags", []):
        assert tag.get("description"), f"tag '{tag['name']}' has no description"
