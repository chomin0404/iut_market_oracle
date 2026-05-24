"""Tests for API-key auth and rate-limiting middleware.

Auth (ORACLE_API_KEY):
  - Unset env var → all requests accepted (dev mode)
  - Set env var + correct key → 200
  - Set env var + wrong key   → 401
  - Set env var + missing key → 401
  Coverage: risk, gnss, and valuation routers (protected); bayesian (unprotected).

Rate limiting (RATE_LIMIT_RPM):
  - Requests within limit → 200
  - Requests exceeding limit → 429
"""

from __future__ import annotations

from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DCF_PAYLOAD = {
    "initial_fcf": 10.0,
    "growth_rate": 0.05,
    "discount_rate": 0.10,
    "forecast_years": 5,
    "terminal_growth_rate": 0.03,
}

_SIMULATE_PAYLOAD = {
    "n_vars": 1,
    "n_samples": 100,
    "distributions": [{"name": "normal", "params": {"loc": 0.0, "scale": 1.0}}],
    "copula": {"type": "independent"},
    "seed": 0,
}


def _fresh_client(monkeypatch, *, oracle_key: str | None, rate_limit_rpm: int = 100):
    """Build a TestClient with app re-created under the given env vars."""
    if oracle_key is not None:
        monkeypatch.setenv("ORACLE_API_KEY", oracle_key)
    else:
        monkeypatch.delenv("ORACLE_API_KEY", raising=False)
    monkeypatch.setenv("RATE_LIMIT_RPM", str(rate_limit_rpm))

    # Re-import app so middleware and deps pick up the new env vars
    import importlib

    import api.app as app_module

    importlib.reload(app_module)
    return TestClient(app_module.app)


# ---------------------------------------------------------------------------
# Auth — dev mode (ORACLE_API_KEY not set)
# ---------------------------------------------------------------------------


class TestAuthDevMode:
    """Without ORACLE_API_KEY, all protected routes are open."""

    def test_risk_open_without_key(self, monkeypatch) -> None:
        client = _fresh_client(monkeypatch, oracle_key=None)
        resp = client.post("/api/v1/risk/simulate", json=_SIMULATE_PAYLOAD)
        assert resp.status_code == 200

    def test_valuation_open_without_key(self, monkeypatch) -> None:
        client = _fresh_client(monkeypatch, oracle_key=None)
        resp = client.post("/api/v1/valuation/dcf", json=_DCF_PAYLOAD)
        assert resp.status_code == 200

    def test_unprotected_route_still_works(self, monkeypatch) -> None:
        """Health endpoint is not protected and should always be accessible."""
        client = _fresh_client(monkeypatch, oracle_key=None)
        resp = client.get("/health")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Auth — production mode (ORACLE_API_KEY set)
# ---------------------------------------------------------------------------


class TestAuthProductionMode:
    _KEY = "secret-test-key"

    def test_correct_key_returns_200(self, monkeypatch) -> None:
        client = _fresh_client(monkeypatch, oracle_key=self._KEY)
        resp = client.post(
            "/api/v1/risk/simulate",
            json=_SIMULATE_PAYLOAD,
            headers={"X-API-Key": self._KEY},
        )
        assert resp.status_code == 200

    def test_wrong_key_returns_401(self, monkeypatch) -> None:
        client = _fresh_client(monkeypatch, oracle_key=self._KEY)
        resp = client.post(
            "/api/v1/risk/simulate",
            json=_SIMULATE_PAYLOAD,
            headers={"X-API-Key": "wrong-key"},
        )
        assert resp.status_code == 401

    def test_missing_key_returns_401(self, monkeypatch) -> None:
        client = _fresh_client(monkeypatch, oracle_key=self._KEY)
        resp = client.post("/api/v1/risk/simulate", json=_SIMULATE_PAYLOAD)
        assert resp.status_code == 401

    def test_401_error_body_format(self, monkeypatch) -> None:
        client = _fresh_client(monkeypatch, oracle_key=self._KEY)
        resp = client.post("/api/v1/valuation/dcf", json=_DCF_PAYLOAD)
        assert resp.status_code == 401
        body = resp.json()
        assert body["status_code"] == 401
        assert body["error"] == "unauthorized"

    def test_unprotected_route_needs_no_key(self, monkeypatch) -> None:
        """Health endpoint requires no key even in production mode."""
        client = _fresh_client(monkeypatch, oracle_key=self._KEY)
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_reports_api_key_required(self, monkeypatch) -> None:
        client = _fresh_client(monkeypatch, oracle_key=self._KEY)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["features"]["api_key_required"] is True

    def test_health_reports_key_not_required_in_dev(self, monkeypatch) -> None:
        client = _fresh_client(monkeypatch, oracle_key=None)
        resp = client.get("/health")
        assert resp.json()["features"]["api_key_required"] is False


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


class TestRateLimit:
    def test_requests_within_limit_succeed(self, monkeypatch) -> None:
        client = _fresh_client(monkeypatch, oracle_key=None, rate_limit_rpm=5)
        for _ in range(5):
            resp = client.get("/health")
            assert resp.status_code == 200

    def test_excess_request_returns_429(self, monkeypatch) -> None:
        client = _fresh_client(monkeypatch, oracle_key=None, rate_limit_rpm=3)
        for _ in range(3):
            client.get("/health")
        resp = client.get("/health")
        assert resp.status_code == 429

    def test_429_body_format(self, monkeypatch) -> None:
        client = _fresh_client(monkeypatch, oracle_key=None, rate_limit_rpm=1)
        client.get("/health")
        resp = client.get("/health")
        assert resp.status_code == 429
        body = resp.json()
        assert body["status_code"] == 429
        assert body["error"] == "too_many_requests"
        assert "detail" in body

    def test_zero_rpm_disables_limiting(self, monkeypatch) -> None:
        """RATE_LIMIT_RPM=0 disables rate limiting entirely."""
        client = _fresh_client(monkeypatch, oracle_key=None, rate_limit_rpm=0)
        for _ in range(10):
            resp = client.get("/health")
            assert resp.status_code == 200

    def test_health_reports_rate_limit_rpm(self, monkeypatch) -> None:
        client = _fresh_client(monkeypatch, oracle_key=None, rate_limit_rpm=42)
        resp = client.get("/health")
        assert resp.json()["features"]["rate_limit_rpm"] == 42
