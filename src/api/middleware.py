"""ASGI middleware collection for the IUT Market Oracle API.

Middlewares (applied outermost → innermost):
  1. RequestIdMiddleware  — assigns a short correlation ID to every request.
  2. BodySizeLimitMiddleware — rejects payloads declared larger than the limit.
  3. RateLimitMiddleware  — per-IP sliding-window rate limiter.

Note: RateLimitMiddleware uses an in-process dict keyed by client IP.
Not shared across Uvicorn workers — for multi-worker deployments replace
with a Redis-backed counter.

Implementation: pure ASGI middleware (not BaseHTTPMiddleware) to avoid the
unawaited-coroutine RuntimeWarning that BaseHTTPMiddleware can trigger during
TestClient teardown in some Starlette versions.
"""

from __future__ import annotations

import contextvars
import json
import threading
import time
import uuid
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

_WINDOW_SECONDS: float = 60.0

# ---------------------------------------------------------------------------
# Request correlation ID — accessible anywhere within the same async context
# ---------------------------------------------------------------------------

_request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


def get_request_id() -> str:
    """Return the correlation ID for the current request (8 hex chars or '-')."""
    return _request_id_var.get()


# ASGI type aliases
_Scope = MutableMapping[str, Any]
_Receive = Callable[[], Awaitable[MutableMapping[str, Any]]]
_Send = Callable[[MutableMapping[str, Any]], Awaitable[None]]


class RateLimitMiddleware:
    """Per-IP sliding-window rate limiter (pure ASGI).

    Parameters
    ----------
    requests_per_minute:
        Maximum requests per IP within a 60-second window.
        Set to 0 to disable rate limiting entirely.
    """

    def __init__(self, app: Any, requests_per_minute: int = 100) -> None:
        self._app = app
        self._rpm = requests_per_minute
        self._buckets: defaultdict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    async def __call__(self, scope: _Scope, receive: _Receive, send: _Send) -> None:
        if scope["type"] != "http" or self._rpm <= 0:
            await self._app(scope, receive, send)
            return

        client = scope.get("client")
        ip: str = client[0] if client else "unknown"
        now = time.monotonic()

        with self._lock:
            dq = self._buckets[ip]
            while dq and now - dq[0] > _WINDOW_SECONDS:
                dq.popleft()
            if len(dq) >= self._rpm:
                await self._send_429(send)
                return
            dq.append(now)

        await self._app(scope, receive, send)

    @staticmethod
    async def _send_429(send: _Send) -> None:
        body = json.dumps(
            {
                "status_code": 429,
                "error": "too_many_requests",
                "detail": "Rate limit exceeded. Retry after 60 seconds.",
            }
        ).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 429,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                    # RFC 6585 §4 — clients SHOULD honour Retry-After
                    (b"retry-after", str(int(_WINDOW_SECONDS)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


# ---------------------------------------------------------------------------
# Request ID middleware
# ---------------------------------------------------------------------------


class RequestIdMiddleware:
    """Assigns a short correlation ID to every HTTP request (pure ASGI).

    The ID is stored in ``_request_id_var`` so that any code within the same
    async context can call ``get_request_id()`` to retrieve it (e.g. for
    structured logging).  The ID is also echoed back to the client as the
    ``X-Request-Id`` response header.
    """

    def __init__(self, app: Any) -> None:
        self._app = app

    async def __call__(self, scope: _Scope, receive: _Receive, send: _Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        request_id = uuid.uuid4().hex[:8]
        token = _request_id_var.set(request_id)

        async def _send_with_id(message: MutableMapping[str, Any]) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((b"x-request-id", request_id.encode()))
                message = {**message, "headers": headers}
            await send(message)

        try:
            await self._app(scope, receive, _send_with_id)
        finally:
            _request_id_var.reset(token)


# ---------------------------------------------------------------------------
# Body size limit middleware
# ---------------------------------------------------------------------------

_BODY_SIZE_LIMIT: int = 1_048_576  # 1 MiB


class BodySizeLimitMiddleware:
    """Rejects requests whose declared Content-Length exceeds *max_bytes* (pure ASGI).

    Inspection is header-only; the request body is never buffered.
    Requests without a Content-Length header pass through unchanged
    (chunked transfer encoding is handled downstream).

    Parameters
    ----------
    max_bytes:
        Maximum allowed body size in bytes.  Default: 1 MiB (1 048 576).
    """

    def __init__(self, app: Any, max_bytes: int = _BODY_SIZE_LIMIT) -> None:
        self._app = app
        self._max_bytes = max_bytes

    async def __call__(self, scope: _Scope, receive: _Receive, send: _Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        headers: dict[bytes, bytes] = dict(scope.get("headers", []))
        raw_cl = headers.get(b"content-length")
        if raw_cl is not None:
            try:
                content_length = int(raw_cl)
            except ValueError:
                content_length = 0
            if content_length > self._max_bytes:
                await self._send_413(send, self._max_bytes)
                return

        await self._app(scope, receive, send)

    @staticmethod
    async def _send_413(send: _Send, max_bytes: int) -> None:
        limit_mib = max_bytes // _BODY_SIZE_LIMIT
        body = json.dumps(
            {
                "status_code": 413,
                "error": "request_entity_too_large",
                "detail": f"Request body exceeds the {limit_mib} MiB limit.",
            }
        ).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
