"""Simple sliding-window rate-limiter middleware.

Configure with the ``RATE_LIMIT_RPM`` environment variable (default 0 = disabled).
When the limit is exceeded the middleware returns HTTP 429 with a JSON body
that matches the unified error format used by the rest of the application.

Note: Uses an in-process dict keyed by client IP.  Not shared across Uvicorn
workers — for multi-worker deployments replace with a Redis-backed counter.

Implementation: pure ASGI middleware (not BaseHTTPMiddleware) to avoid the
unawaited-coroutine RuntimeWarning that BaseHTTPMiddleware can trigger during
TestClient teardown in some Starlette versions.
"""

from __future__ import annotations

import json
import threading
import time
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

_WINDOW_SECONDS: float = 60.0

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
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
