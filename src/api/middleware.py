"""Simple sliding-window rate-limiter middleware.

Configure with the ``RATE_LIMIT_RPM`` environment variable (default 100).
When the limit is exceeded the middleware returns HTTP 429 with a JSON body
that matches the unified error format used by the rest of the application.

Note: Uses an in-process dict keyed by client IP.  Not shared across Uvicorn
workers — for multi-worker deployments replace with a Redis-backed counter.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

_WINDOW_SECONDS: float = 60.0


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Per-IP sliding-window rate limiter.

    Parameters
    ----------
    requests_per_minute:
        Maximum number of requests allowed per IP within a 60-second window.
        Set to 0 to disable rate limiting entirely.
    """

    def __init__(self, app, requests_per_minute: int = 100) -> None:
        super().__init__(app)
        self._rpm = requests_per_minute
        self._buckets: defaultdict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        if self._rpm <= 0:
            return await call_next(request)

        ip = request.client.host if request.client else "unknown"
        now = time.monotonic()

        with self._lock:
            dq = self._buckets[ip]
            # Evict timestamps older than the window
            while dq and now - dq[0] > _WINDOW_SECONDS:
                dq.popleft()
            if len(dq) >= self._rpm:
                return JSONResponse(
                    status_code=429,
                    content={
                        "status_code": 429,
                        "error": "too_many_requests",
                        "detail": "Rate limit exceeded. Retry after 60 seconds.",
                    },
                )
            dq.append(now)

        return await call_next(request)
