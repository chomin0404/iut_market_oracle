"""Shared FastAPI dependencies for authentication and common checks.

Usage
-----
Create a reusable API-key dependency with :func:`make_api_key_dep` and inject
it via ``Depends()`` in any route that needs protection::

    from api.dependencies import make_api_key_dep

    _require_key = make_api_key_dep("X-My-API-Key", "MY_API_KEY")

    @router.post("/protected")
    async def endpoint(_: None = Depends(_require_key)) -> ...:
        ...

When the environment variable is **not** set the endpoint is open
(local / dev mode).  Set the variable in production to enable enforcement.
"""

from __future__ import annotations

import os

from fastapi import HTTPException, Security
from fastapi.security.api_key import APIKeyHeader


def make_api_key_dep(header_name: str, env_var: str):
    """Return a FastAPI dependency that validates an API-key header.

    Parameters
    ----------
    header_name:
        HTTP header name clients must send (e.g. ``"X-Ideas-API-Key"``).
    env_var:
        Environment variable that holds the expected key value.
        If the variable is empty / unset, all requests are accepted.

    Returns
    -------
    An ``async`` callable suitable for use with ``Depends()``.
    Raises HTTP **401** when a key is required but absent or wrong.
    """
    _scheme = APIKeyHeader(name=header_name, auto_error=False)

    async def _check(key: str | None = Security(_scheme)) -> None:
        required = os.getenv(env_var)
        if required and key != required:
            raise HTTPException(
                status_code=401,
                detail=f"Invalid or missing {header_name} header.",
            )

    return _check
