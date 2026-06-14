"""API-layer request/response schemas (one sub-module per router domain).

Schema layering convention
--------------------------
This package (``api.schemas``) is the **API DTO layer**.
It contains FastAPI request/response models (Pydantic ``BaseModel``) that are
consumed exclusively by ``api.routers.*``.

``src/schemas/`` is the **domain layer**.  It holds canonical business types
(MCSimReport, ObservationEpoch, etc.) shared across routers, tests, and
non-API modules.  API schemas may import from ``schemas.*`` but the reverse
dependency is forbidden: ``schemas.*`` must never import from ``api.schemas.*``.

Naming note: sub-module filenames mirror the router domain (e.g. ``gnss.py``,
``bayesian.py``) — not the domain-layer schema file of the same name.
The two packages contain different types serving different abstraction levels.
"""
