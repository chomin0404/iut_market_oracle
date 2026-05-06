"""FastAPI application for the IUT Market Oracle research platform.

Run:
    uv run uvicorn src.api.app:app --reload

Interactive docs:
    http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path as _Path

# Add src/ to sys.path before any intra-project imports so that
# "from schemas import ..." resolves to src/schemas.py, not the root-level one.
_src_dir = str(_Path(__file__).parent.parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

from api.routers import (
    bayesian,
    entropy,
    exit_,
    experiments,
    forge,
    gnss,
    graph,
    ideas,
    matroid,
    model,
    report,
    strategy,
    twin,
    valuation,
)
from modeling_api.examples import (
    EXAMPLE_MODEL_RECOMMENDATION,
    EXAMPLE_MODEL_SPEC,
    EXAMPLE_PARSED_IDEA_RESPONSE,
)

# ---------------------------------------------------------------------------
# OpenAPI tag metadata
# ---------------------------------------------------------------------------

_TAGS_METADATA = [
    {
        "name": "system",
        "description": "Health check and API metadata.",
    },
    {
        "name": "valuation",
        "description": "DCF valuation, reverse DCF (implied growth rate), and scenario runner.",
    },
    {
        "name": "bayesian",
        "description": "Bayesian conjugate update (Beta / Normal prior) from evidence lists.",
    },
    {
        "name": "graph",
        "description": "Portfolio skill/dependency graph metrics (Fiedler value, portfolio score).",
    },
    {
        "name": "experiments",
        "description": "Experiment registry: create, list, load, and update experiment metadata.",
    },
    {
        "name": "twin",
        "description": (
            "Digital twin engine: forward simulation, Bayesian calibration, "
            "Markov regime switching, and Gamma-Poisson market evolution."
        ),
    },
    {
        "name": "exit",
        "description": "Exit strategy option pricing, timing maps, and scenario comparison.",
    },
    {
        "name": "entropy",
        "description": "Shannon entropy, KL divergence, and distribution-shift detection.",
    },
    {
        "name": "report",
        "description": "Report pipeline: generate charts, tables, and summary markdown.",
    },
    {
        "name": "gnss",
        "description": (
            "GNSS Resilience Twin (T1300–T1500): OSNMA/TESLA authentication, "
            "signal-level Monte Carlo spoofing detection, multi-sensor fusion, "
            "and 10-layer probabilistic fault discrimination."
        ),
    },
    {
        "name": "matroid",
        "description": "Matroid log-concavity checks and basis distribution analysis.",
    },
    {
        "name": "model",
        "description": "Model registry: list, retrieve, recommend, and generate model specs.",
    },
    {
        "name": "ideas",
        "description": "Natural-language idea parsing and formalization.",
    },
    {
        "name": "strategy",
        "description": "Strategy Twin: causal effect estimation and macro environment analysis.",
    },
    {
        "name": "forge",
        "description": (
            "ModelForge governance: YAML spec → verify → skeleton → trace → audit. "
            "Every model must pass verification before code is written."
        ),
    },
]

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

_START_TIME = time.time()

app = FastAPI(
    title="IUT Market Oracle API",
    version="0.1.0",
    description=(
        "Quantitative research platform — math spec to decision.\n\n"
        "**Modules**\n"
        "- DCF valuation & reverse DCF\n"
        "- Bayesian conjugate inference\n"
        "- Graph portfolio metrics (Fiedler, RMT, matroid)\n"
        "- Digital twin simulation (Markov regime, Gamma-Poisson)\n"
        "- Exit strategy option pricing\n"
        "- Entropy monitoring & distribution-shift detection\n"
        "- GNSS Resilience Twin — 10-layer spoofing/fault discrimination (T1300–T1500)\n"
        "- Yield twin GP surrogate + D-optimal design (T1600)\n"
        "- Strategy twin with causal inference (T1700)\n"
        "- ModelForge governance (YAML spec → verify → skeleton → trace → audit)\n\n"
        "**Usage**\n"
        "```\n"
        "uv run uvicorn src.api.app:app --reload\n"
        "```\n"
        "Open `/docs` for interactive Swagger UI or `/redoc` for ReDoc."
    ),
    openapi_tags=_TAGS_METADATA,
)

# ---------------------------------------------------------------------------
# CORS — allow all origins by default; restrict via CORS_ORIGINS env var.
# Example: CORS_ORIGINS="https://myapp.example.com,https://staging.example.com"
# ---------------------------------------------------------------------------

_raw_origins = os.environ.get("CORS_ORIGINS", "*")
_allow_origins: list[str] = (
    ["*"] if _raw_origins == "*" else [o.strip() for o in _raw_origins.split(",") if o.strip()]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allow_origins,
    allow_credentials=_raw_origins != "*",  # credentials require explicit origins
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(valuation.router, prefix="/valuation", tags=["valuation"])
app.include_router(bayesian.router, prefix="/bayesian", tags=["bayesian"])
app.include_router(graph.router, prefix="/graph", tags=["graph"])
app.include_router(experiments.router, prefix="/experiments", tags=["experiments"])
app.include_router(twin.router, prefix="/twin", tags=["twin"])
app.include_router(exit_.router, prefix="/exit", tags=["exit"])
app.include_router(entropy.router, prefix="/entropy", tags=["entropy"])
app.include_router(report.router, prefix="/report", tags=["report"])
app.include_router(gnss.router, prefix="/gnss", tags=["gnss"])
app.include_router(matroid.router, prefix="/matroid", tags=["matroid"])
app.include_router(model.router, prefix="/model", tags=["model"])
app.include_router(ideas.router, prefix="/ideas", tags=["ideas"])
app.include_router(strategy.router, prefix="/strategy", tags=["strategy"])
app.include_router(forge.router, prefix="/forge", tags=["forge"])

# ---------------------------------------------------------------------------
# System endpoints
# ---------------------------------------------------------------------------


class _APIInfo(BaseModel):
    title: str
    version: str
    docs: str
    redoc: str
    health: str
    openapi: str


class _HealthResponse(BaseModel):
    status: str
    version: str
    uptime_seconds: float


@app.get("/", response_model=_APIInfo, tags=["system"], include_in_schema=True)
def root() -> _APIInfo:
    """Return basic API metadata and links to documentation."""
    return _APIInfo(
        title=app.title,
        version=app.version,
        docs="/docs",
        redoc="/redoc",
        health="/health",
        openapi="/openapi.json",
    )


@app.get("/health", response_model=_HealthResponse, tags=["system"])
def health() -> _HealthResponse:
    """Health check — returns status, version, and uptime.

    Returns HTTP 200 when the service is ready to accept requests.
    Use this endpoint for liveness/readiness probes in container orchestration.
    """
    return _HealthResponse(
        status="ok",
        version=app.version,
        uptime_seconds=round(time.time() - _START_TIME, 2),
    )


# ---------------------------------------------------------------------------
# Unified error handlers
# ---------------------------------------------------------------------------


class _ErrorResponse(BaseModel):
    status_code: int
    error: str
    detail: object


@app.exception_handler(StarletteHTTPException)
async def _http_exception_handler(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content=_ErrorResponse(
            status_code=exc.status_code,
            error=_HTTP_STATUS_LABELS.get(exc.status_code, "error"),
            detail=exc.detail,
        ).model_dump(),
        headers=getattr(exc, "headers", None),
    )


@app.exception_handler(RequestValidationError)
async def _validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content=_ErrorResponse(
            status_code=422,
            error="unprocessable_entity",
            detail=jsonable_encoder(exc.errors()),
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content=_ErrorResponse(
            status_code=500,
            error="internal_server_error",
            detail=str(exc),
        ).model_dump(),
    )


_HTTP_STATUS_LABELS: dict[int, str] = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
    422: "unprocessable_entity",
    429: "too_many_requests",
    500: "internal_server_error",
    503: "service_unavailable",
}

# ---------------------------------------------------------------------------
# Custom OpenAPI: inject response-schema examples for schemas that cannot
# use Body(openapi_examples=...) because they appear only in responses.
# ---------------------------------------------------------------------------

_RESPONSE_EXAMPLES: dict[str, dict] = {
    "ModelRecommendation": EXAMPLE_MODEL_RECOMMENDATION,
    "ModelSpec": EXAMPLE_MODEL_SPEC,
    "ParsedIdeaResponse": EXAMPLE_PARSED_IDEA_RESPONSE,
}


def _custom_openapi() -> dict:
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
        tags=_TAGS_METADATA,
    )
    component_schemas: dict = schema.get("components", {}).get("schemas", {})
    for name, example in _RESPONSE_EXAMPLES.items():
        if name in component_schemas:
            component_schemas[name]["examples"] = [example]
    app.openapi_schema = schema
    return app.openapi_schema


app.openapi = _custom_openapi
