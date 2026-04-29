"""FastAPI application for the IUT Market Oracle research platform.

Run:
    uv run uvicorn src.api.app:app --reload

Interactive docs:
    http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

# Add src/ to sys.path before any intra-project imports so that
# "from schemas import ..." resolves to src/schemas.py, not the root-level one.
_src_dir = str(_Path(__file__).parent.parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

from api.routers import (
    bayesian,
    entropy,
    exit_,
    experiments,
    gnss,
    graph,
    ideas,
    matroid,
    model,
    report,
    twin,
    valuation,
)
from modeling_api.examples import (
    EXAMPLE_MODEL_RECOMMENDATION,
    EXAMPLE_MODEL_SPEC,
    EXAMPLE_PARSED_IDEA_RESPONSE,
)

app = FastAPI(
    title="IUT Market Oracle API",
    version="0.1.0",
    description=(
        "Quantitative research platform: DCF valuation, Bayesian inference, "
        "graph portfolio metrics, digital twin simulation, exit strategy pricing, "
        "entropy monitoring, and report generation."
    ),
)

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
    )
    component_schemas: dict = schema.get("components", {}).get("schemas", {})
    for name, example in _RESPONSE_EXAMPLES.items():
        if name in component_schemas:
            component_schemas[name]["examples"] = [example]
    app.openapi_schema = schema
    return app.openapi_schema


app.openapi = _custom_openapi
