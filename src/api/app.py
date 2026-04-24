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

from api.routers import (
    bayesian,
    entropy,
    exit_,
    experiments,
    gnss,
    graph,
    matroid,
    report,
    twin,
    valuation,
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
