"""Mathematical model registry and LLM generation endpoints (T1400)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel, Field

from models.registry import load_registry, search_registry
from schemas import ModelRecommendation, ModelRegistryEntry, ModelSpec

router = APIRouter()

# Load registry once at module import time.
_registry: list[ModelRegistryEntry] = load_registry()


class GenerateRequest(BaseModel):
    idea: str = Field(..., min_length=1, description="Natural-language idea to formalise")
    domain: str | None = Field(None, description="Optional domain hint, e.g. 'finance'")


class RecommendRequest(BaseModel):
    description: str = Field(..., min_length=1, description="Problem or phenomenon to model")
    signals: list[str] | None = Field(
        None,
        description="Explicit problem characteristics, e.g. ['latent dynamics exist']",
    )


@router.get("/registry", response_model=list[ModelRegistryEntry])
def list_registry(
    query: str | None = None,
    category: str | None = None,
    tags: str | None = None,
) -> list[ModelRegistryEntry]:
    """Return registry entries, optionally filtered by query, category, and/or tags.

    - **query**: case-insensitive substring matched against name, problem_type, tags
    - **category**: exact (case-insensitive) category match
    - **tags**: comma-separated list of required tags (e.g. `bayesian,finance`)
    """
    tag_list: list[str] | None = [t.strip() for t in tags.split(",") if t.strip()] if tags else None
    return search_registry(query=query, category=category, tags=tag_list, registry=_registry)


@router.get("/registry/{model_id}", response_model=ModelRegistryEntry)
def get_registry_entry(model_id: str) -> ModelRegistryEntry:
    """Return a single registry entry by its snake_case *model_id*.

    Raises **404** if the id is not found.
    """
    for entry in _registry:
        if entry.id == model_id:
            return entry
    raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found in registry.")


_RECOMMEND_EXAMPLES = {
    "gps_spoofing": {
        "summary": "Sequential anomaly detection — GPS spoofing",
        "value": {
            "description": (
                "Sequential anomaly detection with latent state, physical constraints,"
                " and high uncertainty."
            ),
            "signals": [
                "latent dynamics exist",
                "physical constraints apply",
                "high uncertainty",
            ],
        },
    }
}

_GENERATE_EXAMPLES = {
    "gps_spoofing": {
        "summary": "EKF-based GPS spoofing detector",
        "value": {
            "idea": (
                "GPS spoofing detection using EKF on pseudorange residuals"
                " with CUSUM alarm threshold."
            ),
            "domain": "navigation_security",
        },
    }
}


@router.post("/recommend", response_model=ModelRecommendation)
def recommend_model(
    req: Annotated[RecommendRequest, Body(openapi_examples=_RECOMMEND_EXAMPLES)],
) -> ModelRecommendation:
    """Recommend mathematical models for a problem description using the Anthropic API.

    Registry model IDs are passed as context so the LLM can prefer known entries.
    Returns HTTP 503 if the API key is missing and HTTP 502 for API errors.
    """
    try:
        from models.recommender import recommend_models

        registry_ids = [e.id for e in _registry]
        return recommend_models(
            description=req.description,
            signals=req.signals,
            registry_ids=registry_ids,
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"ANTHROPIC_API_KEY environment variable is not set: {exc}",
        )
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"LLM API error: {exc}")


@router.post("/generate", response_model=ModelSpec)
def generate_model(
    req: Annotated[GenerateRequest, Body(openapi_examples=_GENERATE_EXAMPLES)],
) -> ModelSpec:
    """Generate a ModelSpec from a natural-language idea using the Anthropic API.

    Requires the `ANTHROPIC_API_KEY` environment variable to be set.
    Returns HTTP 503 if the API key is missing and HTTP 502 for API errors.
    """
    try:
        from models.generator import generate_model_spec

        return generate_model_spec(idea=req.idea, domain=req.domain)
    except KeyError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"ANTHROPIC_API_KEY environment variable is not set: {exc}",
        )
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"LLM API error: {exc}")
