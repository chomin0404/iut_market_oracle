"""Idea formalisation endpoints (Formalize-Idea skill)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Body, Depends, HTTPException

from api.dependencies import make_api_key_dep
from modeling_api.examples import EXAMPLE_IDEA_INPUT
from schemas import IdeaInput, ParsedIdeaResponse

router = APIRouter()

# Dependency: validates X-Ideas-API-Key when IDEAS_API_KEY env var is set.
# When IDEAS_API_KEY is unset the endpoint is open (local / dev mode).
_require_ideas_key = make_api_key_dep("X-Ideas-API-Key", "IDEAS_API_KEY")


@router.post("/parse", response_model=ParsedIdeaResponse)
async def parse_idea(
    idea: Annotated[
        IdeaInput,
        Body(
            openapi_examples={
                "gps_spoofing": {
                    "summary": "GPS spoofing detection and defense",
                    "value": EXAMPLE_IDEA_INPUT,
                }
            }
        ),
    ],
    _: None = Depends(_require_ideas_key),
) -> ParsedIdeaResponse:
    """Formalise a structured research idea into a ParsedIdeaResponse.

    Analyses the provided ``IdeaInput`` using the Anthropic API and returns:

    - **problem_structure**: boolean flags for sequential, latent-state,
      decision-variable, physical-constraint, and uncertainty properties.
    - **candidate_families**: ordered list of suitable mathematical model
      families (snake_case identifiers, most suitable first).
    - **missing_information**: data or domain knowledge required before
      modelling can proceed.

    Authentication: set ``IDEAS_API_KEY`` env var to require ``X-Ideas-API-Key``
    header. If the env var is unset the endpoint is open (local/dev mode).

    Returns HTTP **401** if authentication fails,
    HTTP **503** if ``ANTHROPIC_API_KEY`` is not set,
    HTTP **502** for any API or parsing error.
    """

    try:
        from models.formalizer import parse_idea as _parse

        return await _parse(idea)
    except KeyError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"ANTHROPIC_API_KEY environment variable is not set: {exc}",
        )
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    except (RuntimeError, OSError, AttributeError) as exc:
        raise HTTPException(status_code=502, detail=f"LLM API error: {exc}")
