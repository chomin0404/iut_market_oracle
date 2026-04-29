"""Idea formalisation endpoints (Formalize-Idea skill)."""

from __future__ import annotations

import os
from typing import Annotated

from fastapi import APIRouter, Body, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader

from modeling_api.examples import EXAMPLE_IDEA_INPUT
from schemas import IdeaInput, ParsedIdeaResponse

router = APIRouter()

_API_KEY_HEADER = APIKeyHeader(name="X-Ideas-API-Key", auto_error=False)


def _check_api_key(key: str | None) -> None:
    """Validate the X-Ideas-API-Key header if IDEAS_API_KEY env var is set.

    If IDEAS_API_KEY is not set the endpoint is open (local/dev mode).
    Raises HTTP 401 when the key is required but missing or wrong.
    """
    required = os.getenv("IDEAS_API_KEY")
    if required and key != required:
        raise HTTPException(status_code=401, detail="Invalid or missing X-Ideas-API-Key")


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
    api_key: str | None = Security(_API_KEY_HEADER),
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
    _check_api_key(api_key)

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
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"LLM API error: {exc}")
