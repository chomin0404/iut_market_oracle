"""LLM-based model recommendation engine (T1400).

Given a free-text problem description and optional explicit signals,
calls Claude via tool_use to produce a validated ModelRecommendation.

Environment:
    ANTHROPIC_API_KEY: required; raises KeyError if absent.
"""

from __future__ import annotations

import json
import os

import anthropic

from schemas import ModelRecommendation

DEFAULT_MODEL = "claude-opus-4-6"

_SYSTEM_PROMPT = """\
You are an expert in applied mathematics, statistics, and machine learning.
Given a problem description and optional signals, identify the problem class
and recommend the most appropriate mathematical model families.

Guidelines:
- Infer a concise, precise problem_type label (snake_case preferred).
- List recommended_models in descending order of suitability.
  Use canonical snake_case identifiers (e.g. kalman_filter, gaussian_process).
  Include both well-known classes and specific variants when relevant.
- Each rationale item must correspond 1-to-1 with a recommended model and
  explain *why* that model fits the problem (one sentence per item).
- Prefer models that match the provided registry IDs when applicable,
  but do not restrict recommendations to the registry alone.

Always call the `recommend_models` tool with your complete recommendation.
"""

_TOOL_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "problem_type": {
            "type": "string",
            "description": (
                "Concise label for the inferred problem class, e.g. "
                "'sequential_decision_under_uncertainty'"
            ),
        },
        "recommended_models": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "description": (
                "Ordered list of recommended model identifiers (snake_case). "
                "First entry is the primary recommendation."
            ),
        },
        "rationale": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "description": (
                "One rationale sentence per recommended model, "
                "in the same order as recommended_models."
            ),
        },
    },
    "required": ["problem_type", "recommended_models", "rationale"],
}

_RECOMMEND_TOOL: anthropic.types.ToolParam = {
    "name": "recommend_models",
    "description": (
        "Recommend mathematical models appropriate for the described problem. "
        "Return problem_type, an ordered list of model identifiers, "
        "and one rationale sentence per model."
    ),
    "input_schema": _TOOL_SCHEMA,
}


def recommend_models(
    description: str,
    signals: list[str] | None = None,
    registry_ids: list[str] | None = None,
    model: str = DEFAULT_MODEL,
) -> ModelRecommendation:
    """Recommend mathematical models for the given problem description.

    Args:
        description:  Free-text description of the problem or phenomenon.
        signals:      Optional explicit characteristics of the problem
                      (e.g. ['latent dynamics exist', 'online observations available']).
        registry_ids: Known model IDs from the registry passed as context hints.
                      If None, no registry context is injected.
        model:        Anthropic model ID to use.

    Returns:
        Validated ModelRecommendation with problem_type, recommended_models, rationale.

    Raises:
        KeyError:  ANTHROPIC_API_KEY is not set.
        ValueError: LLM did not call the tool or returned an unexpected structure.
        pydantic.ValidationError: Tool output fails ModelRecommendation validation.
    """
    api_key = os.environ["ANTHROPIC_API_KEY"]
    client = anthropic.Anthropic(api_key=api_key)

    parts: list[str] = [f"Problem description: {description}"]
    if signals:
        parts.append("Observed signals:\n" + "\n".join(f"- {s}" for s in signals))
    if registry_ids:
        parts.append(
            "Available registry model IDs (prefer these when applicable):\n"
            + ", ".join(registry_ids)
        )

    user_content = "\n\n".join(parts)

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=_SYSTEM_PROMPT,
        tools=[_RECOMMEND_TOOL],
        tool_choice={"type": "any"},
        messages=[{"role": "user", "content": user_content}],
    )

    tool_block: anthropic.types.ToolUseBlock | None = None
    for block in response.content:
        if block.type == "tool_use" and block.name == "recommend_models":
            tool_block = block
            break

    if tool_block is None:
        raise ValueError(
            f"LLM did not call 'recommend_models'. Response stop_reason={response.stop_reason!r}."
        )

    raw: dict = (
        tool_block.input if isinstance(tool_block.input, dict) else json.loads(tool_block.input)
    )

    rec = ModelRecommendation(**raw)

    if len(rec.rationale) != len(rec.recommended_models):
        raise ValueError(
            f"rationale length ({len(rec.rationale)}) must equal "
            f"recommended_models length ({len(rec.recommended_models)})."
        )

    return rec
