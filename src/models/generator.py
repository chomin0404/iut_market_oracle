"""LLM-based mathematical model specification generator (T1400).

Uses the Anthropic SDK tool_use API to coerce Claude into producing a
validated ModelSpec JSON object from a free-text idea description.

Environment:
    ANTHROPIC_API_KEY: required; raises KeyError if absent.
"""

from __future__ import annotations

import json
import os

import anthropic

from schemas import ModelSpec

# Default model used for generation; override via the `model` argument.
DEFAULT_MODEL = "claude-opus-4-6"

_SYSTEM_PROMPT = """\
You are an expert mathematical modeller.
Given a research idea or phenomenon, produce a rigorous, self-consistent
mathematical model specification.
Be precise with notation: use standard mathematical symbols and clearly
define all variables, parameters, and constraints.
Always call the `create_model_spec` tool with the complete specification.
"""

# Build input_schema from ModelSpec JSON schema, dropping pydantic meta-fields.
_MODEL_SPEC_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "problem_type": {
            "type": "string",
            "description": "Class of mathematical problem (e.g. 'Bayesian state estimation')",
        },
        "objective": {
            "type": "string",
            "description": "The optimisation or inference objective in precise terms",
        },
        "state_variables": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Latent/hidden state variables with domains",
        },
        "observables": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Observed or measured quantities",
        },
        "parameters": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Model parameters to be calibrated or inferred",
        },
        "constraints": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Mathematical constraints (inequalities, conservation laws, etc.)",
        },
        "uncertainty": {
            "type": "object",
            "additionalProperties": {"type": "string"},
            "description": "Noise/uncertainty specifications keyed by source name",
        },
        "equations": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Key equations in mathematical notation",
        },
        "priors": {
            "type": "object",
            "additionalProperties": {"type": "string"},
            "description": "Prior distributions over parameters",
        },
        "loss_function": {
            "type": ["string", "null"],
            "description": "Loss, energy, or objective function; null if not applicable",
        },
        "solver": {
            "type": "string",
            "description": "Recommended solver or inference algorithm",
        },
        "outputs": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Model outputs and their interpretations",
        },
        "assumptions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Modelling assumptions and their justifications",
        },
        "evidence_needed": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Data or domain knowledge required to calibrate the model",
        },
    },
    "required": [
        "problem_type",
        "objective",
        "state_variables",
        "observables",
        "parameters",
        "equations",
        "solver",
        "outputs",
    ],
}

_CREATE_MODEL_SPEC_TOOL: anthropic.types.ToolParam = {
    "name": "create_model_spec",
    "description": (
        "Produce a formal mathematical model specification for the given research idea. "
        "Fill every field as precisely and completely as possible."
    ),
    "input_schema": _MODEL_SPEC_SCHEMA,
}


def generate_model_spec(
    idea: str,
    domain: str | None = None,
    model: str = DEFAULT_MODEL,
) -> ModelSpec:
    """Convert a natural-language research idea into a validated ModelSpec.

    Uses Anthropic tool_use to enforce the JSON schema and then validates
    the result with pydantic.

    Args:
        idea:   Natural-language description of the phenomenon or problem.
        domain: Optional domain hint (e.g. 'finance', 'physics', 'epidemiology').
        model:  Anthropic model ID to use.

    Returns:
        Validated ModelSpec instance.

    Raises:
        KeyError: ANTHROPIC_API_KEY environment variable is not set.
        anthropic.AuthenticationError: API key is invalid.
        ValueError: LLM did not call the tool or returned an unexpected response.
        pydantic.ValidationError: Tool output fails ModelSpec validation.
    """
    api_key = os.environ["ANTHROPIC_API_KEY"]
    client = anthropic.Anthropic(api_key=api_key)

    user_content = f"Research idea: {idea}"
    if domain:
        user_content += f"\nDomain: {domain}"

    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=_SYSTEM_PROMPT,
        tools=[_CREATE_MODEL_SPEC_TOOL],
        tool_choice={"type": "any"},
        messages=[{"role": "user", "content": user_content}],
    )

    # Extract the tool_use block from the response.
    tool_use_block: anthropic.types.ToolUseBlock | None = None
    for block in response.content:
        if block.type == "tool_use" and block.name == "create_model_spec":
            tool_use_block = block
            break

    if tool_use_block is None:
        raise ValueError(
            f"LLM did not call 'create_model_spec'. Response stop_reason={response.stop_reason!r}."
        )

    raw: dict = (
        tool_use_block.input
        if isinstance(tool_use_block.input, dict)
        else json.loads(tool_use_block.input)
    )

    return ModelSpec(**raw)
