"""LLM-based idea formalizer (Formalize-Idea skill backend).

Takes a structured IdeaInput, infers the mathematical problem structure,
and returns a ParsedIdeaResponse with candidate model families and gaps.

Environment:
    ANTHROPIC_API_KEY: required; raises KeyError if absent.
"""

from __future__ import annotations

import json
import os

import anthropic

from schemas import IdeaInput, ParsedIdeaResponse

DEFAULT_MODEL = "claude-sonnet-4-6"

_SYSTEM_PROMPT = """\
You are an expert in applied mathematics, Bayesian inference, and mathematical modelling.
Given a structured description of a research idea, you must:

1. Classify the problem structure into boolean flags (problem_structure).
2. Recommend candidate mathematical model families ordered by suitability (candidate_families).
3. List missing information or domain knowledge needed to proceed (missing_information).

Guidelines for problem_structure flags:
- is_sequential: True if observations arrive over time or the system evolves temporally.
- has_latent_state: True if hidden/unobserved state variables are believed to exist.
- has_decision_variables: True if the problem involves optimisation or control actions.
- has_physical_constraints: True if physical laws or hard domain constraints apply.
- is_high_uncertainty: True if the uncertainty_level is 'high'.
- is_data_scarce: True if data_regime is 'small'.

Guidelines for candidate_families:
- Use canonical snake_case identifiers (e.g. kalman_filter, particle_filter, mdp).
- Order from most to least suitable given the problem structure.
- Include at least two candidates.

Always call the `parse_idea` tool with your complete analysis.
"""

_TOOL_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "problem_structure": {
            "type": "object",
            "additionalProperties": {"type": "boolean"},
            "description": (
                "Boolean flags characterising the mathematical structure of the problem. "
                "Must include: is_sequential, has_latent_state, has_decision_variables, "
                "has_physical_constraints, is_high_uncertainty, is_data_scarce."
            ),
        },
        "candidate_families": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 2,
            "description": (
                "Ordered list of candidate mathematical model families (snake_case). "
                "Most suitable first."
            ),
        },
        "missing_information": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Data, measurements, or domain knowledge needed before modelling can proceed. "
                "Empty list if nothing critical is missing."
            ),
        },
    },
    "required": ["problem_structure", "candidate_families", "missing_information"],
}

_PARSE_IDEA_TOOL: anthropic.types.ToolParam = {
    "name": "parse_idea",
    "description": (
        "Analyse a structured research idea and return its mathematical problem structure, "
        "candidate model families, and missing information."
    ),
    "input_schema": _TOOL_SCHEMA,
}


async def parse_idea(idea: IdeaInput, model: str = DEFAULT_MODEL) -> ParsedIdeaResponse:
    """Formalise a structured IdeaInput into a ParsedIdeaResponse (async).

    Args:
        idea:  Validated IdeaInput describing the research problem.
        model: Anthropic model ID to use.

    Returns:
        ParsedIdeaResponse with problem_structure, candidate_families,
        and missing_information.

    Raises:
        KeyError:              ANTHROPIC_API_KEY is not set.
        ValueError:            LLM did not call the tool or returned unexpected structure.
        pydantic.ValidationError: Tool output fails ParsedIdeaResponse validation.
    """
    api_key = os.environ["ANTHROPIC_API_KEY"]
    client = anthropic.AsyncAnthropic(api_key=api_key)

    user_content = (
        f"Title: {idea.title}\n"
        f"Description: {idea.description}\n"
        f"Goal type: {idea.goal_type}\n"
        f"Time horizon: {idea.time_horizon}\n"
        f"Data regime: {idea.data_regime}\n"
        f"Uncertainty level: {idea.uncertainty_level}\n"
        f"Physical constraints: {idea.physical_constraints}\n"
        f"Decision variables present: {idea.decision_variables_present}\n"
        f"Latent state present: {idea.latent_state_present}\n"
    )
    if idea.domain:
        user_content += f"Domain: {idea.domain}\n"

    response = await client.messages.create(
        model=model,
        max_tokens=1024,
        system=_SYSTEM_PROMPT,
        tools=[_PARSE_IDEA_TOOL],
        tool_choice={"type": "any"},
        messages=[{"role": "user", "content": user_content}],
    )

    tool_block: anthropic.types.ToolUseBlock | None = None
    for block in response.content:
        if block.type == "tool_use" and block.name == "parse_idea":
            tool_block = block
            break

    if tool_block is None:
        raise ValueError(
            f"LLM did not call 'parse_idea'. Response stop_reason={response.stop_reason!r}."
        )

    raw: dict = (
        tool_block.input if isinstance(tool_block.input, dict) else json.loads(tool_block.input)
    )

    return ParsedIdeaResponse(**raw)
