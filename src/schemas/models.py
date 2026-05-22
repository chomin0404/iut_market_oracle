"""T1400 Mathematical Model and Formalize-Idea schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ModelSpec(BaseModel):
    """Formal specification of a mathematical model."""

    problem_type: str = Field(..., min_length=1, description="Class of mathematical problem")
    objective: str = Field(..., min_length=1, description="Optimisation or inference objective")
    state_variables: list[str] = Field(..., description="Latent/state variables and their domains")
    observables: list[str] = Field(..., description="Observed/measured quantities")
    parameters: list[str] = Field(..., description="Model parameters to be calibrated")
    constraints: list[str] = Field(default_factory=list, description="Mathematical constraints")
    uncertainty: dict[str, str] = Field(
        default_factory=dict, description="Noise/uncertainty specifications"
    )
    equations: list[str] = Field(..., description="Key equations in mathematical notation")
    priors: dict[str, str] = Field(
        default_factory=dict, description="Prior distributions over parameters"
    )
    loss_function: str | None = Field(None, description="Loss or energy function if applicable")
    solver: str = Field(..., min_length=1, description="Recommended solver or inference algorithm")
    outputs: list[str] = Field(..., description="Model outputs and their interpretations")
    assumptions: list[str] = Field(default_factory=list, description="Modelling assumptions")
    evidence_needed: list[str] = Field(
        default_factory=list, description="Data or domain knowledge required"
    )


class ModelRegistryEntry(ModelSpec):
    """ModelSpec augmented with registry metadata."""

    id: str = Field(..., min_length=1, description="snake_case unique identifier")
    name: str = Field(..., min_length=1, description="Human-readable model name")
    category: str = Field(default="", description="Top-level domain category")
    tags: list[str] = Field(default_factory=list, description="Searchable keyword tags")
    references: list[str] = Field(default_factory=list, description="Key citations")


class ModelRecommendation(BaseModel):
    """LLM-generated model recommendation for a given problem description."""

    problem_type: str = Field(..., min_length=1, description="Inferred class of the problem")
    recommended_models: list[str] = Field(
        ..., min_length=1, description="Ordered list of recommended model identifiers"
    )
    rationale: list[str] = Field(
        ..., min_length=1, description="Reasons each model class is appropriate"
    )


class IdeaInput(BaseModel):
    title: str = Field(..., min_length=3, max_length=200, description="Short problem title.")
    description: str = Field(..., min_length=10, description="Natural-language problem statement.")
    domain: str | None = Field(default=None, description="Optional domain label.")
    goal_type: Literal[
        "prediction",
        "optimization",
        "control",
        "anomaly_detection",
        "simulation",
        "causal_inference",
        "decision_support",
    ] = Field(default="prediction", description="Primary modeling goal.")
    time_horizon: Literal["static", "sequential", "continuous"] = Field(
        default="static",
        description="Temporal structure of the problem.",
    )
    data_regime: Literal["small", "medium", "large", "unknown"] = Field(
        default="unknown",
        description="Approximate amount of available data.",
    )
    uncertainty_level: Literal["low", "medium", "high", "unknown"] = Field(
        default="unknown",
        description="Expected uncertainty level.",
    )
    physical_constraints: bool = Field(
        default=False,
        description="Whether the system must respect physical laws or domain constraints.",
    )
    decision_variables_present: bool = Field(
        default=False,
        description="Whether optimization or control variables are present.",
    )
    latent_state_present: bool = Field(
        default=False,
        description="Whether hidden system states are believed to exist.",
    )


class ProblemStructure(BaseModel):
    """Typed boolean flags characterising the mathematical structure of a problem."""

    is_sequential: bool = Field(..., description="Observations arrive over time / system evolves.")
    has_latent_state: bool = Field(..., description="Hidden/unobserved state variables exist.")
    has_decision_variables: bool = Field(
        ..., description="Problem involves optimisation or control actions."
    )
    has_physical_constraints: bool = Field(
        ..., description="Physical laws or hard domain constraints apply."
    )
    is_high_uncertainty: bool = Field(..., description="uncertainty_level == 'high'.")
    is_data_scarce: bool = Field(..., description="data_regime == 'small'.")


class ParsedIdeaResponse(BaseModel):
    problem_structure: ProblemStructure
    candidate_families: list[str]
    missing_information: list[str]
