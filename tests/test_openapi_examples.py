"""Regression tests that pin OpenAPI examples after the Body/custom_openapi refactor.

Two verification layers:

1. **Operation-level layer** (``Body(openapi_examples=...)``):
   Endpoints that accept a Pydantic model as request body expose per-operation
   examples.  These appear under
   ``paths/<route>/post/requestBody/content/application/json/examples``
   in the live OpenAPI document.  Covers ``IdeaInput`` (``POST /ideas/parse``),
   ``RecommendRequest`` (``POST /model/recommend``), and ``GenerateRequest``
   (``POST /model/generate``).

2. **Component-level layer** (``custom_openapi()``):
   Response-only schemas cannot use ``Body()``, so their examples are injected
   via the ``_custom_openapi`` hook in ``api.app``.  These appear under
   ``components/schemas/<Name>/examples`` in the live OpenAPI document.
   Covers ``ModelRecommendation``, ``ModelSpec``, and ``ParsedIdeaResponse``.

If an example is accidentally removed, mutated, or the constant is
de-referenced, these tests will catch it before the change reaches the docs UI.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.app import app
from modeling_api.examples import (
    EXAMPLE_IDEA_INPUT,
    EXAMPLE_MODEL_RECOMMENDATION,
    EXAMPLE_MODEL_SPEC,
    EXAMPLE_PARSED_IDEA_RESPONSE,
)

client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _op_examples(openapi: dict, path: str, method: str = "post") -> dict:
    """Return the ``examples`` dict from a request-body operation."""
    op = openapi["paths"][path][method]
    return (
        op.get("requestBody", {})
        .get("content", {})
        .get("application/json", {})
        .get("examples", {})
    )


def _component_example(openapi: dict, schema_name: str) -> dict:
    """Return ``examples[0]`` from an OpenAPI component schema."""
    schemas = openapi.get("components", {}).get("schemas", {})
    assert schema_name in schemas, (
        f"Component schema '{schema_name}' not found in /openapi.json — "
        "is the model wired to a FastAPI route?"
    )
    examples = schemas[schema_name].get("examples")
    assert examples, f"'examples' missing or empty in OpenAPI component '{schema_name}'"
    return examples[0]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def openapi() -> dict:
    r = client.get("/openapi.json")
    assert r.status_code == 200
    return r.json()


# ---------------------------------------------------------------------------
# Layer 1 — Operation-level examples (Body openapi_examples)
# ---------------------------------------------------------------------------


class TestOperationExamples:
    """Verify Body(openapi_examples=...) examples appear in the live OpenAPI document."""

    # POST /ideas/parse -------------------------------------------------------

    def test_ideas_parse_has_gps_spoofing_example(self, openapi: dict) -> None:
        examples = _op_examples(openapi, "/ideas/parse")
        assert "gps_spoofing" in examples

    def test_ideas_parse_example_value_matches_constant(self, openapi: dict) -> None:
        value = _op_examples(openapi, "/ideas/parse")["gps_spoofing"]["value"]
        assert value == EXAMPLE_IDEA_INPUT

    def test_ideas_parse_example_title(self, openapi: dict) -> None:
        value = _op_examples(openapi, "/ideas/parse")["gps_spoofing"]["value"]
        assert value["title"] == "GPS spoofing detection and defense"

    def test_ideas_parse_example_goal_type(self, openapi: dict) -> None:
        value = _op_examples(openapi, "/ideas/parse")["gps_spoofing"]["value"]
        assert value["goal_type"] == "anomaly_detection"

    def test_ideas_parse_example_time_horizon(self, openapi: dict) -> None:
        value = _op_examples(openapi, "/ideas/parse")["gps_spoofing"]["value"]
        assert value["time_horizon"] == "sequential"

    # POST /model/recommend ---------------------------------------------------

    def test_model_recommend_has_gps_spoofing_example(self, openapi: dict) -> None:
        examples = _op_examples(openapi, "/model/recommend")
        assert "gps_spoofing" in examples

    def test_model_recommend_example_has_signals(self, openapi: dict) -> None:
        value = _op_examples(openapi, "/model/recommend")["gps_spoofing"]["value"]
        assert isinstance(value.get("signals"), list)
        assert len(value["signals"]) > 0

    # POST /model/generate ----------------------------------------------------

    def test_model_generate_has_gps_spoofing_example(self, openapi: dict) -> None:
        examples = _op_examples(openapi, "/model/generate")
        assert "gps_spoofing" in examples

    def test_model_generate_example_has_domain(self, openapi: dict) -> None:
        value = _op_examples(openapi, "/model/generate")["gps_spoofing"]["value"]
        assert value.get("domain") == "navigation_security"


# ---------------------------------------------------------------------------
# Layer 2 — Component-level examples (custom_openapi response schemas)
# ---------------------------------------------------------------------------


class TestComponentExamples:
    """Verify custom_openapi() injects examples into response-only component schemas."""

    # ModelRecommendation -----------------------------------------------------

    def test_model_recommendation_example_matches_constant(self, openapi: dict) -> None:
        assert _component_example(openapi, "ModelRecommendation") == EXAMPLE_MODEL_RECOMMENDATION

    def test_model_recommendation_problem_type(self, openapi: dict) -> None:
        ex = _component_example(openapi, "ModelRecommendation")
        assert ex["problem_type"] == "sequential_anomaly_detection"

    def test_model_recommendation_contains_ekf(self, openapi: dict) -> None:
        ex = _component_example(openapi, "ModelRecommendation")
        assert "extended_kalman_filter" in ex["recommended_models"]

    def test_model_recommendation_rationale_length_matches_models(self, openapi: dict) -> None:
        ex = _component_example(openapi, "ModelRecommendation")
        assert len(ex["rationale"]) == len(ex["recommended_models"])

    # ModelSpec ---------------------------------------------------------------

    def test_model_spec_example_matches_constant(self, openapi: dict) -> None:
        assert _component_example(openapi, "ModelSpec") == EXAMPLE_MODEL_SPEC

    def test_model_spec_problem_type(self, openapi: dict) -> None:
        ex = _component_example(openapi, "ModelSpec")
        assert ex["problem_type"] == "sequential_anomaly_detection"

    def test_model_spec_solver_mentions_kalman(self, openapi: dict) -> None:
        ex = _component_example(openapi, "ModelSpec")
        assert "Kalman" in ex["solver"]

    def test_model_spec_loss_function_present(self, openapi: dict) -> None:
        ex = _component_example(openapi, "ModelSpec")
        assert ex.get("loss_function") is not None

    def test_model_spec_required_fields_present(self, openapi: dict) -> None:
        ex = _component_example(openapi, "ModelSpec")
        for field in ("objective", "state_variables", "observables", "parameters",
                      "equations", "solver", "outputs"):
            assert field in ex

    # ParsedIdeaResponse ------------------------------------------------------

    def test_parsed_idea_response_example_matches_constant(self, openapi: dict) -> None:
        assert _component_example(openapi, "ParsedIdeaResponse") == EXAMPLE_PARSED_IDEA_RESPONSE

    def test_parsed_idea_response_has_all_structure_keys(self, openapi: dict) -> None:
        ps = _component_example(openapi, "ParsedIdeaResponse")["problem_structure"]
        for key in (
            "is_sequential",
            "has_latent_state",
            "has_decision_variables",
            "has_physical_constraints",
            "is_high_uncertainty",
            "is_data_scarce",
        ):
            assert key in ps

    def test_parsed_idea_response_candidate_families_has_kalman(self, openapi: dict) -> None:
        ex = _component_example(openapi, "ParsedIdeaResponse")
        assert "kalman_filter" in ex["candidate_families"]
