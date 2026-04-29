"""Tests for the LLM-based model generator (T1400)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from schemas import ModelSpec

# Minimal valid ModelSpec payload returned by the mock tool_use block.
_MOCK_SPEC_PAYLOAD: dict = {
    "problem_type": "Bayesian state estimation",
    "objective": "Minimise mean squared estimation error",
    "state_variables": ["x_t: latent state"],
    "observables": ["z_t: noisy measurement"],
    "parameters": ["F: transition matrix", "H: observation matrix"],
    "constraints": ["P_t positive semi-definite"],
    "uncertainty": {"process": "w_t ~ N(0, Q)", "observation": "v_t ~ N(0, R)"},
    "equations": ["x_t = F x_{t-1} + w_t", "z_t = H x_t + v_t"],
    "priors": {"x_0": "N(0, I)"},
    "loss_function": "tr(P_{t|t})",
    "solver": "Kalman recursion",
    "outputs": ["x̂_t: filtered estimate", "P_t: posterior covariance"],
    "assumptions": ["Linear Gaussian model"],
    "evidence_needed": ["Q, R covariance matrices"],
}


def _build_mock_response(payload: dict) -> MagicMock:
    """Construct a mock anthropic.types.Message with a single tool_use block."""
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.name = "create_model_spec"
    tool_block.input = payload

    response = MagicMock()
    response.content = [tool_block]
    response.stop_reason = "tool_use"
    return response


class TestGenerateModelSpec:
    def test_returns_model_spec(self) -> None:
        mock_response = _build_mock_response(_MOCK_SPEC_PAYLOAD)

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}),
            patch("anthropic.Anthropic") as MockClient,
        ):
            MockClient.return_value.messages.create.return_value = mock_response
            from models.generator import generate_model_spec

            spec = generate_model_spec(idea="A linear sensor fusion system")

        assert isinstance(spec, ModelSpec)

    def test_all_required_fields_populated(self) -> None:
        mock_response = _build_mock_response(_MOCK_SPEC_PAYLOAD)

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}),
            patch("anthropic.Anthropic") as MockClient,
        ):
            MockClient.return_value.messages.create.return_value = mock_response
            from models.generator import generate_model_spec

            spec = generate_model_spec(idea="A linear sensor fusion system")

        assert spec.problem_type
        assert spec.objective
        assert len(spec.state_variables) >= 1
        assert len(spec.observables) >= 1
        assert len(spec.parameters) >= 1
        assert len(spec.equations) >= 1
        assert spec.solver
        assert len(spec.outputs) >= 1

    def test_domain_hint_passed_to_api(self) -> None:
        mock_response = _build_mock_response(_MOCK_SPEC_PAYLOAD)

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}),
            patch("anthropic.Anthropic") as MockClient,
        ):
            create_mock = MockClient.return_value.messages.create
            create_mock.return_value = mock_response
            from models.generator import generate_model_spec

            generate_model_spec(idea="Mean reversion strategy", domain="finance")

        call_kwargs = create_mock.call_args
        messages = call_kwargs[1]["messages"] if call_kwargs[1] else call_kwargs[0][4]
        user_content = messages[0]["content"]
        assert "finance" in user_content.lower()

    def test_missing_api_key_raises_key_error(self, monkeypatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        # Re-import to avoid module-level caching issues.
        import importlib

        import models.generator as gen_mod

        importlib.reload(gen_mod)

        with pytest.raises(KeyError):
            gen_mod.generate_model_spec(idea="Test idea")

    def test_no_tool_use_block_raises_value_error(self) -> None:
        """If the LLM returns no tool_use block, ValueError must be raised."""
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Here is my answer..."

        response = MagicMock()
        response.content = [text_block]
        response.stop_reason = "end_turn"

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}),
            patch("anthropic.Anthropic") as MockClient,
        ):
            MockClient.return_value.messages.create.return_value = response
            from models.generator import generate_model_spec

            with pytest.raises(ValueError, match="create_model_spec"):
                generate_model_spec(idea="Some idea")

    def test_tool_input_as_json_string(self) -> None:
        """input field may arrive as a JSON string (some SDK versions); must be handled."""
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.name = "create_model_spec"
        tool_block.input = json.dumps(_MOCK_SPEC_PAYLOAD)  # string, not dict

        response = MagicMock()
        response.content = [tool_block]
        response.stop_reason = "tool_use"

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}),
            patch("anthropic.Anthropic") as MockClient,
        ):
            MockClient.return_value.messages.create.return_value = response
            from models.generator import generate_model_spec

            spec = generate_model_spec(idea="Sensor fusion")

        assert isinstance(spec, ModelSpec)
