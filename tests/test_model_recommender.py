"""Tests for the model recommendation engine (T1400)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from schemas import ModelRecommendation

_MOCK_REC_PAYLOAD: dict = {
    "problem_type": "sequential_decision_under_uncertainty",
    "recommended_models": [
        "bayesian_state_space",
        "bayesian_experimental_design",
        "differentiable_optimization",
    ],
    "rationale": [
        "Latent dynamics exist and must be tracked with a state-space prior.",
        "Online observations allow adaptive experiment selection to maximise information gain.",
        "Next-best action requires a differentiable objective for gradient-based optimisation.",
    ],
}


def _build_mock_response(payload: dict) -> MagicMock:
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.name = "recommend_models"
    tool_block.input = payload

    response = MagicMock()
    response.content = [tool_block]
    response.stop_reason = "tool_use"
    return response


class TestRecommendModels:
    def test_returns_model_recommendation(self) -> None:
        mock_resp = _build_mock_response(_MOCK_REC_PAYLOAD)

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}),
            patch("anthropic.Anthropic") as MockClient,
        ):
            MockClient.return_value.messages.create.return_value = mock_resp
            from models.recommender import recommend_models

            rec = recommend_models(description="A system with hidden state and online observations")

        assert isinstance(rec, ModelRecommendation)

    def test_all_fields_populated(self) -> None:
        mock_resp = _build_mock_response(_MOCK_REC_PAYLOAD)

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}),
            patch("anthropic.Anthropic") as MockClient,
        ):
            MockClient.return_value.messages.create.return_value = mock_resp
            from models.recommender import recommend_models

            rec = recommend_models(description="Sequential decision problem")

        assert rec.problem_type == "sequential_decision_under_uncertainty"
        assert len(rec.recommended_models) == 3
        assert len(rec.rationale) == 3

    def test_rationale_length_mismatch_raises(self) -> None:
        bad_payload = {**_MOCK_REC_PAYLOAD, "rationale": ["Only one reason"]}
        mock_resp = _build_mock_response(bad_payload)

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}),
            patch("anthropic.Anthropic") as MockClient,
        ):
            MockClient.return_value.messages.create.return_value = mock_resp
            from models.recommender import recommend_models

            with pytest.raises(ValueError, match="rationale length"):
                recommend_models(description="Any description")

    def test_signals_included_in_user_content(self) -> None:
        mock_resp = _build_mock_response(_MOCK_REC_PAYLOAD)

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}),
            patch("anthropic.Anthropic") as MockClient,
        ):
            create_mock = MockClient.return_value.messages.create
            create_mock.return_value = mock_resp
            from models.recommender import recommend_models

            recommend_models(
                description="My problem",
                signals=["latent dynamics exist", "online observations are available"],
            )

        call_kwargs = create_mock.call_args[1]
        user_content = call_kwargs["messages"][0]["content"]
        assert "latent dynamics exist" in user_content

    def test_registry_ids_included_in_user_content(self) -> None:
        mock_resp = _build_mock_response(_MOCK_REC_PAYLOAD)
        ids = ["kalman_filter", "gaussian_process", "hawkes_process"]

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}),
            patch("anthropic.Anthropic") as MockClient,
        ):
            create_mock = MockClient.return_value.messages.create
            create_mock.return_value = mock_resp
            from models.recommender import recommend_models

            recommend_models(description="Any", registry_ids=ids)

        call_kwargs = create_mock.call_args[1]
        user_content = call_kwargs["messages"][0]["content"]
        assert "kalman_filter" in user_content

    def test_missing_api_key_raises_key_error(self, monkeypatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        import importlib

        import models.recommender as rec_mod

        importlib.reload(rec_mod)

        with pytest.raises(KeyError):
            rec_mod.recommend_models(description="Test")

    def test_no_tool_use_block_raises_value_error(self) -> None:
        text_block = MagicMock()
        text_block.type = "text"

        response = MagicMock()
        response.content = [text_block]
        response.stop_reason = "end_turn"

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}),
            patch("anthropic.Anthropic") as MockClient,
        ):
            MockClient.return_value.messages.create.return_value = response
            from models.recommender import recommend_models

            with pytest.raises(ValueError, match="recommend_models"):
                recommend_models(description="Some problem")

    def test_tool_input_as_json_string(self) -> None:
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.name = "recommend_models"
        tool_block.input = json.dumps(_MOCK_REC_PAYLOAD)

        response = MagicMock()
        response.content = [tool_block]
        response.stop_reason = "tool_use"

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}),
            patch("anthropic.Anthropic") as MockClient,
        ):
            MockClient.return_value.messages.create.return_value = response
            from models.recommender import recommend_models

            rec = recommend_models(description="Sequential decision")

        assert isinstance(rec, ModelRecommendation)


class TestRecommendEndpoint:
    @pytest.fixture(scope="class")
    def client(self):
        from fastapi.testclient import TestClient

        from api.app import app

        return TestClient(app)

    def test_recommend_returns_200_with_mock(self, client) -> None:
        mock_resp = _build_mock_response(_MOCK_REC_PAYLOAD)

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}),
            patch("anthropic.Anthropic") as MockClient,
        ):
            MockClient.return_value.messages.create.return_value = mock_resp
            resp = client.post(
                "/model/recommend",
                json={"description": "A system with latent state and online measurements"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert "problem_type" in data
        assert "recommended_models" in data
        assert "rationale" in data
        assert len(data["recommended_models"]) == len(data["rationale"])

    def test_recommend_with_signals(self, client) -> None:
        mock_resp = _build_mock_response(_MOCK_REC_PAYLOAD)

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}),
            patch("anthropic.Anthropic") as MockClient,
        ):
            MockClient.return_value.messages.create.return_value = mock_resp
            resp = client.post(
                "/model/recommend",
                json={
                    "description": "Sequential sensor fusion",
                    "signals": ["latent dynamics exist", "online observations are available"],
                },
            )

        assert resp.status_code == 200

    def test_recommend_missing_api_key_returns_503(self, client, monkeypatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        resp = client.post(
            "/model/recommend",
            json={"description": "Any problem"},
        )
        assert resp.status_code == 503

    def test_recommend_empty_description_returns_422(self, client) -> None:
        resp = client.post("/model/recommend", json={"description": ""})
        assert resp.status_code == 422
