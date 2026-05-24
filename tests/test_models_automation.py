"""Additional coverage tests for models/automation.py.

Covers:
  - handle_registry_changed: verify-error path, forge-error path, explicit yaml_path
  - handle_forge_requested: single model, all models, forge error
  - handle_report_requested: with artifacts, without artifacts, explicit project_dir
  - _cli_main: valid event, invalid JSON, missing arg, unknown event
"""

from __future__ import annotations

import asyncio
import json
import sys
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_VALID_YAML = textwrap.dedent("""\
    id: test_model
    name: Test Model
    category: regression
    problem_type: Ordinary least squares regression
    objective: Minimize squared residuals
    state_variables:
      - "beta: coefficient vector"
    observables:
      - "y: response"
      - "X: design matrix"
    solver: Closed-form normal equations
    equations:
      - "beta = (X^T X)^{-1} X^T y"
    outputs:
      - "beta: coefficient vector"
    parameters:
      - "beta: regression coefficients"
    references:
      - "Gauss, C.F. (1809)."
""")


def _make_registry(tmp_path: Path) -> Path:
    reg_dir = tmp_path / "configs" / "model_registry"
    reg_dir.mkdir(parents=True)
    (reg_dir / "test_model.yaml").write_text(_VALID_YAML, encoding="utf-8")
    return reg_dir


# ---------------------------------------------------------------------------
# handle_registry_changed — error paths
# ---------------------------------------------------------------------------


class TestHandleRegistryChangedErrors:
    def test_verify_error_on_missing_yaml(self, tmp_path) -> None:
        """YAML file missing → verify raises OSError → result has error key."""
        from models import automation

        result = asyncio.run(
            automation.handle_registry_changed(
                {
                    "model_id": "ghost_model",
                    "yaml_path": str(tmp_path / "nonexistent.yaml"),
                }
            )
        )
        assert result["verification_overall"] == "error"
        assert "error" in result

    def test_forge_error_path(self, tmp_path, monkeypatch) -> None:
        """Verification passes but ModelForge.run raises → forge_status = error."""
        from models import automation

        reg_dir = _make_registry(tmp_path)
        monkeypatch.setattr(automation, "_REGISTRY_DIR", reg_dir)

        mock_forge = MagicMock()
        mock_forge.run.side_effect = RuntimeError("forge exploded")
        with patch("models.automation.ModelForge", return_value=mock_forge):
            result = asyncio.run(
                automation.handle_registry_changed({"model_id": "test_model"})
            )
        assert result.get("forge_status") == "error"
        assert "forge_error" in result

    def test_explicit_yaml_path_overrides_default(self, tmp_path, monkeypatch) -> None:
        """Passing yaml_path explicitly uses that file, not _REGISTRY_DIR lookup."""
        from models import automation
        from models.forge import ModelForge

        reg_dir = _make_registry(tmp_path)
        monkeypatch.setattr(automation, "_REGISTRY_DIR", reg_dir)

        original_init = ModelForge.__init__

        def patched_init(self, project_dir=None, registry_dir=None, artifacts_dir=None):
            original_init(
                self,
                project_dir=tmp_path,
                registry_dir=reg_dir,
                artifacts_dir=tmp_path / "artifacts" / "modelforge",
            )

        monkeypatch.setattr(ModelForge, "__init__", patched_init)
        yaml_path = reg_dir / "test_model.yaml"

        result = asyncio.run(
            automation.handle_registry_changed(
                {"model_id": "test_model", "yaml_path": str(yaml_path)}
            )
        )
        assert result["model_id"] == "test_model"
        assert "verification_overall" in result


# ---------------------------------------------------------------------------
# handle_forge_requested
# ---------------------------------------------------------------------------


class TestHandleForgeRequested:
    def test_forge_single_model(self, tmp_path, monkeypatch) -> None:
        from models import automation
        from models.forge import ModelForge

        reg_dir = _make_registry(tmp_path)
        monkeypatch.setattr(automation, "_REGISTRY_DIR", reg_dir)

        original_init = ModelForge.__init__

        def patched_init(self, project_dir=None, registry_dir=None, artifacts_dir=None):
            original_init(
                self,
                project_dir=tmp_path,
                registry_dir=reg_dir,
                artifacts_dir=tmp_path / "artifacts" / "modelforge",
            )

        monkeypatch.setattr(ModelForge, "__init__", patched_init)

        result = asyncio.run(
            automation.handle_forge_requested({"model_id": "test_model"})
        )
        assert result.get("forge_status") == "ok"
        assert "verification_overall" in result

    def test_forge_all_models(self, tmp_path, monkeypatch) -> None:
        from models import automation
        from models.forge import ModelForge

        reg_dir = _make_registry(tmp_path)
        monkeypatch.setattr(automation, "_REGISTRY_DIR", reg_dir)

        original_init = ModelForge.__init__

        def patched_init(self, project_dir=None, registry_dir=None, artifacts_dir=None):
            original_init(
                self,
                project_dir=tmp_path,
                registry_dir=reg_dir,
                artifacts_dir=tmp_path / "artifacts" / "modelforge",
            )

        monkeypatch.setattr(ModelForge, "__init__", patched_init)

        result = asyncio.run(automation.handle_forge_requested({"model_id": "all"}))
        assert "results" in result
        assert "test_model" in result["results"]

    def test_forge_single_error(self, tmp_path, monkeypatch) -> None:
        from models import automation

        reg_dir = _make_registry(tmp_path)
        monkeypatch.setattr(automation, "_REGISTRY_DIR", reg_dir)

        mock_forge = MagicMock()
        mock_forge.run.side_effect = ValueError("bad model")
        with patch("models.automation.ModelForge", return_value=mock_forge):
            result = asyncio.run(
                automation.handle_forge_requested({"model_id": "test_model"})
            )
        assert result.get("forge_status") == "error"
        assert "error" in result

    def test_forge_defaults_to_all(self, tmp_path, monkeypatch) -> None:
        from models import automation

        mock_forge = MagicMock()
        mock_forge.run_all.return_value = {}
        with patch("models.automation.ModelForge", return_value=mock_forge):
            result = asyncio.run(automation.handle_forge_requested({}))
        mock_forge.run_all.assert_called_once()
        assert "results" in result


# ---------------------------------------------------------------------------
# handle_report_requested
# ---------------------------------------------------------------------------


class TestHandleReportRequested:
    def _make_artifacts(self, tmp_path: Path, model_id: str = "test_model") -> Path:
        """Create minimal forge artifacts under tmp_path."""
        import yaml

        artifacts_dir = tmp_path / "artifacts" / "modelforge"
        model_dir = artifacts_dir / model_id
        model_dir.mkdir(parents=True)

        # spec_snapshot.yaml
        spec = {
            "id": model_id,
            "name": "Test Model",
            "problem_type": "regression",
            "solver": "ols",
            "equations": ["beta = (X^T X)^{-1} X^T y"],
            "assumptions": ["iid errors"],
        }
        (model_dir / "spec_snapshot.yaml").write_text(
            yaml.dump(spec), encoding="utf-8"
        )

        # verification.json
        verification = {
            "overall": "pass",
            "checks": [{"name": "schema_valid", "status": "pass", "message": None}],
        }
        (model_dir / "verification.json").write_text(
            json.dumps(verification), encoding="utf-8"
        )

        # impl_skeleton.py
        (model_dir / "impl_skeleton.py").write_text(
            "# skeleton\ndef run(): pass\n", encoding="utf-8"
        )

        return artifacts_dir

    def test_report_with_full_artifacts(self, tmp_path) -> None:
        from models import automation

        self._make_artifacts(tmp_path, "test_model")

        result = asyncio.run(
            automation.handle_report_requested(
                {"model_id": "test_model", "project_dir": str(tmp_path)}
            )
        )
        assert result.get("status") == "ok"
        assert "report_path" in result
        # Verify the markdown file was written
        report_file = tmp_path / result["report_path"]
        assert report_file.exists()
        content = report_file.read_text(encoding="utf-8")
        assert "test_model" in content

    def test_report_no_artifacts_returns_error(self, tmp_path) -> None:
        from models import automation

        result = asyncio.run(
            automation.handle_report_requested(
                {"model_id": "ghost_model", "project_dir": str(tmp_path)}
            )
        )
        assert result.get("status") == "error"
        assert "error" in result

    def test_report_all_models(self, tmp_path) -> None:
        from models import automation

        reg_dir = _make_registry(tmp_path)
        self._make_artifacts(tmp_path, "test_model")

        result = asyncio.run(
            automation.handle_report_requested(
                {"model_id": "all", "project_dir": str(tmp_path)}
            )
        )
        assert "results" in result
        assert "test_model" in result["results"]

    def test_report_with_trace_nodes(self, tmp_path) -> None:
        """trace.jsonl present → traceability section rendered."""
        from models import automation
        from models.trace import TraceGraph

        artifacts_dir = self._make_artifacts(tmp_path, "test_model")

        # Write a minimal trace entry so TraceGraph.load_model returns nodes
        reg_dir = _make_registry(tmp_path)
        forge_dir = artifacts_dir
        # Use the real forge to generate a trace entry
        from models.forge import ModelForge

        forge = ModelForge(
            project_dir=tmp_path,
            registry_dir=reg_dir,
            artifacts_dir=forge_dir,
        )
        forge.run("test_model")

        result = asyncio.run(
            automation.handle_report_requested(
                {"model_id": "test_model", "project_dir": str(tmp_path)}
            )
        )
        assert result.get("status") == "ok"


# ---------------------------------------------------------------------------
# _cli_main
# ---------------------------------------------------------------------------


class TestCliMain:
    def test_no_args_exits_1(self) -> None:
        with patch.object(sys, "argv", ["models.automation"]):
            with pytest.raises(SystemExit) as exc:
                from models.automation import _cli_main

                _cli_main()
        assert exc.value.code == 1

    def test_invalid_json_exits_1(self) -> None:
        with patch.object(sys, "argv", ["models.automation", "verify_requested", "NOT_JSON"]):
            with pytest.raises(SystemExit) as exc:
                from models.automation import _cli_main

                _cli_main()
        assert exc.value.code == 1

    def test_valid_dispatch(self, tmp_path, monkeypatch, capsys) -> None:
        from models import automation

        reg_dir = _make_registry(tmp_path)
        monkeypatch.setattr(automation, "_REGISTRY_DIR", reg_dir)

        payload = json.dumps({"model_id": "test_model"})
        with patch.object(sys, "argv", ["models.automation", "verify_requested", payload]):
            from models.automation import _cli_main

            _cli_main()
        out = capsys.readouterr().out
        result = json.loads(out)
        assert "overall" in result

    def test_unknown_event_raises_value_error(self) -> None:
        """dispatch raises ValueError for unknown event type — propagates from _cli_main."""
        payload = json.dumps({"model_id": "x"})
        with patch.object(sys, "argv", ["models.automation", "bad_event", payload]):
            from models.automation import _cli_main

            with pytest.raises(ValueError, match="Unknown event type"):
                _cli_main()
