"""Tests for models/__main__.py CLI entry point."""

from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock, patch

import pytest


def _run(args: list[str]) -> None:
    """Execute main() with the given argv (excluding script name)."""
    from models.__main__ import main

    with patch.object(sys, "argv", ["src.models"] + args):
        main()


# ---------------------------------------------------------------------------
# main() — top-level dispatch
# ---------------------------------------------------------------------------


class TestMainDispatch:
    def test_no_args_prints_help_and_exits_0(self) -> None:
        with patch.object(sys, "argv", ["src.models"]):
            with pytest.raises(SystemExit) as exc:
                from models.__main__ import main

                main()
        assert exc.value.code == 0

    def test_unknown_command_exits_1(self) -> None:
        with pytest.raises(SystemExit) as exc:
            _run(["__unknown_cmd__"])
        assert exc.value.code == 1

    def test_trace_without_model_id_exits_1(self) -> None:
        with pytest.raises(SystemExit) as exc:
            _run(["trace"])
        assert exc.value.code == 1


# ---------------------------------------------------------------------------
# _cmd_run
# ---------------------------------------------------------------------------


class TestCmdRun:
    def _mock_forge(self) -> tuple[MagicMock, MagicMock]:
        report = MagicMock()
        report.verification.overall.value = "pass"
        report.skeleton_code_path = "artifacts/modelforge/m/impl_skeleton.py"
        report.trace_node_ids = ["node-1", "node-2"]
        forge = MagicMock()
        forge.run.return_value = report
        forge.run_all.return_value = {"m": report}
        return forge, report

    def test_run_single_model(self, capsys) -> None:
        forge, _ = self._mock_forge()
        with patch("models.forge.ModelForge", return_value=forge):
            _run(["run", "test_model"])
        out = capsys.readouterr().out
        assert "PASS" in out or "test_model" in out

    def test_run_all(self, capsys) -> None:
        forge, _ = self._mock_forge()
        with patch("models.forge.ModelForge", return_value=forge):
            _run(["run", "all"])
        forge.run_all.assert_called_once()

    def test_run_defaults_to_all(self) -> None:
        forge, _ = self._mock_forge()
        with patch("models.forge.ModelForge", return_value=forge):
            _run(["run"])
        forge.run_all.assert_called_once()


# ---------------------------------------------------------------------------
# _cmd_verify
# ---------------------------------------------------------------------------


class TestCmdVerify:
    def _mock_pass_report(self) -> MagicMock:
        check = MagicMock()
        check.name = "schema_valid"
        check.status.value = "pass"
        check.message = "ok"
        report = MagicMock()
        report.overall.value = "pass"
        report.registry_hash = "abcdef1234567890"
        report.checks = [check]
        return report

    def test_verify_single(self, capsys) -> None:
        with patch("models.verifier.verify_yaml_file", return_value=self._mock_pass_report()):
            _run(["verify", "my_model"])
        assert "my_model" in capsys.readouterr().out

    def test_verify_all(self, capsys) -> None:
        with patch("models.verifier.verify_all", return_value={"m1": self._mock_pass_report()}):
            _run(["verify", "all"])
        assert "m1" in capsys.readouterr().out

    def test_verify_defaults_to_all(self) -> None:
        with patch("models.verifier.verify_all", return_value={}) as mock_va:
            _run(["verify"])
        mock_va.assert_called_once()

    def test_verify_all_shows_fails_and_warns(self, capsys) -> None:
        fail_check = MagicMock()
        fail_check.name = "equations_present"
        fail_check.status.value = "fail"
        warn_check = MagicMock()
        warn_check.name = "has_references"
        warn_check.status.value = "warn"
        report = MagicMock()
        report.overall.value = "fail"
        report.checks = [fail_check, warn_check]
        with patch("models.verifier.verify_all", return_value={"m1": report}):
            _run(["verify", "all"])
        out = capsys.readouterr().out
        assert "equations_present" in out
        assert "has_references" in out

    def test_verify_single_shows_check_details(self, capsys) -> None:
        check = MagicMock()
        check.name = "solver_specified"
        check.status.value = "pass"
        check.message = "solver ok"
        report = MagicMock()
        report.overall.value = "pass"
        report.registry_hash = "deadbeef12345678"
        report.checks = [check]
        with patch("models.verifier.verify_yaml_file", return_value=report):
            _run(["verify", "my_model"])
        assert "solver_specified" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# _cmd_trace
# ---------------------------------------------------------------------------


class TestCmdTrace:
    def test_trace_with_nodes(self, capsys) -> None:
        import datetime

        node = MagicMock()
        node.node_type.value = "REGISTRY"
        node.node_id = "test_model-registry-abc"
        node.parent_ids = []
        node.artifact_path = "configs/model_registry/test_model.yaml"
        node.content_hash = "deadbeef12345678"
        node.created_at = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
        graph = MagicMock()
        graph.load_model.return_value = [node]
        with patch("models.trace.TraceGraph", return_value=graph):
            _run(["trace", "test_model"])
        assert "test_model" in capsys.readouterr().out

    def test_trace_no_nodes(self, capsys) -> None:
        graph = MagicMock()
        graph.load_model.return_value = []
        with patch("models.trace.TraceGraph", return_value=graph):
            _run(["trace", "missing_model"])
        assert "No trace nodes" in capsys.readouterr().out

    def test_trace_with_parent_ids(self, capsys) -> None:
        import datetime

        node = MagicMock()
        node.node_type.value = "VERIFICATION"
        node.node_id = "test_model-verification-xyz"
        node.parent_ids = ["test_model-registry-abc"]
        node.artifact_path = "artifacts/modelforge/test_model/verification.json"
        node.content_hash = "cafebabe12345678"
        node.created_at = datetime.datetime(2026, 1, 2, tzinfo=datetime.timezone.utc)
        graph = MagicMock()
        graph.load_model.return_value = [node]
        with patch("models.trace.TraceGraph", return_value=graph):
            _run(["trace", "test_model"])
        assert "test_model-registry-abc" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# _cmd_audit
# ---------------------------------------------------------------------------


class TestCmdAudit:
    def test_audit_no_log_file(self, capsys, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        _run(["audit"])
        assert "No ModelForge audit log" in capsys.readouterr().out

    def test_audit_with_log_file(self, capsys, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        audit_dir = tmp_path / ".claude" / "audit"
        audit_dir.mkdir(parents=True)
        entry = {
            "timestamp": "2026-01-01T00:00:00Z",
            "event": "forge_run",
            "model_id": "test_model",
        }
        (audit_dir / "modelforge.jsonl").write_text(json.dumps(entry) + "\n", encoding="utf-8")
        _run(["audit"])
        assert "test_model" in capsys.readouterr().out

    def test_audit_shows_verification_overall(self, capsys, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        audit_dir = tmp_path / ".claude" / "audit"
        audit_dir.mkdir(parents=True)
        entry = {
            "timestamp": "2026-01-01T00:00:00Z",
            "event": "forge_run",
            "model_id": "test_model",
            "verification_overall": "pass",
        }
        (audit_dir / "modelforge.jsonl").write_text(json.dumps(entry) + "\n", encoding="utf-8")
        _run(["audit"])
        assert "pass" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# _cmd_report  (asyncio.run is called internally)
# ---------------------------------------------------------------------------


class TestCmdReport:
    def test_report_single_model(self, capsys) -> None:
        mock_result = {
            "model_id": "test_model",
            "status": "ok",
            "report_path": "reports/modelforge/test_model.md",
        }
        with patch("asyncio.run", return_value=mock_result):
            _run(["report", "test_model"])
        assert "test_model" in capsys.readouterr().out

    def test_report_all(self, capsys) -> None:
        mock_result = {
            "results": {
                "m1": {"status": "ok", "report_path": "reports/modelforge/m1.md"},
                "m2": {"status": "error", "error": "no artifacts"},
            }
        }
        with patch("asyncio.run", return_value=mock_result):
            _run(["report", "all"])
        out = capsys.readouterr().out
        assert "m1" in out
        assert "m2" in out

    def test_report_defaults_to_all(self, capsys) -> None:
        mock_result = {"results": {}}
        with patch("asyncio.run", return_value=mock_result):
            _run(["report"])
        # No crash on empty results

    def test_report_error_status(self, capsys) -> None:
        mock_result = {
            "model_id": "bad_model",
            "status": "error",
            "error": "No forge artifacts found",
        }
        with patch("asyncio.run", return_value=mock_result):
            _run(["report", "bad_model"])
        assert "bad_model" in capsys.readouterr().out
