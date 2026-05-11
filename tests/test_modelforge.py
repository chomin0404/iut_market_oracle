"""Tests for ModelForge — verifier, trace, forge, automation, and API."""

from __future__ import annotations

import asyncio
import hashlib
import json
import textwrap
from pathlib import Path

import pytest
import yaml

# ---------------------------------------------------------------------------
# Fixtures: minimal YAML content
# ---------------------------------------------------------------------------

_VALID_YAML = textwrap.dedent("""\
    id: test_model
    name: Test Model
    category: regression
    problem_type: Ordinary least squares regression
    objective: Minimize squared residuals ||y - X beta||^2
    state_variables:
      - "beta: coefficient vector ∈ ℝ^p"
    observables:
      - "y: response variable"
      - "X: design matrix"
    solver: Closed-form normal equations
    equations:
      - "beta = (X^T X)^{-1} X^T y"
    outputs:
      - "beta: coefficient vector"
    parameters:
      - "beta: regression coefficients ∈ ℝ^p"
    references:
      - "Gauss, C.F. (1809). Theoria Motus."
""")

_VALID_YAML_NO_REFS = textwrap.dedent("""\
    id: no_refs_model
    name: No Refs Model
    category: regression
    problem_type: Simple test model without references
    objective: Fit line to data
    state_variables:
      - "y_hat: predicted value"
    observables:
      - "x: predictor"
      - "y: response"
    solver: brute-force
    equations:
      - "y = a x + b"
    outputs:
      - "y: prediction"
    parameters:
      - "a: slope"
      - "b: intercept"
""")

_YAML_MISSING_EQUATIONS = textwrap.dedent("""\
    id: no_eq_model
    name: No Equations Model
    category: regression
    problem_type: test
    objective: test objective
    state_variables:
      - "s: state"
    observables:
      - "o: observation"
    solver: something
    equations: []
    outputs:
      - "y: output"
    parameters: []
""")

_YAML_MISSING_SOLVER = textwrap.dedent("""\
    id: no_solver_model
    name: No Solver
    category: test
    problem_type: test
    objective: test objective
    state_variables:
      - "s: state"
    observables:
      - "o: observation"
    solver: " "
    equations:
      - "y = x"
    outputs:
      - "y: output"
    parameters: []
""")

_YAML_INVALID_SCHEMA = "not_a_valid_yaml_for_model_registry: true\n"

_KALMAN_YAML_PATH = Path("configs/model_registry/kalman_filter.yaml")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def write_yaml(tmp_path: Path, content: str, filename: str = "test_model.yaml") -> Path:
    p = tmp_path / filename
    p.write_text(content, encoding="utf-8")
    return p


# ===========================================================================
# verifier.py tests
# ===========================================================================


class TestVerifyYamlFile:
    def test_valid_yaml_passes(self, tmp_path: Path) -> None:
        from models.verifier import verify_yaml_file

        p = write_yaml(tmp_path, _VALID_YAML)
        report = verify_yaml_file(p)
        # No FAIL checks — overall is PASS or WARN (refs present → PASS)
        assert report.overall.value in ("pass", "warn")
        names = [c.name for c in report.checks]
        assert "schema_valid" in names
        assert "equations_present" in names

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        from models.verifier import verify_yaml_file

        with pytest.raises(FileNotFoundError):
            verify_yaml_file(tmp_path / "nonexistent.yaml")

    def test_invalid_schema_fails(self, tmp_path: Path) -> None:
        from models.verifier import verify_yaml_file

        p = write_yaml(tmp_path, _YAML_INVALID_SCHEMA)
        report = verify_yaml_file(p)
        assert report.overall.value == "fail"
        schema_check = next(c for c in report.checks if c.name == "schema_valid")
        assert schema_check.status.value == "fail"

    def test_missing_equations_fails(self, tmp_path: Path) -> None:
        from models.verifier import verify_yaml_file

        p = write_yaml(tmp_path, _YAML_MISSING_EQUATIONS)
        report = verify_yaml_file(p)
        assert report.overall.value == "fail"
        eq_check = next(c for c in report.checks if c.name == "equations_present")
        assert eq_check.status.value == "fail"

    def test_missing_solver_fails(self, tmp_path: Path) -> None:
        from models.verifier import verify_yaml_file

        p = write_yaml(tmp_path, _YAML_MISSING_SOLVER)
        report = verify_yaml_file(p)
        assert report.overall.value == "fail"
        solver_check = next(c for c in report.checks if c.name == "solver_specified")
        assert solver_check.status.value == "fail"

    def test_missing_references_warns(self, tmp_path: Path) -> None:
        from models.verifier import verify_yaml_file

        p = write_yaml(tmp_path, _VALID_YAML_NO_REFS)
        report = verify_yaml_file(p)
        ref_check = next(c for c in report.checks if c.name == "references_present")
        assert ref_check.status.value == "warn"

    def test_registry_hash_is_sha256(self, tmp_path: Path) -> None:
        from models.verifier import verify_yaml_file

        p = write_yaml(tmp_path, _VALID_YAML)
        report = verify_yaml_file(p)
        expected = hashlib.sha256(p.read_bytes()).hexdigest()
        assert report.registry_hash == expected

    def test_model_id_from_stem(self, tmp_path: Path) -> None:
        from models.verifier import verify_yaml_file

        p = write_yaml(tmp_path, _VALID_YAML, filename="kalman_filter.yaml")
        report = verify_yaml_file(p)
        assert report.model_id == "kalman_filter"

    def test_seven_checks_present(self, tmp_path: Path) -> None:
        from models.verifier import verify_yaml_file

        p = write_yaml(tmp_path, _VALID_YAML)
        report = verify_yaml_file(p)
        assert len(report.checks) == 7

    def test_parameter_not_in_equations_warns(self, tmp_path: Path) -> None:
        from models.verifier import verify_yaml_file

        yaml_with_mismatch = textwrap.dedent("""\
            id: mismatch
            name: Mismatch
            category: test
            problem_type: test
            objective: test objective
            state_variables:
              - "s: state"
            observables:
              - "o: observation"
            solver: analytic
            equations:
              - "y = alpha * x"
            outputs:
              - "y: output"
            parameters:
              - "gamma: unrelated param"
            references:
              - "some ref"
        """)
        p = write_yaml(tmp_path, yaml_with_mismatch)
        report = verify_yaml_file(p)
        pie_check = next(c for c in report.checks if c.name == "parameter_in_equations")
        assert pie_check.status.value == "warn"
        assert "gamma" in pie_check.message


class TestVerifyAll:
    def test_verify_all_returns_dict(self, tmp_path: Path) -> None:
        from models.verifier import verify_all

        write_yaml(tmp_path, _VALID_YAML, "model_a.yaml")
        write_yaml(tmp_path, _VALID_YAML_NO_REFS, "model_b.yaml")
        results = verify_all(tmp_path)
        assert set(results.keys()) == {"model_a", "model_b"}

    def test_verify_all_missing_dir_raises(self, tmp_path: Path) -> None:
        from models.verifier import verify_all

        with pytest.raises(FileNotFoundError):
            verify_all(tmp_path / "nonexistent_dir")

    def test_verify_all_empty_dir_returns_empty(self, tmp_path: Path) -> None:
        from models.verifier import verify_all

        results = verify_all(tmp_path)
        assert results == {}


# ===========================================================================
# trace.py tests
# ===========================================================================


class TestMakeNodeId:
    def test_deterministic(self) -> None:
        from models.trace import make_node_id
        from schemas import TraceNodeType

        nid1 = make_node_id("abc123", TraceNodeType.REGISTRY, "kalman_filter")
        nid2 = make_node_id("abc123", TraceNodeType.REGISTRY, "kalman_filter")
        assert nid1 == nid2

    def test_length_16(self) -> None:
        from models.trace import make_node_id
        from schemas import TraceNodeType

        nid = make_node_id("abc123", TraceNodeType.VERIFICATION, "test_model")
        assert len(nid) == 16

    def test_different_types_differ(self) -> None:
        from models.trace import make_node_id
        from schemas import TraceNodeType

        nid_r = make_node_id("abc", TraceNodeType.REGISTRY, "m")
        nid_v = make_node_id("abc", TraceNodeType.VERIFICATION, "m")
        assert nid_r != nid_v

    def test_different_models_differ(self) -> None:
        from models.trace import make_node_id
        from schemas import TraceNodeType

        nid_a = make_node_id("abc", TraceNodeType.REGISTRY, "model_a")
        nid_b = make_node_id("abc", TraceNodeType.REGISTRY, "model_b")
        assert nid_a != nid_b


class TestTraceGraph:
    def _make_node(
        self,
        content_hash: str = "deadbeef",
        node_type_str: str = "registry",
        model_id: str = "test_model",
        parent_ids: list[str] | None = None,
    ):
        from models.trace import make_node_id
        from schemas import TraceNode, TraceNodeType

        ntype = TraceNodeType(node_type_str)
        nid = make_node_id(content_hash, ntype, model_id)
        return TraceNode(
            node_id=nid,
            node_type=ntype,
            model_id=model_id,
            artifact_path="some/path.yaml",
            content_hash=content_hash,
            parent_ids=parent_ids or [],
        )

    def test_append_and_load_all(self, tmp_path: Path) -> None:
        from models.trace import TraceGraph

        graph = TraceGraph(path=tmp_path / "trace.jsonl")
        node = self._make_node()
        graph.append(node)
        nodes = graph.load_all()
        assert len(nodes) == 1
        assert nodes[0].node_id == node.node_id

    def test_load_all_empty_if_no_file(self, tmp_path: Path) -> None:
        from models.trace import TraceGraph

        graph = TraceGraph(path=tmp_path / "trace.jsonl")
        assert graph.load_all() == []

    def test_append_multiple_ordered(self, tmp_path: Path) -> None:
        from models.trace import TraceGraph

        graph = TraceGraph(path=tmp_path / "trace.jsonl")
        n1 = self._make_node("hash1", "registry")
        n2 = self._make_node("hash2", "verification", parent_ids=[n1.node_id])
        graph.append(n1)
        graph.append(n2)
        nodes = graph.load_all()
        assert nodes[0].node_id == n1.node_id
        assert nodes[1].node_id == n2.node_id

    def test_load_model_filters(self, tmp_path: Path) -> None:
        from models.trace import TraceGraph

        graph = TraceGraph(path=tmp_path / "trace.jsonl")
        n_a = self._make_node("hash1", "registry", model_id="model_a")
        n_b = self._make_node("hash2", "registry", model_id="model_b")
        graph.append(n_a)
        graph.append(n_b)
        assert len(graph.load_model("model_a")) == 1
        assert graph.load_model("model_a")[0].model_id == "model_a"

    def test_get_node(self, tmp_path: Path) -> None:
        from models.trace import TraceGraph

        graph = TraceGraph(path=tmp_path / "trace.jsonl")
        node = self._make_node()
        graph.append(node)
        found = graph.get_node(node.node_id)
        assert found is not None
        assert found.node_id == node.node_id

    def test_get_node_missing_returns_none(self, tmp_path: Path) -> None:
        from models.trace import TraceGraph

        graph = TraceGraph(path=tmp_path / "trace.jsonl")
        assert graph.get_node("nonexistent") is None

    def test_ancestry_bfs(self, tmp_path: Path) -> None:
        from models.trace import TraceGraph

        graph = TraceGraph(path=tmp_path / "trace.jsonl")
        n1 = self._make_node("h1", "registry")
        n2 = self._make_node("h2", "verification", parent_ids=[n1.node_id])
        n3 = self._make_node("h3", "generated_code", parent_ids=[n2.node_id])
        for n in (n1, n2, n3):
            graph.append(n)
        ancestors = graph.ancestry(n3.node_id)
        ancestor_ids = {a.node_id for a in ancestors}
        assert n2.node_id in ancestor_ids
        assert n1.node_id in ancestor_ids
        assert n3.node_id not in ancestor_ids

    def test_ancestry_root_returns_empty(self, tmp_path: Path) -> None:
        from models.trace import TraceGraph

        graph = TraceGraph(path=tmp_path / "trace.jsonl")
        node = self._make_node()
        graph.append(node)
        assert graph.ancestry(node.node_id) == []

    def test_ancestry_missing_node_returns_empty(self, tmp_path: Path) -> None:
        from models.trace import TraceGraph

        graph = TraceGraph(path=tmp_path / "trace.jsonl")
        assert graph.ancestry("nonexistent") == []

    def test_descendants(self, tmp_path: Path) -> None:
        from models.trace import TraceGraph

        graph = TraceGraph(path=tmp_path / "trace.jsonl")
        n1 = self._make_node("h1", "registry")
        n2 = self._make_node("h2", "verification", parent_ids=[n1.node_id])
        n3 = self._make_node("h3", "generated_code", parent_ids=[n1.node_id])
        for n in (n1, n2, n3):
            graph.append(n)
        descendants = graph.descendants(n1.node_id)
        desc_ids = {d.node_id for d in descendants}
        assert n2.node_id in desc_ids
        assert n3.node_id in desc_ids
        assert n1.node_id not in desc_ids

    def test_to_dict_structure(self, tmp_path: Path) -> None:
        from models.trace import TraceGraph

        graph = TraceGraph(path=tmp_path / "trace.jsonl")
        n1 = self._make_node("h1", "registry")
        n2 = self._make_node("h2", "verification", parent_ids=[n1.node_id])
        for n in (n1, n2):
            graph.append(n)
        d = graph.to_dict()
        assert "nodes" in d
        assert "edges" in d
        assert len(d["nodes"]) == 2
        assert len(d["edges"]) == 1
        assert d["edges"][0] == {"from": n1.node_id, "to": n2.node_id}


# ===========================================================================
# forge.py tests
# ===========================================================================


class TestGenerateSkeleton:
    def _entry(self) -> object:
        from schemas import ModelRegistryEntry

        data = yaml.safe_load(_VALID_YAML)
        return ModelRegistryEntry(**data)

    def test_output_is_importable_python(self) -> None:
        from models.forge import generate_skeleton

        entry = self._entry()
        src = generate_skeleton(entry, "test_model")
        # Should not raise SyntaxError
        compile(src, "<string>", "exec")

    def test_class_name_camelcase(self) -> None:
        from models.forge import generate_skeleton

        entry = self._entry()
        src = generate_skeleton(entry, "test_model")
        assert "class TestModelParams" in src
        assert "class TestModel:" in src

    def test_equations_in_docstring(self) -> None:
        from models.forge import generate_skeleton

        entry = self._entry()
        src = generate_skeleton(entry, "test_model")
        assert "beta = (X^T X)^{-1} X^T y" in src

    def test_deterministic(self) -> None:
        from models.forge import generate_skeleton

        entry = self._entry()
        s1 = generate_skeleton(entry, "test_model")
        s2 = generate_skeleton(entry, "test_model")
        assert s1 == s2

    def test_no_parameters_generates_pass(self) -> None:
        from models.forge import generate_skeleton
        from schemas import ModelRegistryEntry

        data = yaml.safe_load(_VALID_YAML_NO_REFS)
        data["parameters"] = []
        entry = ModelRegistryEntry(**data)
        src = generate_skeleton(entry, "no_refs_model")
        assert "pass" in src

    def test_fit_and_predict_present(self) -> None:
        from models.forge import generate_skeleton

        entry = self._entry()
        src = generate_skeleton(entry, "test_model")
        assert "def fit(" in src
        assert "def predict(" in src
        assert "raise NotImplementedError" in src


class TestModelForge:
    def _make_registry_dir(self, tmp_path: Path) -> Path:
        reg_dir = tmp_path / "configs" / "model_registry"
        reg_dir.mkdir(parents=True)
        (reg_dir / "test_model.yaml").write_text(_VALID_YAML, encoding="utf-8")
        return reg_dir

    def test_run_creates_artifacts(self, tmp_path: Path) -> None:
        from models.forge import ModelForge

        reg_dir = self._make_registry_dir(tmp_path)
        artifacts_dir = tmp_path / "artifacts" / "modelforge"
        forge = ModelForge(
            project_dir=tmp_path,
            registry_dir=reg_dir,
            artifacts_dir=artifacts_dir,
        )
        report = forge.run("test_model")
        assert (artifacts_dir / "test_model" / "impl_skeleton.py").exists()
        assert (artifacts_dir / "test_model" / "verification.json").exists()
        assert (artifacts_dir / "test_model" / "spec_snapshot.yaml").exists()
        assert report.model_id == "test_model"

    def test_run_returns_forge_report(self, tmp_path: Path) -> None:
        from models.forge import ModelForge
        from schemas import ForgeReport

        reg_dir = self._make_registry_dir(tmp_path)
        artifacts_dir = tmp_path / "artifacts" / "modelforge"
        forge = ModelForge(
            project_dir=tmp_path,
            registry_dir=reg_dir,
            artifacts_dir=artifacts_dir,
        )
        report = forge.run("test_model")
        assert isinstance(report, ForgeReport)
        assert len(report.trace_node_ids) == 4  # registry + verify + skeleton + audit

    def test_run_missing_model_raises(self, tmp_path: Path) -> None:
        from models.forge import ModelForge

        reg_dir = self._make_registry_dir(tmp_path)
        artifacts_dir = tmp_path / "artifacts" / "modelforge"
        forge = ModelForge(
            project_dir=tmp_path,
            registry_dir=reg_dir,
            artifacts_dir=artifacts_dir,
        )
        with pytest.raises(FileNotFoundError):
            forge.run("nonexistent_model")

    def test_run_writes_trace_nodes(self, tmp_path: Path) -> None:
        from models.forge import ModelForge
        from models.trace import TraceGraph

        reg_dir = self._make_registry_dir(tmp_path)
        artifacts_dir = tmp_path / "artifacts" / "modelforge"
        forge = ModelForge(
            project_dir=tmp_path,
            registry_dir=reg_dir,
            artifacts_dir=artifacts_dir,
        )
        forge.run("test_model")
        graph = TraceGraph(path=artifacts_dir / "trace.jsonl")
        nodes = graph.load_model("test_model")
        types = {n.node_type.value for n in nodes}
        assert "registry" in types
        assert "verification" in types
        assert "generated_code" in types
        assert "audit_entry" in types

    def test_run_writes_audit_log(self, tmp_path: Path) -> None:
        from models.forge import ModelForge

        reg_dir = self._make_registry_dir(tmp_path)
        artifacts_dir = tmp_path / "artifacts" / "modelforge"
        forge = ModelForge(
            project_dir=tmp_path,
            registry_dir=reg_dir,
            artifacts_dir=artifacts_dir,
        )
        forge.run("test_model")
        audit_log = tmp_path / ".claude" / "audit" / "modelforge.jsonl"
        assert audit_log.exists()
        lines = [json.loads(ln) for ln in audit_log.read_text().splitlines() if ln.strip()]
        assert len(lines) == 1
        assert lines[0]["model_id"] == "test_model"
        assert lines[0]["event"] == "forge_run"

    def test_run_all_processes_all_models(self, tmp_path: Path) -> None:
        from models.forge import ModelForge

        reg_dir = tmp_path / "configs" / "model_registry"
        reg_dir.mkdir(parents=True)
        (reg_dir / "model_a.yaml").write_text(
            _VALID_YAML.replace("id: test_model", "id: model_a").replace(
                "name: Test Model", "name: Model A"
            ),
            encoding="utf-8",
        )
        (reg_dir / "model_b.yaml").write_text(
            _VALID_YAML.replace("id: test_model", "id: model_b").replace(
                "name: Test Model", "name: Model B"
            ),
            encoding="utf-8",
        )
        artifacts_dir = tmp_path / "artifacts" / "modelforge"
        forge = ModelForge(
            project_dir=tmp_path,
            registry_dir=reg_dir,
            artifacts_dir=artifacts_dir,
        )
        reports = forge.run_all()
        assert set(reports.keys()) == {"model_a", "model_b"}

    def test_run_idempotent_trace_appends(self, tmp_path: Path) -> None:
        """Running forge twice deduplicates deterministic nodes (idempotency).

        Nodes per run:
          - REGISTRY      : deterministic (YAML hash) → deduplicated on 2nd run
          - VERIFICATION  : includes created_at timestamp → always new
          - GENERATED_CODE: deterministic (skeleton source) → deduplicated on 2nd run
          - AUDIT_ENTRY   : includes timestamp → always new

        Expected: 4 (run 1) + 2 (run 2: verification + audit) = 6
        """
        from models.forge import ModelForge
        from models.trace import TraceGraph

        reg_dir = self._make_registry_dir(tmp_path)
        artifacts_dir = tmp_path / "artifacts" / "modelforge"
        forge = ModelForge(
            project_dir=tmp_path,
            registry_dir=reg_dir,
            artifacts_dir=artifacts_dir,
        )
        forge.run("test_model")
        forge.run("test_model")
        graph = TraceGraph(path=artifacts_dir / "trace.jsonl")
        nodes = graph.load_model("test_model")
        # 4 (run 1) + 2 (run 2: verification + audit, non-deterministic due to timestamps) = 6
        assert len(nodes) == 6
        # All node_ids must be unique (no true duplicates in log)
        node_ids = [n.node_id for n in nodes]
        assert len(node_ids) == len(set(node_ids))

    def test_verification_report_embedded(self, tmp_path: Path) -> None:
        from models.forge import ModelForge

        reg_dir = self._make_registry_dir(tmp_path)
        artifacts_dir = tmp_path / "artifacts" / "modelforge"
        forge = ModelForge(
            project_dir=tmp_path,
            registry_dir=reg_dir,
            artifacts_dir=artifacts_dir,
        )
        report = forge.run("test_model")
        assert report.verification.model_id == "test_model"
        assert report.verification.overall.value in ("pass", "warn", "fail")


# ===========================================================================
# automation.py tests
# ===========================================================================


class TestDispatch:
    def _make_registry(self, tmp_path: Path) -> Path:
        reg_dir = tmp_path / "configs" / "model_registry"
        reg_dir.mkdir(parents=True)
        (reg_dir / "test_model.yaml").write_text(_VALID_YAML, encoding="utf-8")
        return reg_dir

    def test_dispatch_unknown_event_raises(self) -> None:
        from models.automation import dispatch

        with pytest.raises(ValueError, match="Unknown event type"):
            asyncio.run(dispatch("nonexistent_event", {}))

    def test_handle_verify_requested_single(self, tmp_path: Path, monkeypatch) -> None:
        from models import automation

        reg_dir = self._make_registry(tmp_path)
        monkeypatch.setattr(automation, "_REGISTRY_DIR", reg_dir)
        result = asyncio.run(automation.handle_verify_requested({"model_id": "test_model"}))
        assert result["model_id"] == "test_model"
        assert "overall" in result
        assert "checks" in result

    def test_handle_verify_requested_all(self, tmp_path: Path, monkeypatch) -> None:
        from models import automation

        reg_dir = self._make_registry(tmp_path)
        monkeypatch.setattr(automation, "_REGISTRY_DIR", reg_dir)
        result = asyncio.run(automation.handle_verify_requested({"model_id": "all"}))
        assert "results" in result
        assert "test_model" in result["results"]

    def test_handle_verify_requested_missing(self, tmp_path: Path, monkeypatch) -> None:
        from models import automation

        monkeypatch.setattr(automation, "_REGISTRY_DIR", tmp_path / "configs" / "model_registry")
        result = asyncio.run(automation.handle_verify_requested({"model_id": "ghost"}))
        assert result.get("overall") == "error" or "error" in result

    def test_handle_registry_changed(self, tmp_path: Path, monkeypatch) -> None:
        from models import automation
        from models.forge import ModelForge

        reg_dir = self._make_registry(tmp_path)
        monkeypatch.setattr(automation, "_REGISTRY_DIR", reg_dir)

        # Patch ModelForge to use tmp dirs
        _tmp = tmp_path
        _reg = reg_dir
        original_init = ModelForge.__init__

        def patched_init(self, project_dir=None, registry_dir=None, artifacts_dir=None):
            original_init(
                self,
                project_dir=_tmp,
                registry_dir=_reg,
                artifacts_dir=_tmp / "artifacts" / "modelforge",
            )

        monkeypatch.setattr(ModelForge, "__init__", patched_init)

        result = asyncio.run(automation.handle_registry_changed({"model_id": "test_model"}))
        assert result["model_id"] == "test_model"
        assert "verification_overall" in result
        assert result.get("forge_status") == "ok"

    def test_dispatch_verify_requested(self, tmp_path: Path, monkeypatch) -> None:
        from models import automation

        reg_dir = self._make_registry(tmp_path)
        monkeypatch.setattr(automation, "_REGISTRY_DIR", reg_dir)
        result = asyncio.run(automation.dispatch("verify_requested", {"model_id": "test_model"}))
        assert "overall" in result


# ===========================================================================
# FastAPI endpoint tests
# ===========================================================================


@pytest.fixture()
def client(tmp_path: Path, monkeypatch):
    """TestClient with patched ModelForge and registry paths."""
    import sys

    _src = str(Path(__file__).parent.parent / "src")
    if _src not in sys.path:
        sys.path.insert(0, _src)

    from fastapi.testclient import TestClient

    from api.app import app
    from api.routers import forge as forge_router_module
    from models.forge import ModelForge

    # Prepare registry + artifacts in tmp
    reg_dir = tmp_path / "configs" / "model_registry"
    reg_dir.mkdir(parents=True)
    (reg_dir / "test_model.yaml").write_text(_VALID_YAML, encoding="utf-8")
    artifacts_dir = tmp_path / "artifacts" / "modelforge"

    # Patch router-level constants
    monkeypatch.setattr(forge_router_module, "_REGISTRY_DIR", reg_dir)
    audit_log = tmp_path / ".claude" / "audit" / "modelforge.jsonl"
    audit_log.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(forge_router_module, "_AUDIT_LOG", audit_log)

    # Patch ModelForge constructor to use tmp dirs
    _tmp = tmp_path
    _reg = reg_dir
    _art = artifacts_dir
    original_init = ModelForge.__init__

    def patched_init(self, project_dir=None, registry_dir=None, artifacts_dir=None):
        original_init(self, project_dir=_tmp, registry_dir=_reg, artifacts_dir=_art)

    monkeypatch.setattr(ModelForge, "__init__", patched_init)

    # Patch TraceGraph default path so /forge/trace reads from the same tmp artifacts dir
    from models import trace as trace_module

    _trace_path = artifacts_dir / "trace.jsonl"
    original_trace_init = trace_module.TraceGraph.__init__

    def patched_trace_init(self, path=None):
        original_trace_init(self, path=path if path is not None else _trace_path)

    monkeypatch.setattr(trace_module.TraceGraph, "__init__", patched_trace_init)

    return TestClient(app)


class TestForgeApiEndpoints:
    def test_forge_run_returns_200(self, client) -> None:
        resp = client.post("/forge/run/test_model")
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_id"] == "test_model"
        assert "verification" in data
        assert "skeleton_code_path" in data

    def test_forge_run_missing_model_returns_404(self, client) -> None:
        resp = client.post("/forge/run/ghost_model")
        assert resp.status_code == 404

    def test_forge_verify_returns_200(self, client) -> None:
        resp = client.post("/forge/verify/test_model")
        assert resp.status_code == 200
        data = resp.json()
        assert "overall" in data
        assert "checks" in data

    def test_forge_verify_missing_returns_404(self, client) -> None:
        resp = client.post("/forge/verify/ghost_model")
        assert resp.status_code == 404

    def test_forge_trace_returns_list(self, client) -> None:
        # Run first to populate trace
        client.post("/forge/run/test_model")
        resp = client.get("/forge/trace/test_model")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) >= 4

    def test_forge_graph_returns_nodes_edges(self, client) -> None:
        client.post("/forge/run/test_model")
        resp = client.get("/forge/graph")
        assert resp.status_code == 200
        data = resp.json()
        assert "nodes" in data
        assert "edges" in data

    def test_forge_audit_empty_when_no_log(self, client, monkeypatch, tmp_path) -> None:
        from api.routers import forge as forge_router_module

        empty_log = tmp_path / "empty_audit.jsonl"
        monkeypatch.setattr(forge_router_module, "_AUDIT_LOG", empty_log)
        resp = client.get("/forge/audit")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_forge_audit_returns_entries_after_run(self, client) -> None:
        client.post("/forge/run/test_model")
        resp = client.get("/forge/audit")
        assert resp.status_code == 200
        entries = resp.json()
        assert len(entries) >= 1
        assert entries[0]["model_id"] == "test_model"

    def test_forge_run_all_returns_200(self, client) -> None:
        resp = client.post("/forge/run/all")
        assert resp.status_code == 200

    def test_forge_verify_all_returns_200(self, client) -> None:
        resp = client.post("/forge/verify/all")
        assert resp.status_code == 200


# ===========================================================================
# Real registry integration test (uses actual kalman_filter.yaml)
# ===========================================================================


class TestRealRegistryIntegration:
    def test_kalman_filter_yaml_passes_all_fail_checks(self, tmp_path: Path) -> None:
        """kalman_filter.yaml should have no FAIL checks."""
        if not _KALMAN_YAML_PATH.exists():
            pytest.skip("kalman_filter.yaml not found")
        from models.verifier import verify_yaml_file

        report = verify_yaml_file(_KALMAN_YAML_PATH)
        failed = [c for c in report.checks if c.status.value == "fail"]
        assert failed == [], f"Unexpected FAIL checks: {[c.name for c in failed]}"

    def test_kalman_filter_forge_full_pipeline(self, tmp_path: Path) -> None:
        """Full forge pipeline on the real kalman_filter.yaml."""
        if not _KALMAN_YAML_PATH.exists():
            pytest.skip("kalman_filter.yaml not found")

        import shutil

        from models.forge import ModelForge
        from models.trace import TraceGraph

        # Copy kalman YAML into tmp_path to keep project_dir and artifacts_dir consistent
        reg_dir = tmp_path / "configs" / "model_registry"
        reg_dir.mkdir(parents=True)
        shutil.copy(_KALMAN_YAML_PATH, reg_dir / "kalman_filter.yaml")

        artifacts_dir = tmp_path / "artifacts" / "modelforge"
        forge = ModelForge(
            project_dir=tmp_path,
            registry_dir=reg_dir,
            artifacts_dir=artifacts_dir,
        )
        report = forge.run("kalman_filter")
        assert report.model_id == "kalman_filter"
        assert (artifacts_dir / "kalman_filter" / "impl_skeleton.py").exists()
        graph = TraceGraph(path=artifacts_dir / "trace.jsonl")
        nodes = graph.load_model("kalman_filter")
        assert len(nodes) == 4
        compile(
            (artifacts_dir / "kalman_filter" / "impl_skeleton.py").read_text(encoding="utf-8"),
            "<skeleton>",
            "exec",
        )
