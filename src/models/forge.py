"""ModelForge — unified pipeline orchestrator (ModelForge).

Pipeline for one model:
    1. Load YAML  (configs/model_registry/<id>.yaml)
    2. Hash YAML  → registry TraceNode
    3. Verify     → VerificationReport + verification TraceNode
    4. Generate   → impl_skeleton.py + generated_code TraceNode
    5. Write artifacts under artifacts/modelforge/<model_id>/
    6. Append all TraceNodes to artifacts/modelforge/trace.jsonl
    7. Append audit entry to .claude/audit/modelforge.jsonl

Usage::

    forge = ModelForge()
    report = forge.run("kalman_filter")
    print(report.verification.overall)
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import yaml

from models.trace import TraceGraph, make_node_id
from models.verifier import verify_yaml_file
from schemas import (
    ForgeReport,
    ModelRegistryEntry,
    TraceNode,
    TraceNodeType,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_REGISTRY_DIR = Path("configs") / "model_registry"
_ARTIFACTS_DIR = Path("artifacts") / "modelforge"
_AUDIT_LOG = Path(".claude") / "audit" / "modelforge.jsonl"

_SKELETON_INDENT = "    "


# ---------------------------------------------------------------------------
# Skeleton code generator (deterministic, no LLM)
# ---------------------------------------------------------------------------


def _to_class_name(model_id: str) -> str:
    """kalman_filter → KalmanFilter"""
    return "".join(word.capitalize() for word in model_id.split("_"))


def _to_method_name(param: str) -> str:
    """Extract short param name, sanitise for Python identifier."""
    short = param.split(":")[0].split(" ")[0].strip()
    short = re.sub(r"[^a-zA-Z0-9_]", "_", short)
    return short.strip("_") or "param"


def generate_skeleton(entry: ModelRegistryEntry, model_id: str) -> str:
    """Generate a deterministic Python stub from a registry entry.

    The stub is immediately importable and documents the math spec inline.
    It does NOT contain algorithm logic (that is authored by humans / LLM separately).

    Args:
        entry:    Validated ModelRegistryEntry.
        model_id: Registry ID string (e.g. "kalman_filter").

    Returns:
        Python source code as a string.
    """
    cls = _to_class_name(model_id)
    params_block = "\n".join(
        f"    {_to_method_name(p)}: float  # {p}" for p in (entry.parameters or [])
    )

    lines: list[str] = [
        f'"""Auto-generated skeleton for {entry.name} ({model_id}).',
        "",
        f"Problem type: {entry.problem_type}",
        f"Solver:       {entry.solver}",
        "",
        "Equations:",
        *[f"    {eq}" for eq in entry.equations],
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "from dataclasses import dataclass",
        "",
        "",
        "@dataclass",
        f"class {cls}Params:",
        f'    """{entry.name} parameters."""',
        "",
    ]

    if entry.parameters:
        lines.append(params_block)
    else:
        lines.append("    pass  # No parameters declared in registry")

    lines += [
        "",
        "",
        f"class {cls}:",
        f'    """{entry.name}.',
        "",
        f"    Problem: {entry.problem_type}",
        f"    Solver:  {entry.solver}",
        "",
        "    Equations:",
    ]
    for eq in entry.equations:
        lines.append(f"        {eq}")

    if entry.assumptions:
        lines.append("")
        lines.append("    Assumptions:")
        for a in entry.assumptions:
            lines.append(f"        - {a}")

    lines += [
        '    """',
        "",
        f"    def __init__(self, params: {cls}Params) -> None:",
        "        self._params = params",
        "",
        "    def fit(self, *args, **kwargs):  # type: ignore[override]",
        '        """Fit / calibrate the model. Fill in implementation."""',
        "        raise NotImplementedError",
        "",
        "    def predict(self, *args, **kwargs):  # type: ignore[override]",
        '        """Run the model forward. Fill in implementation."""',
        "        raise NotImplementedError",
    ]

    if entry.outputs:
        lines += [
            "",
            "    # Expected outputs:",
        ]
        for o in entry.outputs:
            lines.append(f"    #   {o}")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------


def _write_audit_entry(
    model_id: str,
    event: str,
    detail: dict,
    project_dir: Path,
) -> str:
    """Append one JSONL line to .claude/audit/modelforge.jsonl.

    Returns the JSON string that was written.
    """
    audit_path = project_dir / _AUDIT_LOG
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.now(UTC).isoformat(),
        "model_id": model_id,
        "event": event,
        **detail,
    }
    line = json.dumps(record, ensure_ascii=False)
    with audit_path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    return line


# ---------------------------------------------------------------------------
# ModelForge
# ---------------------------------------------------------------------------


class ModelForge:
    """Orchestrator: YAML → verify → skeleton → trace → audit.

    Args:
        project_dir:  Project root (defaults to cwd). Paths are resolved relative to this.
        registry_dir: Override model registry directory.
        artifacts_dir: Override artifacts output directory.
    """

    def __init__(
        self,
        project_dir: Path | None = None,
        registry_dir: Path | None = None,
        artifacts_dir: Path | None = None,
    ) -> None:
        self._root = (project_dir or Path.cwd()).resolve()
        self._registry_dir = (registry_dir or (self._root / _REGISTRY_DIR)).resolve()
        self._artifacts_dir = (artifacts_dir or (self._root / _ARTIFACTS_DIR)).resolve()
        self._graph = TraceGraph(path=self._artifacts_dir / "trace.jsonl")

    # ── Public ────────────────────────────────────────────────────────────

    def run(self, model_id: str) -> ForgeReport:
        """Execute full ModelForge pipeline for one model.

        Args:
            model_id: Registry YAML stem (e.g. "kalman_filter").

        Returns:
            ForgeReport linking all produced artifacts and trace nodes.

        Raises:
            FileNotFoundError: YAML not found in registry directory.
        """
        yaml_path = self._registry_dir / f"{model_id}.yaml"
        if not yaml_path.exists():
            raise FileNotFoundError(f"Registry entry not found: {yaml_path}")

        model_dir = self._artifacts_dir / model_id
        model_dir.mkdir(parents=True, exist_ok=True)

        # 1. Hash YAML → registry node
        raw_bytes = yaml_path.read_bytes()
        registry_hash = hashlib.sha256(raw_bytes).hexdigest()
        registry_node_id = make_node_id(registry_hash, TraceNodeType.REGISTRY, model_id)

        registry_node = TraceNode(
            node_id=registry_node_id,
            node_type=TraceNodeType.REGISTRY,
            model_id=model_id,
            artifact_path=str(yaml_path.relative_to(self._root)),
            content_hash=registry_hash,
            parent_ids=[],
        )

        # 2. Write spec snapshot
        snapshot_path = model_dir / "spec_snapshot.yaml"
        snapshot_path.write_bytes(raw_bytes)

        # 3. Verify
        verification = verify_yaml_file(yaml_path)
        verification_json = verification.model_dump_json(indent=2)
        verification_path = model_dir / "verification.json"
        verification_path.write_text(verification_json, encoding="utf-8")

        v_hash = hashlib.sha256(verification_json.encode()).hexdigest()
        v_node_id = make_node_id(v_hash, TraceNodeType.VERIFICATION, model_id)
        v_node = TraceNode(
            node_id=v_node_id,
            node_type=TraceNodeType.VERIFICATION,
            model_id=model_id,
            artifact_path=str(verification_path.relative_to(self._root)),
            content_hash=v_hash,
            parent_ids=[registry_node_id],
        )

        # 4. Generate skeleton code
        data = yaml.safe_load(raw_bytes)
        entry = ModelRegistryEntry(**data)
        skeleton_src = generate_skeleton(entry, model_id)
        skeleton_path = model_dir / "impl_skeleton.py"
        skeleton_path.write_text(skeleton_src, encoding="utf-8")

        sk_hash = hashlib.sha256(skeleton_src.encode()).hexdigest()
        sk_node_id = make_node_id(sk_hash, TraceNodeType.GENERATED_CODE, model_id)
        sk_node = TraceNode(
            node_id=sk_node_id,
            node_type=TraceNodeType.GENERATED_CODE,
            model_id=model_id,
            artifact_path=str(skeleton_path.relative_to(self._root)),
            content_hash=sk_hash,
            parent_ids=[registry_node_id, v_node_id],
        )

        # 5. Append trace nodes
        for node in (registry_node, v_node, sk_node):
            self._graph.append(node)

        # 6. Audit
        audit_detail = {
            "registry_hash": registry_hash,
            "verification_overall": verification.overall.value,
            "skeleton_path": str(skeleton_path.relative_to(self._root)),
            "trace_node_ids": [registry_node_id, v_node_id, sk_node_id],
        }
        audit_line = _write_audit_entry(model_id, "forge_run", audit_detail, self._root)
        audit_hash = hashlib.sha256(audit_line.encode()).hexdigest()
        audit_node_id = make_node_id(audit_hash, TraceNodeType.AUDIT_ENTRY, model_id)
        audit_node = TraceNode(
            node_id=audit_node_id,
            node_type=TraceNodeType.AUDIT_ENTRY,
            model_id=model_id,
            artifact_path=str(_AUDIT_LOG),
            content_hash=audit_hash,
            parent_ids=[registry_node_id, v_node_id, sk_node_id],
        )
        self._graph.append(audit_node)

        return ForgeReport(
            model_id=model_id,
            registry_yaml_path=str(yaml_path.relative_to(self._root)),
            verification=verification,
            skeleton_code_path=str(skeleton_path.relative_to(self._root)),
            trace_node_ids=[registry_node_id, v_node_id, sk_node_id, audit_node_id],
        )

    def run_all(self) -> dict[str, ForgeReport]:
        """Run the full pipeline for every model in the registry.

        Returns:
            dict mapping model_id → ForgeReport.
        """
        reports: dict[str, ForgeReport] = {}
        for yaml_path in sorted(self._registry_dir.glob("*.yaml")):
            model_id = yaml_path.stem
            reports[model_id] = self.run(model_id)
        return reports
