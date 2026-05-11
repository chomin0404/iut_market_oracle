"""Cloud Functions-style event handlers for ModelForge automation.

Each handler is a coroutine matching the Cloud Functions interface:

    async def handle_<event>(event: dict) -> dict

Supported event types:
    registry_changed   — fired when a model registry YAML is edited
    forge_requested    — fired when /modelforge-run is invoked
    verify_requested   — fired when /modelforge-verify is invoked
    report_requested   — fired when /modelforge-report is invoked

Invocation:
    From hooks:
        python -m src.models.automation registry_changed '{"model_id": "kalman_filter"}'

    From Python:
        import asyncio
        from models.automation import dispatch
        result = asyncio.run(dispatch("registry_changed", {"model_id": "kalman_filter"}))

Cloud Functions deployment note:
    Each handler is independently deployable as a Cloud Function.
    Replace the asyncio.run() wrapper with the Functions Framework HTTP trigger.
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml

from models.forge import ModelForge
from models.verifier import verify_all, verify_yaml_file

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_REGISTRY_DIR = Path("configs") / "model_registry"
_ARTIFACTS_DIR = Path("artifacts") / "modelforge"
_REPORTS_DIR = Path("reports") / "modelforge"

# ---------------------------------------------------------------------------
# Event handlers
# ---------------------------------------------------------------------------


async def handle_registry_changed(event: dict) -> dict:
    """Auto-verify + forge when a registry YAML changes.

    Expected event keys:
        model_id (str, required): Registry entry ID.
        yaml_path (str, optional): Explicit YAML path override.

    Returns:
        {"model_id": ..., "verification_overall": ..., "forge_status": "ok"|"error", ...}
    """
    model_id: str = event["model_id"]
    yaml_path_str: str | None = event.get("yaml_path")

    if yaml_path_str:
        yaml_path = Path(yaml_path_str)
    else:
        yaml_path = _REGISTRY_DIR / f"{model_id}.yaml"

    result: dict = {"model_id": model_id, "trigger": "registry_changed"}

    # Verify first (fast, no side effects)
    try:
        report = verify_yaml_file(yaml_path)
        result["verification_overall"] = report.overall.value
        result["failed_checks"] = [c.name for c in report.checks if c.status.value == "fail"]
    except Exception as exc:  # noqa: BLE001
        result["verification_overall"] = "error"
        result["error"] = str(exc)
        return result

    # Full forge (writes artifacts + trace + audit)
    try:
        forge = ModelForge()
        forge_report = forge.run(model_id)
        result["forge_status"] = "ok"
        result["skeleton_path"] = forge_report.skeleton_code_path
        result["trace_node_ids"] = forge_report.trace_node_ids
    except Exception as exc:  # noqa: BLE001
        result["forge_status"] = "error"
        result["forge_error"] = str(exc)

    return result


async def handle_forge_requested(event: dict) -> dict:
    """Run the full ModelForge pipeline on demand.

    Expected event keys:
        model_id (str): Registry entry ID; "all" runs the entire registry.

    Returns:
        {"results": {model_id: {"verification_overall": ..., "forge_status": ...}}}
    """
    model_id: str = event.get("model_id", "all")
    forge = ModelForge()

    if model_id == "all":
        reports = forge.run_all()
        return {
            "results": {
                mid: {
                    "verification_overall": r.verification.overall.value,
                    "forge_status": "ok",
                    "skeleton_path": r.skeleton_code_path,
                }
                for mid, r in reports.items()
            }
        }

    try:
        report = forge.run(model_id)
        return {
            "model_id": model_id,
            "verification_overall": report.verification.overall.value,
            "forge_status": "ok",
            "skeleton_path": report.skeleton_code_path,
            "trace_node_ids": report.trace_node_ids,
        }
    except Exception as exc:  # noqa: BLE001
        return {"model_id": model_id, "forge_status": "error", "error": str(exc)}


async def handle_verify_requested(event: dict) -> dict:
    """Verify one or all registry entries without generating artifacts.

    Expected event keys:
        model_id (str): Registry entry ID; "all" verifies everything.

    Returns:
        {"results": {model_id: {"overall": ..., "checks": [...]}}}
    """
    model_id: str = event.get("model_id", "all")

    if model_id == "all":
        reports = verify_all(_REGISTRY_DIR)
        return {
            "results": {
                mid: {
                    "overall": r.overall.value,
                    "checks": [
                        {"name": c.name, "status": c.status.value, "message": c.message}
                        for c in r.checks
                    ],
                }
                for mid, r in reports.items()
            }
        }

    yaml_path = _REGISTRY_DIR / f"{model_id}.yaml"
    try:
        report = verify_yaml_file(yaml_path)
        return {
            "model_id": model_id,
            "overall": report.overall.value,
            "checks": [
                {"name": c.name, "status": c.status.value, "message": c.message}
                for c in report.checks
            ],
        }
    except Exception as exc:  # noqa: BLE001
        return {"model_id": model_id, "overall": "error", "error": str(exc)}


async def handle_report_requested(event: dict) -> dict:
    """Generate a markdown diagnostic report from forge artifacts.

    Expected event keys:
        model_id (str): Registry entry ID; "all" generates for every forged model.

    For each model the report covers:
        - YAML spec summary (name, problem_type, solver, equations)
        - Verification results (per-check status)
        - Skeleton code path and SHA-256 prefix
        - Traceability chain (REGISTRY → VERIFICATION → GENERATED_CODE → AUDIT_ENTRY)

    Report written to: reports/modelforge/<model_id>.md
    """
    model_id: str = event.get("model_id", "all")
    project_dir = Path(event.get("project_dir", Path.cwd())).resolve()
    artifacts_dir = project_dir / _ARTIFACTS_DIR
    reports_dir = project_dir / _REPORTS_DIR
    reports_dir.mkdir(parents=True, exist_ok=True)

    def _report_for(mid: str) -> dict:
        model_artifacts = artifacts_dir / mid
        if not model_artifacts.exists():
            return {
                "model_id": mid,
                "status": "error",
                "error": "No forge artifacts found. Run /modelforge-run first.",
            }

        lines: list[str] = [
            f"# ModelForge Report — `{mid}`",
            "",
            f"*Generated: {datetime.now(UTC).isoformat()}*",
            "",
        ]

        # Spec snapshot
        spec_path = model_artifacts / "spec_snapshot.yaml"
        if spec_path.exists():
            try:
                spec = yaml.safe_load(spec_path.read_bytes())
                lines += [
                    "## Math Spec",
                    "",
                    f"- **Name**: {spec.get('name', mid)}",
                    f"- **Problem type**: {spec.get('problem_type', '—')}",
                    f"- **Solver**: {spec.get('solver', '—')}",
                    "",
                    "### Equations",
                    "",
                ]
                for eq in spec.get("equations", []):
                    lines.append(f"    {eq}")
                lines.append("")
                if spec.get("assumptions"):
                    lines.append("### Assumptions")
                    lines.append("")
                    for a in spec["assumptions"]:
                        lines.append(f"- {a}")
                    lines.append("")
            except Exception as exc:  # noqa: BLE001
                lines.append(f"*(spec snapshot unreadable: {exc})*\n")

        # Verification results
        verification_path = model_artifacts / "verification.json"
        if verification_path.exists():
            try:
                v = json.loads(verification_path.read_text(encoding="utf-8"))
                overall = v.get("overall", "?").upper()
                lines += [
                    "## Verification",
                    "",
                    f"**Overall**: `{overall}`",
                    "",
                    "| Check | Status | Message |",
                    "|---|---|---|",
                ]
                for c in v.get("checks", []):
                    msg = c.get("message") or ""
                    lines.append(f"| `{c['name']}` | `{c['status'].upper()}` | {msg} |")
                lines.append("")
            except Exception as exc:  # noqa: BLE001
                lines.append(f"*(verification.json unreadable: {exc})*\n")

        # Skeleton code path
        skeleton_path = model_artifacts / "impl_skeleton.py"
        if skeleton_path.exists():
            rel = skeleton_path.relative_to(project_dir)
            lines += [
                "## Generated Code",
                "",
                f"Skeleton: `{rel}`",
                "",
            ]

        # Traceability
        trace_path = artifacts_dir / "trace.jsonl"
        if trace_path.exists():
            from models.trace import TraceGraph  # local import to avoid circular at module level

            graph = TraceGraph(path=trace_path)
            nodes = graph.load_model(mid)
            if nodes:
                lines += [
                    "## Traceability Chain",
                    "",
                    "| Type | Node ID | Parent IDs | Artifact |",
                    "|---|---|---|---|",
                ]
                for n in nodes:
                    parents = ", ".join(n.parent_ids) if n.parent_ids else "—"
                    row = (
                        f"| `{n.node_type.value}` | `{n.node_id}` "
                        f"| `{parents}` | `{n.artifact_path}` |"
                    )
                    lines.append(row)
                lines.append("")

        report_text = "\n".join(lines)
        out_path = reports_dir / f"{mid}.md"
        out_path.write_text(report_text, encoding="utf-8")
        return {
            "model_id": mid,
            "status": "ok",
            "report_path": str(out_path.relative_to(project_dir)),
        }

    if model_id == "all":
        results: dict[str, dict] = {}
        for yaml_path in sorted((project_dir / _REGISTRY_DIR).glob("*.yaml")):
            mid = yaml_path.stem
            results[mid] = _report_for(mid)
        return {"results": results}

    return _report_for(model_id)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_HANDLERS: dict[str, object] = {
    "registry_changed": handle_registry_changed,
    "forge_requested": handle_forge_requested,
    "verify_requested": handle_verify_requested,
    "report_requested": handle_report_requested,
}


async def dispatch(event_type: str, event: dict) -> dict:
    """Route an event to the appropriate handler.

    Args:
        event_type: One of registry_changed | forge_requested | verify_requested.
        event:      Event payload dict.

    Returns:
        Handler result dict.

    Raises:
        ValueError: Unknown event_type.
    """
    handler = _HANDLERS.get(event_type)
    if handler is None:
        raise ValueError(f"Unknown event type: {event_type!r}. Valid types: {list(_HANDLERS)}")
    return await handler(event)  # type: ignore[operator]


# ---------------------------------------------------------------------------
# CLI entrypoint (called from hooks)
# ---------------------------------------------------------------------------


def _cli_main() -> None:
    """CLI: python -m src.models.automation <event_type> <json_payload>"""
    if len(sys.argv) < 2:
        print(
            "Usage: python -m src.models.automation <event_type> [json_payload]",
            file=sys.stderr,
        )
        sys.exit(1)

    event_type = sys.argv[1]
    payload_str = sys.argv[2] if len(sys.argv) > 2 else "{}"
    try:
        event = json.loads(payload_str)
    except json.JSONDecodeError as exc:
        print(f"Invalid JSON payload: {exc}", file=sys.stderr)
        sys.exit(1)

    result = asyncio.run(dispatch(event_type, event))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _cli_main()
