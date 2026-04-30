"""CLI entrypoint for ModelForge: python -m src.models <command> [args].

Commands:
    run <model_id|all>     — full forge pipeline
    verify <model_id|all>  — static verification only
    trace <model_id>       — show traceability chain
    audit                  — tail the audit log
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _cmd_run(model_id: str) -> None:
    from models.forge import ModelForge

    forge = ModelForge()
    if model_id == "all":
        reports = forge.run_all()
        for mid, r in reports.items():
            status = r.verification.overall.value.upper()
            print(f"[{status}] {mid} → {r.skeleton_code_path}")
    else:
        r = forge.run(model_id)
        status = r.verification.overall.value.upper()
        print(f"[{status}] {model_id}")
        print(f"  Skeleton: {r.skeleton_code_path}")
        print(f"  Trace nodes: {', '.join(r.trace_node_ids)}")


def _cmd_verify(model_id: str) -> None:
    from models.verifier import verify_yaml_file, verify_all

    registry_dir = Path("configs") / "model_registry"
    if model_id == "all":
        reports = verify_all(registry_dir)
        for mid, r in sorted(reports.items()):
            status = r.overall.value.upper()
            fails = [c.name for c in r.checks if c.status.value == "fail"]
            warns = [c.name for c in r.checks if c.status.value == "warn"]
            suffix = ""
            if fails:
                suffix += f"  FAIL: {', '.join(fails)}"
            if warns:
                suffix += f"  WARN: {', '.join(warns)}"
            print(f"[{status}] {mid}{suffix}")
    else:
        yaml_path = registry_dir / f"{model_id}.yaml"
        r = verify_yaml_file(yaml_path)
        print(f"[{r.overall.value.upper()}] {model_id}  hash={r.registry_hash[:12]}…")
        for c in r.checks:
            icon = {"pass": "✓", "warn": "⚠", "fail": "✗"}.get(c.status.value, "?")
            msg = f"  {icon} {c.name}"
            if c.message:
                msg += f": {c.message}"
            print(msg)


def _cmd_trace(model_id: str) -> None:
    from models.trace import TraceGraph

    graph = TraceGraph()
    nodes = graph.load_model(model_id)
    if not nodes:
        print(f"No trace nodes found for model_id={model_id!r}")
        return
    print(f"Trace for {model_id} ({len(nodes)} nodes):")
    for n in nodes:
        parents = ", ".join(n.parent_ids) if n.parent_ids else "—"
        print(f"  [{n.node_type.value:18s}] {n.node_id}  ← {parents}")
        print(f"    path={n.artifact_path}")
        print(f"    hash={n.content_hash[:16]}…  at={n.created_at.isoformat()}")


def _cmd_audit(tail: int = 20) -> None:
    audit_path = Path(".claude") / "audit" / "modelforge.jsonl"
    if not audit_path.exists():
        print("No ModelForge audit log found.")
        return
    lines = audit_path.read_text(encoding="utf-8").strip().splitlines()
    recent = lines[-tail:]
    print(f"ModelForge audit log (last {len(recent)} entries):")
    for line in recent:
        record = json.loads(line)
        print(
            f"  {record['timestamp']}  [{record['event']:20s}] {record['model_id']}"
            + (f"  {record.get('verification_overall', '')}" if "verification_overall" in record else "")
        )


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(0)

    cmd = args[0]

    if cmd == "run":
        model_id = args[1] if len(args) > 1 else "all"
        _cmd_run(model_id)
    elif cmd == "verify":
        model_id = args[1] if len(args) > 1 else "all"
        _cmd_verify(model_id)
    elif cmd == "trace":
        if len(args) < 2:
            print("Usage: python -m src.models trace <model_id>", file=sys.stderr)
            sys.exit(1)
        _cmd_trace(args[1])
    elif cmd == "audit":
        _cmd_audit()
    else:
        print(f"Unknown command: {cmd!r}", file=sys.stderr)
        print(__doc__, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
