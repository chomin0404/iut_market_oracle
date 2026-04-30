#!/usr/bin/env bash
# on-registry-change.sh — Cloud Functions-style trigger for model registry edits.
#
# Fires after any Edit/Write to configs/model_registry/*.yaml.
# Dispatches the "registry_changed" event to src.models.automation.
#
# Payload from Claude (stdin): {"tool_input": {"file_path": "..."}, ...}
set -euo pipefail

python - <<'PY'
import json
import os
import subprocess
import sys
from pathlib import Path

payload = json.load(sys.stdin)
project_dir = Path(os.environ.get("CLAUDE_PROJECT_DIR", Path.cwd())).resolve()
file_path_str = payload.get("tool_input", {}).get("file_path", "")
file_path = Path(file_path_str)

# Only trigger for model_registry YAML files
registry_dir = project_dir / "configs" / "model_registry"
try:
    file_path.relative_to(registry_dir)
except ValueError:
    sys.exit(0)

if file_path.suffix != ".yaml":
    sys.exit(0)

if not file_path.exists():
    sys.exit(0)

model_id = file_path.stem
event_payload = json.dumps({"model_id": model_id, "yaml_path": str(file_path)})

result = subprocess.run(
    ["uv", "run", "python", "-m", "src.models.automation", "registry_changed", event_payload],
    cwd=project_dir,
    capture_output=True,
    text=True,
)

if result.returncode != 0:
    sys.stderr.write(f"ModelForge automation failed for {model_id}:\n")
    sys.stderr.write(result.stderr[-2000:])
    # Non-blocking: exit 0 so Claude is not interrupted
    sys.exit(0)

output = result.stdout.strip()
if output:
    try:
        data = json.loads(output)
        v_status = data.get("verification_overall", "?")
        forge_status = data.get("forge_status", "?")
        sys.stdout.write(
            f"[ModelForge] {model_id}: verify={v_status}  forge={forge_status}\n"
        )
        failed = data.get("failed_checks", [])
        if failed:
            sys.stdout.write(f"  FAIL checks: {', '.join(failed)}\n")
    except json.JSONDecodeError:
        sys.stdout.write(output + "\n")

sys.exit(0)
PY
