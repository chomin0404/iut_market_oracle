#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
import json
import os
from pathlib import Path

payload = json.load(__import__('sys').stdin)
if payload.get("stop_hook_active") is True:
    raise SystemExit(0)

project_dir = Path(os.environ.get("CLAUDE_PROJECT_DIR", Path.cwd()))
state_dir = project_dir / ".claude" / "state"
missing = []
if not (state_dir / "lint.ok").exists():
    missing.append("make lint")
if not (state_dir / "test-cov.ok").exists():
    missing.append("make test-cov")

if not missing:
    raise SystemExit(0)

reason = (
    "Before stopping, run the remaining verification commands: "
    + ", ".join(missing)
    + ". After they pass, you may stop."
)
print(json.dumps({"decision": "block", "reason": reason}, ensure_ascii=False))
PY
