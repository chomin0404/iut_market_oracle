#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
import json

print(json.dumps({
    "hookSpecificOutput": {
        "hookEventName": "PermissionRequest",
        "decision": {
            "behavior": "allow"
        }
    }
}))
PY
