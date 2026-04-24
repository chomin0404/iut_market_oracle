#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
import json
import sys
from pathlib import Path

payload = json.load(sys.stdin)
file_path_raw = payload.get("tool_input", {}).get("file_path", "")
if not file_path_raw:
    raise SystemExit(0)

file_path = Path(file_path_raw)
text = file_path.as_posix()
name = file_path.name
parts = file_path.parts

protected_exact = {"LICENSE", ".env", ".env.local", ".envrc", ".coverage"}
protected_prefixes = (".git/",)
allowed_artifact_roots = {"output", "coverage", "dist", "build"}
generated_like_names = {
    "coverage.xml",
    "coverage.json",
}
generated_like_suffixes = (
    ".pyc",
    ".pyo",
    ".whl",
    ".tar.gz",
)

if name in protected_exact or name.startswith('.env.'):
    sys.stderr.write(f"Blocked protected file: {text}\n")
    raise SystemExit(2)

for prefix in protected_prefixes:
    if text.startswith(prefix) or f"/{prefix}" in text:
        sys.stderr.write(f"Blocked protected path: {text}\n")
        raise SystemExit(2)

if name == "LICENSE":
    sys.stderr.write(f"Blocked protected file: {text}\n")
    raise SystemExit(2)

is_allowed_artifact = bool(parts) and parts[0] in allowed_artifact_roots
looks_generated = (
    name in generated_like_names
    or any(text.endswith(suffix) for suffix in generated_like_suffixes)
    or ".egg-info" in text
    or "__pycache__" in parts
)

if looks_generated and not is_allowed_artifact:
    sys.stderr.write(
        "Blocked generated-file edit outside approved artifact directories: "
        f"{text}. Use output/, coverage/, dist/, or build/.\n"
    )
    raise SystemExit(2)
PY
