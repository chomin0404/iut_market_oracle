#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
import json
import os
import subprocess
import sys
from pathlib import Path

payload = json.load(sys.stdin)
project_dir = Path(os.environ.get("CLAUDE_PROJECT_DIR", Path.cwd())).resolve()
file_path = Path(payload.get("tool_input", {}).get("file_path", "")).resolve()

if not str(file_path).endswith('.py'):
    raise SystemExit(0)
if not file_path.exists():
    raise SystemExit(0)

try:
    relative = file_path.relative_to(project_dir)
except ValueError:
    relative = file_path

commands = [["ruff", "check", str(file_path)]]

candidate_tests = []
if "tests" in relative.parts:
    candidate_tests.append(file_path)
else:
    stem = file_path.stem
    tests_dir = project_dir / "tests"
    if tests_dir.exists():
        candidate_tests.extend(sorted(tests_dir.glob(f"test*{stem}*.py")))

unique_tests = []
seen = set()
for path in candidate_tests:
    if path.exists() and path not in seen:
        unique_tests.append(path)
        seen.add(path)

if unique_tests:
    commands.append(["python", "-m", "pytest", *[str(p) for p in unique_tests]])

results = []
for cmd in commands:
    completed = subprocess.run(cmd, cwd=project_dir, capture_output=True, text=True)
    results.append((cmd, completed.returncode, completed.stdout, completed.stderr))
    if completed.returncode != 0:
        sys.stderr.write(f"post-edit check failed for {relative}\n")
        sys.stderr.write(f"command: {' '.join(cmd)}\n")
        if completed.stdout:
            sys.stderr.write(completed.stdout[-4000:])
            if not completed.stdout.endswith("\n"):
                sys.stderr.write("\n")
        if completed.stderr:
            sys.stderr.write(completed.stderr[-4000:])
            if not completed.stderr.endswith("\n"):
                sys.stderr.write("\n")
        raise SystemExit(2)

summary = [f"post-edit checks passed for {relative}"]
for cmd, _, _, _ in results:
    summary.append("- " + " ".join(cmd))
sys.stdout.write("\n".join(summary) + "\n")
PY
