#!/usr/bin/env bash
set -euo pipefail

project_dir="${CLAUDE_PROJECT_DIR:-$(pwd)}"
state_dir="$project_dir/.claude/state"
mkdir -p "$state_dir"
rm -f "$state_dir/lint.ok" "$state_dir/test-cov.ok"
