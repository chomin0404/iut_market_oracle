#!/usr/bin/env bash
set -euo pipefail

kind="${1:-}"
project_dir="${CLAUDE_PROJECT_DIR:-$(pwd)}"
state_dir="$project_dir/.claude/state"
mkdir -p "$state_dir"

case "$kind" in
  lint)
    printf '%s\n' "ok" > "$state_dir/lint.ok"
    ;;
  test-cov)
    printf '%s\n' "ok" > "$state_dir/test-cov.ok"
    ;;
  *)
    echo "Unknown verification marker: $kind" >&2
    exit 1
    ;;
esac
