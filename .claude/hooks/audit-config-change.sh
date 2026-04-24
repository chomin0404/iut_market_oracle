#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
import gzip
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

MAX_BYTES = 1_000_000
MAX_ARCHIVES = 10
COMPRESS_ARCHIVES = True

payload = json.load(sys.stdin)
project_dir = Path(os.environ.get("CLAUDE_PROJECT_DIR", Path.cwd()))
audit_dir = project_dir / ".claude" / "audit"
audit_dir.mkdir(parents=True, exist_ok=True)
audit_log = audit_dir / "config-changes.jsonl"
rotation_manifest = audit_dir / "rotation-policy.json"

file_path = str(payload.get("file_path", ""))
source = str(payload.get("source", ""))
hook_event_name = str(payload.get("hook_event_name", "ConfigChange"))

watched = (
    file_path.endswith(".claude/settings.json")
    or file_path.endswith("CLAUDE.md")
    or file_path.endswith("pyproject.toml")
    or "/.github/workflows/" in file_path
)

if not watched:
    raise SystemExit(0)


def rotate_if_needed(path: Path) -> None:
    if not path.exists() or path.stat().st_size < MAX_BYTES:
        return

    oldest_plain = path.with_name(f"{path.name}.10")
    oldest_gz = path.with_name(f"{path.name}.10.gz")
    if oldest_plain.exists():
        oldest_plain.unlink()
    if oldest_gz.exists():
        oldest_gz.unlink()

    for index in range(MAX_ARCHIVES - 1, 0, -1):
        src_plain = path.with_name(f"{path.name}.{index}")
        src_gz = path.with_name(f"{path.name}.{index}.gz")
        dst_plain = path.with_name(f"{path.name}.{index + 1}")
        dst_gz = path.with_name(f"{path.name}.{index + 1}.gz")

        if src_gz.exists():
            src_gz.rename(dst_gz)
        elif src_plain.exists():
            src_plain.rename(dst_plain)

    rotated = path.with_name(f"{path.name}.1")
    shutil.move(str(path), str(rotated))

    if COMPRESS_ARCHIVES:
        gz_path = rotated.with_suffix(rotated.suffix + ".gz")
        with rotated.open("rb") as src, gzip.open(gz_path, "wb", compresslevel=9) as dst:
            shutil.copyfileobj(src, dst)
        rotated.unlink()


rotate_if_needed(audit_log)

record = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "hook_event_name": hook_event_name,
    "source": source,
    "file_path": file_path,
    "cwd": payload.get("cwd"),
    "session_id": payload.get("session_id"),
}

with audit_log.open("a", encoding="utf-8") as f:
    f.write(json.dumps(record, ensure_ascii=False) + "\n")

policy = {
    "path": str(audit_log.relative_to(project_dir)),
    "rotation": "size-based",
    "max_bytes": MAX_BYTES,
    "archive_count": MAX_ARCHIVES,
    "compression": "gzip" if COMPRESS_ARCHIVES else "none",
    "retention_rule": "keep latest active log plus 10 archives",
    "scope": [
        ".claude/settings.json",
        "CLAUDE.md",
        "pyproject.toml",
        ".github/workflows/*",
    ],
}
rotation_manifest.write_text(json.dumps(policy, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
