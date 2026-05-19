"""JSON persistence for GNSS Resilience Twin run results (T1500).

Saves the full twin run — request config, input observations, and diagnostic
report — as a single JSON file under ``output/<run_id>/twin_run.json``.

Schema
------
::

    {
      "schema_version": "1.0",
      "run_id":         "<8-char hex>",
      "produced_at":    "<ISO-8601 UTC>",
      "request":        { ... TwinRunRequest fields ... },
      "report":         { ... TwinRunReport fields  ... }
    }

``request`` preserves the raw ``ObservationEpoch`` list so that any future
re-run can be reproduced from the saved file alone.

Usage::

    from gnss.persistence import new_run_id, save_twin_run

    run_id = new_run_id()
    # ... run inference ...
    rel_path = save_twin_run(req.model_dump(), report.model_dump(), run_id)
    # rel_path → "output/a1b2c3d4/twin_run.json"
"""

from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

SCHEMA_VERSION: str = "1.0"

# Resolved once at import time; stable for the lifetime of the process.
_PROJECT_ROOT: Path = Path(__file__).parents[2]
_DEFAULT_OUTPUT_DIR: Path = _PROJECT_ROOT / "output"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def new_run_id() -> str:
    """Return a short unique run identifier (8 hex characters)."""
    return uuid.uuid4().hex[:8]


def save_twin_run(
    request_dict: dict[str, Any],
    report_dict: dict[str, Any],
    run_id: str,
    output_dir: Path | None = None,
) -> str:
    """Persist a twin run to ``<output_dir>/<run_id>/twin_run.json``.

    Parameters
    ----------
    request_dict:
        Serialised ``TwinRunRequest`` — typically from ``.model_dump(mode="json")``.
        Must include ``observations`` for reproducibility.
    report_dict:
        Serialised ``TwinRunReport`` — typically from ``.model_dump(mode="json")``.
    run_id:
        Unique run identifier, e.g. from :func:`new_run_id`.
    output_dir:
        Base output directory.  Defaults to ``<project_root>/output``.

    Returns
    -------
    str
        Path to the saved file, **relative to the project root**.
        Falls back to the absolute path when ``output_dir`` is outside the
        project tree (e.g. in tests using ``tmp_path``).
    """
    base = output_dir if output_dir is not None else _DEFAULT_OUTPUT_DIR
    run_dir = base / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "produced_at": datetime.now(timezone.utc).isoformat(),
        "request": request_dict,
        "report": report_dict,
    }

    out_path = run_dir / "twin_run.json"
    out_path.write_text(
        json.dumps(payload, indent=2, default=_json_default),
        encoding="utf-8",
    )

    try:
        return str(out_path.relative_to(_PROJECT_ROOT))
    except ValueError:
        # output_dir is outside project root (e.g. pytest tmp_path)
        return str(out_path)


def load_twin_run(path: str | Path) -> dict[str, Any]:
    """Load a previously saved twin run from disk.

    Parameters
    ----------
    path:
        Absolute path **or** path relative to the project root.

    Returns
    -------
    dict
        Parsed JSON payload with keys ``schema_version``, ``run_id``,
        ``produced_at``, ``request``, and ``report``.

    Raises
    ------
    FileNotFoundError
        When the file does not exist.
    ValueError
        When ``schema_version`` is missing or unsupported.
    """
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = _PROJECT_ROOT / resolved

    if not resolved.exists():
        raise FileNotFoundError(f"Twin run file not found: {resolved}")

    data: dict[str, Any] = json.loads(resolved.read_text(encoding="utf-8"))

    version = data.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(f"Unsupported schema_version '{version}'; expected '{SCHEMA_VERSION}'")

    return data


def purge_old_runs(
    max_age_days: int = 7,
    output_dir: Path | None = None,
) -> int:
    """Delete run directories older than *max_age_days*.

    Age is determined by the ``produced_at`` field in ``twin_run.json``.
    If the JSON is unreadable the directory's mtime is used as a fallback.

    Parameters
    ----------
    max_age_days:
        Runs older than this many days are removed.  Default: 7.
    output_dir:
        Base directory to scan.  Defaults to ``<project_root>/output``.

    Returns
    -------
    int
        Number of run directories deleted.
    """
    base = output_dir if output_dir is not None else _DEFAULT_OUTPUT_DIR
    if not base.exists():
        return 0

    now = datetime.now(timezone.utc)
    deleted = 0

    for run_dir in base.iterdir():
        if not run_dir.is_dir():
            continue

        age_days = _run_age_days(run_dir, now)
        if age_days is not None and age_days > max_age_days:
            shutil.rmtree(run_dir)
            deleted += 1

    return deleted


def _run_age_days(run_dir: Path, now: datetime) -> float | None:
    """Return the age of *run_dir* in days, or ``None`` on error."""
    json_path = run_dir / "twin_run.json"
    if json_path.exists():
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            produced_at = datetime.fromisoformat(data["produced_at"])
            if produced_at.tzinfo is None:
                produced_at = produced_at.replace(tzinfo=timezone.utc)
            return (now - produced_at).total_seconds() / 86400
        except (KeyError, ValueError, OSError):
            pass
    # Fallback: directory mtime
    try:
        mtime = datetime.fromtimestamp(run_dir.stat().st_mtime, tz=timezone.utc)
        return (now - mtime).total_seconds() / 86400
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _json_default(obj: Any) -> Any:
    """Fallback serialiser for types not handled by the standard library."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
