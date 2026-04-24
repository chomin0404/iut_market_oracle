"""Experiment registry and folder management.

Each experiment occupies one directory under EXPERIMENTS_ROOT:

    experiments/
        registry.md          ← Markdown table index (one row per experiment)
        exp-001/
            meta.yaml        ← ExperimentMeta serialised as YAML
            <config, result, notes placed by caller>
        exp-002/
            meta.yaml
        ...

ID format: ``exp-NNN`` (three-digit zero-padded integer, e.g. ``exp-007``).

Typical usage
-------------
    from experiments.tracker import create_experiment, load_experiment

    meta = create_experiment(
        title="Baseline DCF — bull scenario",
        config_path="configs/scenarios/bull.yaml",
        tags=["valuation", "dcf"],
        random_seed=42,
        summary="First run with updated WACC assumption.",
        experiments_root="experiments/",
    )
    # meta.experiment_id == "exp-001"   (or next available)
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

import yaml

from schemas import ExperimentMeta

# Default location relative to the project root.  Callers may override.
DEFAULT_EXPERIMENTS_ROOT = Path("experiments")
REGISTRY_FILENAME = "registry.md"
META_FILENAME = "meta.yaml"

_ID_PATTERN = re.compile(r"^exp-(\d{3})$")


# ---------------------------------------------------------------------------
# ID management
# ---------------------------------------------------------------------------


def get_next_id(experiments_root: str | Path = DEFAULT_EXPERIMENTS_ROOT) -> str:
    """Return the next available experiment ID (e.g. ``'exp-003'``).

    Scans *experiments_root* for directories whose names match ``exp-NNN``
    and returns the successor of the highest existing ID.  Returns
    ``'exp-001'`` when no experiments exist yet.

    Parameters
    ----------
    experiments_root:
        Directory that contains experiment sub-folders.

    Returns
    -------
    str
        Next ID in ``exp-NNN`` format.
    """
    root = Path(experiments_root)
    existing: list[int] = []
    if root.is_dir():
        for child in root.iterdir():
            m = _ID_PATTERN.match(child.name)
            if m and child.is_dir():
                existing.append(int(m.group(1)))
    next_n = max(existing, default=0) + 1
    if next_n > 999:
        raise OverflowError("Experiment ID would exceed exp-999. Archive old experiments first.")
    return f"exp-{next_n:03d}"


def _id_to_int(exp_id: str) -> int:
    m = _ID_PATTERN.match(exp_id)
    if not m:
        raise ValueError(f"Invalid experiment ID format: {exp_id!r}")
    return int(m.group(1))


# ---------------------------------------------------------------------------
# Meta serialisation
# ---------------------------------------------------------------------------


def _meta_to_dict(meta: ExperimentMeta) -> dict:
    return {
        "experiment_id": meta.experiment_id,
        "title": meta.title,
        "config_path": meta.config_path,
        "result_path": meta.result_path,
        "note_path": meta.note_path,
        "random_seed": meta.random_seed,
        "tags": meta.tags,
        "created_at": meta.created_at.isoformat(),
        "summary": meta.summary,
    }


def _dict_to_meta(d: dict) -> ExperimentMeta:
    created_at = d.get("created_at")
    if isinstance(created_at, str):
        created_at = datetime.fromisoformat(created_at)
    return ExperimentMeta(
        experiment_id=d["experiment_id"],
        title=d["title"],
        config_path=d["config_path"],
        result_path=d.get("result_path"),
        note_path=d.get("note_path"),
        random_seed=d.get("random_seed"),
        tags=d.get("tags", []),
        created_at=created_at or datetime.now(UTC),
        summary=d.get("summary", ""),
    )


def write_meta(meta: ExperimentMeta, exp_dir: Path) -> Path:
    """Write *meta* as YAML inside *exp_dir/meta.yaml*.  Returns the path."""
    dest = exp_dir / META_FILENAME
    dest.write_text(
        yaml.dump(_meta_to_dict(meta), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return dest


def read_meta(exp_dir: Path) -> ExperimentMeta:
    """Read and validate ExperimentMeta from *exp_dir/meta.yaml*."""
    raw = yaml.safe_load((exp_dir / META_FILENAME).read_text(encoding="utf-8"))
    return _dict_to_meta(raw)


# ---------------------------------------------------------------------------
# Registry (Markdown table)
# ---------------------------------------------------------------------------

_REGISTRY_HEADER = (
    "# Experiment Registry\n\n"
    "| ID | Title | Created | Tags | Summary |\n"
    "|----|-------|---------|------|---------|\n"
)


def _registry_row(meta: ExperimentMeta) -> str:
    tags = ", ".join(meta.tags) if meta.tags else "—"
    created = meta.created_at.strftime("%Y-%m-%d %H:%M")
    summary = meta.summary.replace("\n", " ").replace("|", "\\|")[:80]
    return f"| {meta.experiment_id} | {meta.title} | {created} | {tags} | {summary} |\n"


def _append_registry_row(meta: ExperimentMeta, registry_path: Path) -> None:
    """Append one row to registry.md, creating the file+header if needed."""
    if not registry_path.exists():
        registry_path.write_text(_REGISTRY_HEADER, encoding="utf-8")
    with registry_path.open("a", encoding="utf-8") as f:
        f.write(_registry_row(meta))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def create_experiment(
    title: str,
    config_path: str,
    *,
    result_path: str | None = None,
    note_path: str | None = None,
    random_seed: int | None = None,
    tags: list[str] | None = None,
    summary: str = "",
    experiments_root: str | Path = DEFAULT_EXPERIMENTS_ROOT,
) -> ExperimentMeta:
    """Create a new experiment folder and register it.

    1. Assigns the next available ID.
    2. Creates ``experiments_root/exp-NNN/``.
    3. Writes ``meta.yaml`` inside that folder.
    4. Appends a row to ``experiments_root/registry.md``.

    Parameters
    ----------
    title:
        Short human-readable description of the experiment.
    config_path:
        Relative path to the config file used (e.g. ``configs/scenarios/base.yaml``).
    result_path:
        Relative path to the output artifact, if known at creation time.
    note_path:
        Relative path to a research note or derivation document.
    random_seed:
        RNG seed used in the run (for reproducibility).
    tags:
        Arbitrary labels for filtering (e.g. ``["dcf", "bull"]``).
    summary:
        One-sentence narrative description of what this run tests.
    experiments_root:
        Root directory for experiment folders.

    Returns
    -------
    ExperimentMeta
        Validated metadata object for the new experiment.
    """
    root = Path(experiments_root)
    root.mkdir(parents=True, exist_ok=True)

    exp_id = get_next_id(root)
    exp_dir = root / exp_id
    exp_dir.mkdir()

    meta = ExperimentMeta(
        experiment_id=exp_id,
        title=title,
        config_path=config_path,
        result_path=result_path,
        note_path=note_path,
        random_seed=random_seed,
        tags=tags or [],
        created_at=datetime.now(UTC),
        summary=summary,
    )

    write_meta(meta, exp_dir)
    _append_registry_row(meta, root / REGISTRY_FILENAME)

    return meta


def load_experiment(
    exp_id: str,
    experiments_root: str | Path = DEFAULT_EXPERIMENTS_ROOT,
) -> ExperimentMeta:
    """Load a previously created experiment by ID.

    Parameters
    ----------
    exp_id:
        Identifier in ``exp-NNN`` format.
    experiments_root:
        Root directory for experiment folders.

    Raises
    ------
    FileNotFoundError
        If the experiment directory or meta.yaml does not exist.
    """
    root = Path(experiments_root)
    exp_dir = root / exp_id
    if not exp_dir.is_dir():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")
    meta_path = exp_dir / META_FILENAME
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.yaml not found in {exp_dir}")
    return read_meta(exp_dir)


def list_experiments(
    experiments_root: str | Path = DEFAULT_EXPERIMENTS_ROOT,
) -> list[ExperimentMeta]:
    """Return all experiments found in *experiments_root*, sorted by ID.

    Directories that do not contain a valid ``meta.yaml`` are silently skipped.
    """
    root = Path(experiments_root)
    results: list[ExperimentMeta] = []
    if not root.is_dir():
        return results

    dirs = sorted(
        (d for d in root.iterdir() if _ID_PATTERN.match(d.name) and d.is_dir()),
        key=lambda d: _id_to_int(d.name),
    )
    for exp_dir in dirs:
        try:
            results.append(read_meta(exp_dir))
        except Exception:
            pass  # corrupted meta — skip, do not fail the listing

    return results


def update_experiment(
    exp_id: str,
    experiments_root: str | Path = DEFAULT_EXPERIMENTS_ROOT,
    **fields,
) -> ExperimentMeta:
    """Update writable fields of an existing experiment's meta.yaml.

    Allowed fields: ``result_path``, ``note_path``, ``summary``, ``tags``.

    Parameters
    ----------
    exp_id:
        Identifier in ``exp-NNN`` format.
    **fields:
        Key-value pairs to update.

    Returns
    -------
    ExperimentMeta
        Updated metadata.
    """
    _allowed = {"result_path", "note_path", "summary", "tags"}
    unknown = set(fields) - _allowed
    if unknown:
        raise ValueError(f"Fields not updatable: {unknown}. Allowed: {_allowed}")

    root = Path(experiments_root)
    exp_dir = root / exp_id
    meta = read_meta(exp_dir)

    raw = _meta_to_dict(meta)
    raw.update(fields)
    updated = _dict_to_meta(raw)
    write_meta(updated, exp_dir)
    return updated
