"""Model registry: load and search ModelRegistryEntry items from YAML configs."""

from __future__ import annotations

from pathlib import Path

import yaml

from schemas import ModelRegistryEntry

DEFAULT_REGISTRY_DIR = Path(__file__).parent.parent.parent / "configs" / "model_registry"


def load_registry(registry_dir: Path = DEFAULT_REGISTRY_DIR) -> list[ModelRegistryEntry]:
    """Load all *.yaml files in *registry_dir* and return validated ModelRegistryEntry list.

    Raises:
        FileNotFoundError: if registry_dir does not exist.
        pydantic.ValidationError: if any YAML fails ModelRegistryEntry validation.
    """
    if not registry_dir.is_dir():
        raise FileNotFoundError(f"Registry directory not found: {registry_dir}")

    entries: list[ModelRegistryEntry] = []
    for yaml_path in sorted(registry_dir.glob("*.yaml")):
        with yaml_path.open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        entries.append(ModelRegistryEntry(**data))
    return entries


def search_registry(
    query: str | None = None,
    category: str | None = None,
    tags: list[str] | None = None,
    registry: list[ModelRegistryEntry] | None = None,
) -> list[ModelRegistryEntry]:
    """Filter the registry by optional query, category, and/or tags.

    Args:
        query:    Case-insensitive substring matched against *name*, *problem_type*,
                  and the string representation of *tags*.
        category: Exact (case-insensitive) match on entry.category.
        tags:     All provided tags must appear in the entry's tag list
                  (case-insensitive subset match).
        registry: Pre-loaded registry; if None, load_registry() is called.

    Returns:
        Filtered list of ModelRegistryEntry objects.
    """
    if registry is None:
        registry = load_registry()

    results = registry

    if category is not None:
        cat_lower = category.lower()
        results = [e for e in results if e.category.lower() == cat_lower]

    if tags is not None:
        required = {t.lower() for t in tags}
        results = [e for e in results if required.issubset({t.lower() for t in e.tags})]

    if query is not None:
        q = query.lower()
        results = [
            e
            for e in results
            if q in e.name.lower()
            or q in e.problem_type.lower()
            or any(q in t.lower() for t in e.tags)
        ]

    return results
