"""Tests for src/experiments/tracker.py."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from experiments.tracker import (
    META_FILENAME,
    REGISTRY_FILENAME,
    create_experiment,
    get_next_id,
    list_experiments,
    load_experiment,
    read_meta,
    update_experiment,
    write_meta,
)
from schemas import ExperimentMeta

# ---------------------------------------------------------------------------
# get_next_id
# ---------------------------------------------------------------------------


class TestGetNextId:
    def test_empty_dir_returns_exp_001(self, tmp_path: Path):
        assert get_next_id(tmp_path) == "exp-001"

    def test_nonexistent_dir_returns_exp_001(self, tmp_path: Path):
        assert get_next_id(tmp_path / "does_not_exist") == "exp-001"

    def test_increments_from_existing(self, tmp_path: Path):
        (tmp_path / "exp-001").mkdir()
        (tmp_path / "exp-002").mkdir()
        assert get_next_id(tmp_path) == "exp-003"

    def test_skips_gaps(self, tmp_path: Path):
        (tmp_path / "exp-001").mkdir()
        (tmp_path / "exp-005").mkdir()
        assert get_next_id(tmp_path) == "exp-006"

    def test_ignores_non_matching_dirs(self, tmp_path: Path):
        (tmp_path / "scratch").mkdir()
        (tmp_path / "exp-001").mkdir()
        assert get_next_id(tmp_path) == "exp-002"

    def test_ignores_files_with_exp_name(self, tmp_path: Path):
        # Files are not directories; should be ignored
        (tmp_path / "exp-001").touch()
        assert get_next_id(tmp_path) == "exp-001"

    def test_zero_padded_three_digits(self, tmp_path: Path):
        assert get_next_id(tmp_path).startswith("exp-")
        assert len(get_next_id(tmp_path)) == 7  # "exp-NNN"


# ---------------------------------------------------------------------------
# write_meta / read_meta round-trip
# ---------------------------------------------------------------------------


class TestMetaRoundTrip:
    def _make_meta(self, exp_id: str = "exp-001") -> ExperimentMeta:
        return ExperimentMeta(
            experiment_id=exp_id,
            title="Test experiment",
            config_path="configs/scenarios/base.yaml",
            tags=["dcf", "test"],
            random_seed=42,
            summary="Unit test run",
        )

    def test_round_trip_preserves_all_fields(self, tmp_path: Path):
        meta = self._make_meta()
        write_meta(meta, tmp_path)
        loaded = read_meta(tmp_path)
        assert loaded.experiment_id == meta.experiment_id
        assert loaded.title == meta.title
        assert loaded.config_path == meta.config_path
        assert loaded.tags == meta.tags
        assert loaded.random_seed == meta.random_seed
        assert loaded.summary == meta.summary

    def test_meta_yaml_is_valid_yaml(self, tmp_path: Path):
        write_meta(self._make_meta(), tmp_path)
        raw = yaml.safe_load((tmp_path / META_FILENAME).read_text())
        assert isinstance(raw, dict)
        assert raw["experiment_id"] == "exp-001"

    def test_none_fields_survive_round_trip(self, tmp_path: Path):
        meta = ExperimentMeta(
            experiment_id="exp-002",
            title="Minimal",
            config_path="configs/x.yaml",
        )
        write_meta(meta, tmp_path)
        loaded = read_meta(tmp_path)
        assert loaded.result_path is None
        assert loaded.note_path is None
        assert loaded.random_seed is None


# ---------------------------------------------------------------------------
# create_experiment
# ---------------------------------------------------------------------------


class TestCreateExperiment:
    def test_creates_directory(self, tmp_path: Path):
        meta = create_experiment(
            title="First experiment",
            config_path="configs/base.yaml",
            experiments_root=tmp_path,
        )
        assert (tmp_path / meta.experiment_id).is_dir()

    def test_writes_meta_yaml(self, tmp_path: Path):
        meta = create_experiment(
            title="Meta test",
            config_path="configs/base.yaml",
            experiments_root=tmp_path,
        )
        assert (tmp_path / meta.experiment_id / META_FILENAME).exists()

    def test_creates_registry_md(self, tmp_path: Path):
        create_experiment(
            title="Registry test",
            config_path="configs/base.yaml",
            experiments_root=tmp_path,
        )
        assert (tmp_path / REGISTRY_FILENAME).exists()

    def test_registry_contains_experiment_id(self, tmp_path: Path):
        meta = create_experiment(
            title="Registry row test",
            config_path="configs/base.yaml",
            experiments_root=tmp_path,
        )
        registry = (tmp_path / REGISTRY_FILENAME).read_text(encoding="utf-8")
        assert meta.experiment_id in registry

    def test_sequential_ids(self, tmp_path: Path):
        m1 = create_experiment("First", "cfg1.yaml", experiments_root=tmp_path)
        m2 = create_experiment("Second", "cfg2.yaml", experiments_root=tmp_path)
        m3 = create_experiment("Third", "cfg3.yaml", experiments_root=tmp_path)
        assert m1.experiment_id == "exp-001"
        assert m2.experiment_id == "exp-002"
        assert m3.experiment_id == "exp-003"

    def test_tags_stored(self, tmp_path: Path):
        meta = create_experiment(
            title="Tagged",
            config_path="cfg.yaml",
            tags=["dcf", "bull"],
            experiments_root=tmp_path,
        )
        loaded = load_experiment(meta.experiment_id, tmp_path)
        assert loaded.tags == ["dcf", "bull"]

    def test_random_seed_stored(self, tmp_path: Path):
        meta = create_experiment(
            title="Seeded",
            config_path="cfg.yaml",
            random_seed=123,
            experiments_root=tmp_path,
        )
        loaded = load_experiment(meta.experiment_id, tmp_path)
        assert loaded.random_seed == 123

    def test_summary_stored(self, tmp_path: Path):
        meta = create_experiment(
            title="Summary test",
            config_path="cfg.yaml",
            summary="Testing the summary field.",
            experiments_root=tmp_path,
        )
        loaded = load_experiment(meta.experiment_id, tmp_path)
        assert loaded.summary == "Testing the summary field."

    def test_optional_paths_stored(self, tmp_path: Path):
        meta = create_experiment(
            title="Paths test",
            config_path="cfg.yaml",
            result_path="experiments/exp-001/result.yaml",
            note_path="notes/derivation.md",
            experiments_root=tmp_path,
        )
        loaded = load_experiment(meta.experiment_id, tmp_path)
        assert loaded.result_path == "experiments/exp-001/result.yaml"
        assert loaded.note_path == "notes/derivation.md"

    def test_registry_rows_accumulate(self, tmp_path: Path):
        for i in range(3):
            create_experiment(f"Exp {i}", "cfg.yaml", experiments_root=tmp_path)
        content = (tmp_path / REGISTRY_FILENAME).read_text(encoding="utf-8")
        assert content.count("exp-00") == 3


# ---------------------------------------------------------------------------
# load_experiment
# ---------------------------------------------------------------------------


class TestLoadExperiment:
    def test_load_existing(self, tmp_path: Path):
        meta = create_experiment("Load test", "cfg.yaml", experiments_root=tmp_path)
        loaded = load_experiment(meta.experiment_id, tmp_path)
        assert loaded.experiment_id == meta.experiment_id
        assert loaded.title == meta.title

    def test_missing_dir_raises_file_not_found(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_experiment("exp-099", tmp_path)

    def test_missing_meta_raises_file_not_found(self, tmp_path: Path):
        (tmp_path / "exp-001").mkdir()  # directory without meta.yaml
        with pytest.raises(FileNotFoundError, match="meta.yaml"):
            load_experiment("exp-001", tmp_path)


# ---------------------------------------------------------------------------
# list_experiments
# ---------------------------------------------------------------------------


class TestListExperiments:
    def test_empty_root_returns_empty(self, tmp_path: Path):
        assert list_experiments(tmp_path) == []

    def test_nonexistent_root_returns_empty(self, tmp_path: Path):
        assert list_experiments(tmp_path / "ghost") == []

    def test_returns_all_experiments(self, tmp_path: Path):
        for i in range(4):
            create_experiment(f"Exp {i}", "cfg.yaml", experiments_root=tmp_path)
        results = list_experiments(tmp_path)
        assert len(results) == 4

    def test_sorted_by_id(self, tmp_path: Path):
        for i in range(5):
            create_experiment(f"Exp {i}", "cfg.yaml", experiments_root=tmp_path)
        results = list_experiments(tmp_path)
        ids = [r.experiment_id for r in results]
        assert ids == sorted(ids)

    def test_skips_corrupted_meta(self, tmp_path: Path):
        create_experiment("Good", "cfg.yaml", experiments_root=tmp_path)
        bad_dir = tmp_path / "exp-002"
        bad_dir.mkdir()
        (bad_dir / META_FILENAME).write_text("not: valid: yaml: [", encoding="utf-8")
        results = list_experiments(tmp_path)
        assert len(results) == 1  # only the good one


# ---------------------------------------------------------------------------
# update_experiment
# ---------------------------------------------------------------------------


class TestUpdateExperiment:
    def test_update_result_path(self, tmp_path: Path):
        meta = create_experiment("Update test", "cfg.yaml", experiments_root=tmp_path)
        updated = update_experiment(
            meta.experiment_id,
            experiments_root=tmp_path,
            result_path="experiments/exp-001/result.yaml",
        )
        assert updated.result_path == "experiments/exp-001/result.yaml"
        # Persisted on disk:
        reloaded = load_experiment(meta.experiment_id, tmp_path)
        assert reloaded.result_path == "experiments/exp-001/result.yaml"

    def test_update_summary(self, tmp_path: Path):
        meta = create_experiment("Sum update", "cfg.yaml", experiments_root=tmp_path)
        updated = update_experiment(
            meta.experiment_id,
            experiments_root=tmp_path,
            summary="Updated narrative.",
        )
        assert updated.summary == "Updated narrative."

    def test_update_tags(self, tmp_path: Path):
        meta = create_experiment("Tag update", "cfg.yaml", experiments_root=tmp_path)
        updated = update_experiment(
            meta.experiment_id,
            experiments_root=tmp_path,
            tags=["new", "tags"],
        )
        assert updated.tags == ["new", "tags"]

    def test_unknown_field_raises(self, tmp_path: Path):
        meta = create_experiment("Bad field", "cfg.yaml", experiments_root=tmp_path)
        with pytest.raises(ValueError, match="not updatable"):
            update_experiment(
                meta.experiment_id,
                experiments_root=tmp_path,
                title="Cannot change title",
            )

    def test_immutable_fields_unchanged_after_update(self, tmp_path: Path):
        meta = create_experiment(
            "Immutable check",
            "cfg.yaml",
            random_seed=7,
            experiments_root=tmp_path,
        )
        updated = update_experiment(
            meta.experiment_id,
            experiments_root=tmp_path,
            summary="Changed summary",
        )
        assert updated.experiment_id == meta.experiment_id
        assert updated.title == meta.title
        assert updated.random_seed == meta.random_seed
        assert updated.config_path == meta.config_path
