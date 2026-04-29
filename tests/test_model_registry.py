"""Tests for the model registry (T1400)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from models.registry import load_registry, search_registry
from schemas import ModelRegistryEntry


class TestLoadRegistry:
    def test_returns_at_least_ten_entries(self) -> None:
        entries = load_registry()
        assert len(entries) >= 10

    def test_all_entries_are_valid(self) -> None:
        entries = load_registry()
        for entry in entries:
            assert isinstance(entry, ModelRegistryEntry)
            assert entry.id
            assert entry.name
            assert entry.equations  # at least one equation

    def test_ids_are_unique(self) -> None:
        entries = load_registry()
        ids = [e.id for e in entries]
        assert len(ids) == len(set(ids))

    def test_nonexistent_dir_raises(self, tmp_path) -> None:
        from pathlib import Path

        with pytest.raises(FileNotFoundError):
            load_registry(registry_dir=Path(tmp_path) / "no_such_dir")


class TestSearchRegistry:
    @pytest.fixture(scope="class")
    def registry(self):
        return load_registry()

    def test_query_kalman(self, registry) -> None:
        results = search_registry(query="kalman", registry=registry)
        assert len(results) >= 1
        assert any(e.id == "kalman_filter" for e in results)

    def test_query_case_insensitive(self, registry) -> None:
        lower = search_registry(query="kalman", registry=registry)
        upper = search_registry(query="KALMAN", registry=registry)
        assert [e.id for e in lower] == [e.id for e in upper]

    def test_category_filter(self, registry) -> None:
        results = search_registry(category="state_estimation", registry=registry)
        assert all(e.category == "state_estimation" for e in results)
        assert len(results) >= 1

    def test_tags_filter(self, registry) -> None:
        results = search_registry(tags=["bayesian"], registry=registry)
        assert all("bayesian" in e.tags for e in results)
        assert len(results) >= 1

    def test_no_filters_returns_all(self, registry) -> None:
        results = search_registry(registry=registry)
        assert len(results) == len(registry)

    def test_nonexistent_query_returns_empty(self, registry) -> None:
        results = search_registry(query="xyzzy_no_match_ever", registry=registry)
        assert results == []


class TestRegistryRouter:
    @pytest.fixture(scope="class")
    def client(self):
        from api.app import app

        return TestClient(app)

    def test_list_registry(self, client) -> None:
        resp = client.get("/model/registry")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) >= 10

    def test_get_kalman_filter(self, client) -> None:
        resp = client.get("/model/registry/kalman_filter")
        assert resp.status_code == 200
        entry = resp.json()
        assert entry["id"] == "kalman_filter"
        assert "equations" in entry

    def test_get_unknown_model_returns_404(self, client) -> None:
        resp = client.get("/model/registry/does_not_exist")
        assert resp.status_code == 404

    def test_filter_by_query(self, client) -> None:
        resp = client.get("/model/registry", params={"query": "kalman"})
        assert resp.status_code == 200
        ids = [e["id"] for e in resp.json()]
        assert "kalman_filter" in ids

    def test_filter_by_tags(self, client) -> None:
        resp = client.get("/model/registry", params={"tags": "bayesian"})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) >= 1
        for entry in data:
            assert "bayesian" in entry["tags"]
