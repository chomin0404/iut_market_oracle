"""Tests for src/huh_twin/skill_graph.py.

Invariants
----------
ClassifiedSkill
    Frozen dataclass — mutation must raise FrozenInstanceError.

SkillBasis
    Exactly five members; all values are strings.

SKILL_CLASSIFICATION
    - Every entry is a ClassifiedSkill with a valid SkillBasis.
    - No duplicate names.
    - No empty names or notes.

EXPECTED_PROMPT_ITEMS
    - Derived from SKILL_CLASSIFICATION by default → no mismatches.
    - No duplicates by default.

grouped_by_basis
    - All SkillBasis keys present (even if list is empty).
    - Every item appears in exactly one group.
    - Each item is placed under its declared basis.

prompt_items_missing_from_classification
    - Returns [] when lists are in sync.
    - Returns missing names when EXPECTED_PROMPT_ITEMS has extras.

classification_items_not_in_prompt
    - Returns [] when lists are in sync.
    - Returns orphaned names when classification has extras.

duplicate_prompt_items
    - Returns [] for the canonical (no-duplicate) list.
    - Returns each duplicate name exactly once.
    - Does not report names that appear only once.
"""

from __future__ import annotations

import pytest

from huh_twin.skill_graph import (
    EXPECTED_PROMPT_ITEMS,
    SKILL_CLASSIFICATION,
    ClassifiedSkill,
    SkillBasis,
    classification_items_not_in_prompt,
    duplicate_prompt_items,
    grouped_by_basis,
    prompt_items_missing_from_classification,
)


# ---------------------------------------------------------------------------
# SkillBasis
# ---------------------------------------------------------------------------


class TestSkillBasis:
    def test_exactly_five_members(self) -> None:
        assert len(SkillBasis) == 5

    def test_all_members_are_strings(self) -> None:
        for basis in SkillBasis:
            assert isinstance(basis.value, str)

    def test_expected_values_present(self) -> None:
        values = {b.value for b in SkillBasis}
        assert "probabilistic_inference" in values
        assert "valuation_and_capital_allocation" in values
        assert "inverse_problems_and_simulation" in values
        assert "optimal_experiment_design" in values
        assert "strategy_and_implementation" in values


# ---------------------------------------------------------------------------
# ClassifiedSkill
# ---------------------------------------------------------------------------


class TestClassifiedSkill:
    def test_fields_accessible(self) -> None:
        skill = ClassifiedSkill(
            name="test_skill",
            basis=SkillBasis.PROBABILISTIC_INFERENCE,
            note="A test note.",
        )
        assert skill.name == "test_skill"
        assert skill.basis == SkillBasis.PROBABILISTIC_INFERENCE
        assert skill.note == "A test note."

    def test_frozen_raises_on_mutation(self) -> None:
        skill = ClassifiedSkill(
            name="x",
            basis=SkillBasis.STRATEGY_AND_IMPLEMENTATION,
            note="y",
        )
        with pytest.raises((AttributeError, TypeError)):
            skill.name = "mutated"  # type: ignore[misc]

    def test_equality_by_value(self) -> None:
        a = ClassifiedSkill("s", SkillBasis.PROBABILISTIC_INFERENCE, "n")
        b = ClassifiedSkill("s", SkillBasis.PROBABILISTIC_INFERENCE, "n")
        assert a == b

    def test_inequality_on_name_difference(self) -> None:
        a = ClassifiedSkill("s1", SkillBasis.PROBABILISTIC_INFERENCE, "n")
        b = ClassifiedSkill("s2", SkillBasis.PROBABILISTIC_INFERENCE, "n")
        assert a != b


# ---------------------------------------------------------------------------
# SKILL_CLASSIFICATION
# ---------------------------------------------------------------------------


class TestSkillClassification:
    def test_non_empty(self) -> None:
        assert len(SKILL_CLASSIFICATION) > 0

    def test_all_entries_are_classified_skill(self) -> None:
        for item in SKILL_CLASSIFICATION:
            assert isinstance(item, ClassifiedSkill)

    def test_all_bases_are_valid_skill_basis(self) -> None:
        valid = set(SkillBasis)
        for item in SKILL_CLASSIFICATION:
            assert item.basis in valid

    def test_no_duplicate_names(self) -> None:
        names = [item.name for item in SKILL_CLASSIFICATION]
        assert len(names) == len(set(names))

    def test_no_empty_names(self) -> None:
        for item in SKILL_CLASSIFICATION:
            assert item.name.strip() != ""

    def test_no_empty_notes(self) -> None:
        for item in SKILL_CLASSIFICATION:
            assert item.note.strip() != ""

    def test_all_skill_bases_represented(self) -> None:
        used_bases = {item.basis for item in SKILL_CLASSIFICATION}
        assert used_bases == set(SkillBasis)


# ---------------------------------------------------------------------------
# grouped_by_basis
# ---------------------------------------------------------------------------


class TestGroupedByBasis:
    def test_all_basis_keys_present(self) -> None:
        grouped = grouped_by_basis()
        assert set(grouped.keys()) == set(SkillBasis)

    def test_total_item_count_matches_classification(self) -> None:
        grouped = grouped_by_basis()
        total = sum(len(items) for items in grouped.values())
        assert total == len(SKILL_CLASSIFICATION)

    def test_each_item_appears_exactly_once(self) -> None:
        grouped = grouped_by_basis()
        all_items = [item for items in grouped.values() for item in items]
        assert len(all_items) == len(set(id(item) for item in all_items))

    def test_items_placed_under_correct_basis(self) -> None:
        grouped = grouped_by_basis()
        for basis, items in grouped.items():
            for item in items:
                assert item.basis == basis

    def test_returns_new_dict_each_call(self) -> None:
        # Mutations to one result must not affect another
        g1 = grouped_by_basis()
        g2 = grouped_by_basis()
        g1[SkillBasis.PROBABILISTIC_INFERENCE].clear()
        assert len(g2[SkillBasis.PROBABILISTIC_INFERENCE]) > 0

    def test_probabilistic_inference_non_empty(self) -> None:
        assert len(grouped_by_basis()[SkillBasis.PROBABILISTIC_INFERENCE]) > 0

    def test_valuation_non_empty(self) -> None:
        assert len(grouped_by_basis()[SkillBasis.VALUATION_AND_CAPITAL_ALLOCATION]) > 0


# ---------------------------------------------------------------------------
# prompt_items_missing_from_classification
# ---------------------------------------------------------------------------


class TestPromptItemsMissingFromClassification:
    def test_returns_empty_when_in_sync(self) -> None:
        # Default EXPECTED_PROMPT_ITEMS is derived from SKILL_CLASSIFICATION
        assert prompt_items_missing_from_classification() == []

    def test_detects_missing_item(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import huh_twin.skill_graph as sg
        monkeypatch.setattr(sg, "EXPECTED_PROMPT_ITEMS", ["ghost_skill"])
        result = sg.prompt_items_missing_from_classification()
        assert "ghost_skill" in result

    def test_does_not_report_items_present_in_both(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import huh_twin.skill_graph as sg
        present_name = SKILL_CLASSIFICATION[0].name
        monkeypatch.setattr(sg, "EXPECTED_PROMPT_ITEMS", [present_name])
        assert sg.prompt_items_missing_from_classification() == []


# ---------------------------------------------------------------------------
# classification_items_not_in_prompt
# ---------------------------------------------------------------------------


class TestClassificationItemsNotInPrompt:
    def test_returns_empty_when_in_sync(self) -> None:
        assert classification_items_not_in_prompt() == []

    def test_detects_orphaned_classification_item(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import huh_twin.skill_graph as sg
        monkeypatch.setattr(sg, "EXPECTED_PROMPT_ITEMS", [])
        result = sg.classification_items_not_in_prompt()
        assert len(result) == len(SKILL_CLASSIFICATION)

    def test_does_not_report_matched_items(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import huh_twin.skill_graph as sg
        all_names = [item.name for item in SKILL_CLASSIFICATION]
        monkeypatch.setattr(sg, "EXPECTED_PROMPT_ITEMS", all_names)
        assert sg.classification_items_not_in_prompt() == []


# ---------------------------------------------------------------------------
# duplicate_prompt_items
# ---------------------------------------------------------------------------


class TestDuplicatePromptItems:
    def test_returns_empty_for_canonical_list(self) -> None:
        assert duplicate_prompt_items() == []

    def test_detects_single_duplicate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import huh_twin.skill_graph as sg
        monkeypatch.setattr(sg, "EXPECTED_PROMPT_ITEMS", ["a", "b", "a"])
        assert sg.duplicate_prompt_items() == ["a"]

    def test_reports_duplicate_only_once_even_if_triplicated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import huh_twin.skill_graph as sg
        monkeypatch.setattr(sg, "EXPECTED_PROMPT_ITEMS", ["x", "x", "x"])
        assert sg.duplicate_prompt_items() == ["x"]

    def test_detects_multiple_duplicates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import huh_twin.skill_graph as sg
        monkeypatch.setattr(sg, "EXPECTED_PROMPT_ITEMS", ["a", "b", "a", "b", "c"])
        result = sg.duplicate_prompt_items()
        assert set(result) == {"a", "b"}

    def test_unique_item_not_reported_as_duplicate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import huh_twin.skill_graph as sg
        monkeypatch.setattr(sg, "EXPECTED_PROMPT_ITEMS", ["unique", "dup", "dup"])
        assert "unique" not in sg.duplicate_prompt_items()

    def test_empty_list_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import huh_twin.skill_graph as sg
        monkeypatch.setattr(sg, "EXPECTED_PROMPT_ITEMS", [])
        assert sg.duplicate_prompt_items() == []
