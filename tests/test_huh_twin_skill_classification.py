import pytest

from huh_twin.skill_classification import (
    BASIS_COUNTS,
    CLASSIFICATION_DUPLICATES,
    EXPECTED_PROMPT_ITEMS,
    EXTRA_IN_CLASSIFICATION,
    GROUPED_BY_BASIS,
    MISSING_FROM_CLASSIFICATION,
    PROMPT_DUPLICATES,
    SKILL_CLASSIFICATION,
    SkillBasis,
    basis_counts,
    classification_dict,
    get_by_name,
    get_unique,
    grouped_by_basis,
    validate,
)


def test_every_classified_item_has_valid_basis() -> None:
    assert all(isinstance(item.basis, SkillBasis) for item in SKILL_CLASSIFICATION)


def test_expected_prompt_items_count_matches_original_list() -> None:
    assert len(EXPECTED_PROMPT_ITEMS) == 63


def test_prompt_duplicates_snapshot() -> None:
    assert PROMPT_DUPLICATES == ("ブライト・ウィグナー分布",)


def test_classification_duplicates_snapshot() -> None:
    assert CLASSIFICATION_DUPLICATES == ("ブライト・ウィグナー分布",)


def test_no_prompt_item_missing_from_classification() -> None:
    assert MISSING_FROM_CLASSIFICATION == ()


def test_no_extra_classification_item_outside_prompt() -> None:
    assert EXTRA_IN_CLASSIFICATION == ()


def test_validation_summary_is_clean_except_known_duplicate() -> None:
    report = validate()
    assert report["prompt_duplicates"] == ("ブライト・ウィグナー分布",)
    assert report["classification_duplicates"] == ("ブライト・ウィグナー分布",)
    assert report["missing_from_classification"] == ()
    assert report["extra_in_classification"] == ()


def test_get_by_name_returns_all_matches_for_duplicate_name() -> None:
    matches = get_by_name("ブライト・ウィグナー分布")
    assert len(matches) == 2
    assert all(item.basis == SkillBasis.INVERSE_PROBLEMS_AND_SIMULATION for item in matches)


def test_get_unique_returns_single_match() -> None:
    item = get_unique("DCF法")
    assert item.basis == SkillBasis.VALUATION_AND_CAPITAL_ALLOCATION


def test_get_unique_raises_for_duplicate_name() -> None:
    with pytest.raises(ValueError):
        get_unique("ブライト・ウィグナー分布")


def test_get_unique_raises_for_unknown_name() -> None:
    with pytest.raises(KeyError):
        get_unique("unknown")


def test_classification_dict_contains_known_examples() -> None:
    mapping = classification_dict()
    assert mapping["ベイズ統計"]["basis"] == "probabilistic_inference"
    assert mapping["DCF法"]["basis"] == "valuation_and_capital_allocation"
    assert mapping["ラドン変換（Radon Transform）"]["basis"] == "inverse_problems_and_simulation"
    assert mapping["実験計画法（Optimal Design of Experiments）"]["basis"] == "optimal_experiment_design"
    assert mapping["Claudecodeの活用"]["basis"] == "strategy_and_implementation"


def test_grouped_by_basis_covers_all_items() -> None:
    grouped = grouped_by_basis()
    total = sum(len(items) for items in grouped.values())
    assert total == len(SKILL_CLASSIFICATION)
    assert set(grouped.keys()) == set(SkillBasis)


def test_precomputed_grouped_by_basis_matches_accessor() -> None:
    assert grouped_by_basis() == GROUPED_BY_BASIS


def test_precomputed_basis_counts_match_accessor() -> None:
    assert basis_counts() == BASIS_COUNTS


def test_each_basis_has_at_least_one_item() -> None:
    for basis in SkillBasis:
        assert len(GROUPED_BY_BASIS[basis]) > 0


def test_basis_counts_match_expected_snapshot() -> None:
    assert BASIS_COUNTS[SkillBasis.PROBABILISTIC_INFERENCE] == 13
    assert BASIS_COUNTS[SkillBasis.VALUATION_AND_CAPITAL_ALLOCATION] == 13
    assert BASIS_COUNTS[SkillBasis.INVERSE_PROBLEMS_AND_SIMULATION] == 21
    assert BASIS_COUNTS[SkillBasis.OPTIMAL_EXPERIMENT_DESIGN] == 6
    assert BASIS_COUNTS[SkillBasis.STRATEGY_AND_IMPLEMENTATION] == 10
