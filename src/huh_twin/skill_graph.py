"""Skill classification and basis-coverage analysis for the huh_twin research system.

Each skill is assigned to exactly one SkillBasis.  The module exposes helpers
to detect mismatches between the declared classification list and an expected
prompt-item registry, and to check for duplicates.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SkillBasis(str, Enum):
    PROBABILISTIC_INFERENCE = "probabilistic_inference"
    VALUATION_AND_CAPITAL_ALLOCATION = "valuation_and_capital_allocation"
    INVERSE_PROBLEMS_AND_SIMULATION = "inverse_problems_and_simulation"
    OPTIMAL_EXPERIMENT_DESIGN = "optimal_experiment_design"
    STRATEGY_AND_IMPLEMENTATION = "strategy_and_implementation"


@dataclass(frozen=True, slots=True)
class ClassifiedSkill:
    name: str
    basis: SkillBasis
    note: str


# ---------------------------------------------------------------------------
# Canonical skill registry
# ---------------------------------------------------------------------------

SKILL_CLASSIFICATION: list[ClassifiedSkill] = [
    # --- Probabilistic inference ---
    ClassifiedSkill(
        name="bayesian_update",
        basis=SkillBasis.PROBABILISTIC_INFERENCE,
        note="Normal-Normal conjugate update for prior revision.",
    ),
    ClassifiedSkill(
        name="regime_posterior",
        basis=SkillBasis.PROBABILISTIC_INFERENCE,
        note="Discrete Bayesian update over market regimes.",
    ),
    ClassifiedSkill(
        name="prior_config_loading",
        basis=SkillBasis.PROBABILISTIC_INFERENCE,
        note="YAML-driven prior and observation-std configuration.",
    ),
    # --- Valuation and capital allocation ---
    ClassifiedSkill(
        name="dcf_valuation",
        basis=SkillBasis.VALUATION_AND_CAPITAL_ALLOCATION,
        note="Gordon Growth DCF with explicit FCF projection.",
    ),
    ClassifiedSkill(
        name="terminal_value_estimation",
        basis=SkillBasis.VALUATION_AND_CAPITAL_ALLOCATION,
        note="Gordon Growth terminal value under stable-growth assumption.",
    ),
    ClassifiedSkill(
        name="fcf_projection",
        basis=SkillBasis.VALUATION_AND_CAPITAL_ALLOCATION,
        note="Geometric FCF growth over the explicit forecast horizon.",
    ),
    # --- Inverse problems and simulation ---
    ClassifiedSkill(
        name="reverse_dcf",
        basis=SkillBasis.INVERSE_PROBLEMS_AND_SIMULATION,
        note="Bisection-based implied growth rate recovery from a target EV.",
    ),
    ClassifiedSkill(
        name="discount_cash_flows",
        basis=SkillBasis.INVERSE_PROBLEMS_AND_SIMULATION,
        note="Present-value transformation of a cash flow series.",
    ),
    # --- Optimal experiment design ---
    ClassifiedSkill(
        name="one_way_sensitivity",
        basis=SkillBasis.OPTIMAL_EXPERIMENT_DESIGN,
        note="Single-parameter sweep around the base case.",
    ),
    ClassifiedSkill(
        name="two_way_sensitivity",
        basis=SkillBasis.OPTIMAL_EXPERIMENT_DESIGN,
        note="Cartesian grid sweep over two parameters simultaneously.",
    ),
    ClassifiedSkill(
        name="valuation_config_runner",
        basis=SkillBasis.OPTIMAL_EXPERIMENT_DESIGN,
        note="YAML-driven batch execution of sensitivity sweeps.",
    ),
    # --- Strategy and implementation ---
    ClassifiedSkill(
        name="markdown_report_generation",
        basis=SkillBasis.STRATEGY_AND_IMPLEMENTATION,
        note="Markdown report assembly from posterior and valuation outputs.",
    ),
    ClassifiedSkill(
        name="report_file_writing",
        basis=SkillBasis.STRATEGY_AND_IMPLEMENTATION,
        note="Persist a markdown report to disk with parent directory creation.",
    ),
]

# Expected names sourced from the research prompt / spec.
# Must stay in sync with SKILL_CLASSIFICATION.
EXPECTED_PROMPT_ITEMS: list[str] = [item.name for item in SKILL_CLASSIFICATION]


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------


def grouped_by_basis() -> dict[SkillBasis, list[ClassifiedSkill]]:
    """Return SKILL_CLASSIFICATION partitioned by SkillBasis.

    Every SkillBasis key is always present (possibly with an empty list).
    """
    grouped: dict[SkillBasis, list[ClassifiedSkill]] = {basis: [] for basis in SkillBasis}
    for item in SKILL_CLASSIFICATION:
        grouped[item.basis].append(item)
    return grouped


def prompt_items_missing_from_classification() -> list[str]:
    """Names in EXPECTED_PROMPT_ITEMS that have no ClassifiedSkill entry."""
    classified_names = {item.name for item in SKILL_CLASSIFICATION}
    return [name for name in EXPECTED_PROMPT_ITEMS if name not in classified_names]


def classification_items_not_in_prompt() -> list[str]:
    """Names in SKILL_CLASSIFICATION that are absent from EXPECTED_PROMPT_ITEMS."""
    prompt_names = set(EXPECTED_PROMPT_ITEMS)
    return [item.name for item in SKILL_CLASSIFICATION if item.name not in prompt_names]


def duplicate_prompt_items() -> list[str]:
    """Names that appear more than once in EXPECTED_PROMPT_ITEMS (each reported once)."""
    seen: set[str] = set()
    duplicates: list[str] = []
    for name in EXPECTED_PROMPT_ITEMS:
        if name in seen and name not in duplicates:
            duplicates.append(name)
        seen.add(name)
    return duplicates


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    grouped = grouped_by_basis()
    for basis in SkillBasis:
        print(f"[{basis.value}] {len(grouped[basis])}")
        for item in grouped[basis]:
            print(f"  - {item.name}: {item.note}")
        print()

    print("Missing from classification:", prompt_items_missing_from_classification())
    print("Extra in classification:    ", classification_items_not_in_prompt())
    print("Duplicate prompt items:     ", duplicate_prompt_items())
