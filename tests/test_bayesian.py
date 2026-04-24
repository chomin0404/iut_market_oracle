"""Tests for src/bayesian/updater.py and src/bayesian/summary.py."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from bayesian.summary import load_priors_yaml, posterior_to_dict, save_summary
from bayesian.updater import update
from schemas import Evidence, EvidenceKind, PriorSpec

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BETA_PRIOR = PriorSpec(distribution="beta", params={"alpha": 2.0, "beta": 18.0})
NORMAL_PRIOR = PriorSpec(distribution="normal", params={"mu": 0.05, "sigma": 0.02})

CONFIGS_YAML = Path(__file__).parent.parent / "configs" / "priors.yaml"


def _evidence(value: float, weight: float = 1.0) -> Evidence:
    return Evidence(source="test", kind=EvidenceKind.OBSERVATION, value=value, weight=weight)


# ---------------------------------------------------------------------------
# Beta-Binomial conjugate
# ---------------------------------------------------------------------------


class TestBetaUpdate:
    def test_no_evidence_returns_prior_stats(self):
        post = update(BETA_PRIOR, [])
        # Prior Beta(2,18): mean = 2/20 = 0.10
        assert post.mean == pytest.approx(0.10, abs=1e-9)
        assert post.n_evidence == 0

    def test_single_observation_shifts_mean(self):
        """Adding a high-proportion observation should raise the posterior mean."""
        prior_mean = update(BETA_PRIOR, []).mean
        post = update(BETA_PRIOR, [_evidence(0.5, weight=10.0)])
        assert post.mean > prior_mean

    def test_low_proportion_evidence_lowers_mean(self):
        """Adding evidence below the prior mean should push the mean down."""
        post_base = update(BETA_PRIOR, []).mean
        post = update(BETA_PRIOR, [_evidence(0.0, weight=5.0)])
        assert post.mean < post_base

    def test_mean_in_unit_interval(self):
        for v in [0.0, 0.1, 0.5, 0.9, 1.0]:
            post = update(BETA_PRIOR, [_evidence(v)])
            assert 0.0 < post.mean < 1.0

    def test_variance_decreases_with_more_evidence(self):
        post1 = update(BETA_PRIOR, [_evidence(0.2)])
        post10 = update(BETA_PRIOR, [_evidence(0.2)] * 10)
        assert post10.variance < post1.variance

    def test_credible_interval_ordered(self):
        post = update(BETA_PRIOR, [_evidence(0.15, weight=5.0)])
        lo, hi = post.credible_interval_95
        assert lo < hi

    def test_credible_interval_contains_mean(self):
        post = update(BETA_PRIOR, [_evidence(0.15, weight=5.0)])
        lo, hi = post.credible_interval_95
        assert lo <= post.mean <= hi

    def test_weight_accumulates_correctly(self):
        """One evidence with weight=5 == five evidences with weight=1 (same value)."""
        post_bulk = update(BETA_PRIOR, [_evidence(0.3, weight=5.0)])
        post_five = update(BETA_PRIOR, [_evidence(0.3)] * 5)
        assert post_bulk.mean == pytest.approx(post_five.mean, abs=1e-9)
        assert post_bulk.variance == pytest.approx(post_five.variance, abs=1e-9)

    def test_invalid_value_out_of_range(self):
        with pytest.raises(ValueError, match="∈ \\[0, 1\\]"):
            update(BETA_PRIOR, [_evidence(1.5)])

    def test_unsupported_distribution_raises(self):
        with pytest.raises(ValueError, match="Unsupported distribution"):
            update(
                PriorSpec(distribution="dirichlet", params={"alpha": 1.0}),
                [],
            )


# ---------------------------------------------------------------------------
# Normal-Normal conjugate
# ---------------------------------------------------------------------------


class TestNormalUpdate:
    def test_no_evidence_returns_prior_stats(self):
        post = update(NORMAL_PRIOR, [])
        assert post.mean == pytest.approx(0.05, abs=1e-9)
        assert post.variance == pytest.approx(0.02**2, abs=1e-12)
        assert post.n_evidence == 0

    def test_single_observation_moves_mean(self):
        """Observation above prior mean should pull the posterior mean up."""
        post = update(NORMAL_PRIOR, [_evidence(0.10, weight=25.0)])  # τ = 25
        assert post.mean > 0.05

    def test_high_precision_evidence_dominates(self):
        """Very precise observation (high weight) should dominate the prior."""
        post = update(NORMAL_PRIOR, [_evidence(0.20, weight=1e6)])
        assert post.mean == pytest.approx(0.20, abs=1e-3)

    def test_variance_shrinks_with_evidence(self):
        post1 = update(NORMAL_PRIOR, [_evidence(0.06)])
        post10 = update(NORMAL_PRIOR, [_evidence(0.06)] * 10)
        assert post10.variance < post1.variance

    def test_credible_interval_is_symmetric_at_prior(self):
        """With no evidence, the 95% CI should be symmetric around the prior mean."""
        post = update(NORMAL_PRIOR, [])
        lo, hi = post.credible_interval_95
        mid = (lo + hi) / 2
        assert mid == pytest.approx(post.mean, abs=1e-9)

    def test_credible_interval_ordered_and_contains_mean(self):
        post = update(NORMAL_PRIOR, [_evidence(0.07, weight=100.0)])
        lo, hi = post.credible_interval_95
        assert lo < post.mean < hi

    def test_precision_additive_same_as_bulk(self):
        """Three evidence items with τ=10 each == one item with τ=30 (same value)."""
        v = 0.08
        post_bulk = update(NORMAL_PRIOR, [_evidence(v, weight=30.0)])
        post_three = update(NORMAL_PRIOR, [_evidence(v, weight=10.0)] * 3)
        assert post_bulk.mean == pytest.approx(post_three.mean, abs=1e-9)
        assert post_bulk.variance == pytest.approx(post_three.variance, abs=1e-12)

    def test_invalid_zero_weight_raises(self):
        from pydantic import ValidationError as PydanticValidationError

        with pytest.raises(PydanticValidationError, match="greater than 0"):
            # weight=0 is blocked by Evidence schema (gt=0) before reaching updater.
            _evidence(0.05, weight=0.0)


# ---------------------------------------------------------------------------
# summary utilities
# ---------------------------------------------------------------------------


class TestPosteriorToDict:
    def test_keys_present(self):
        post = update(BETA_PRIOR, [_evidence(0.15)])
        d = posterior_to_dict(post, prior=BETA_PRIOR, label="test")
        assert set(d.keys()) >= {
            "label",
            "mean",
            "variance",
            "std",
            "credible_interval_95",
            "n_evidence",
            "updated_at",
            "prior",
        }

    def test_std_consistent_with_variance(self):
        post = update(NORMAL_PRIOR, [_evidence(0.06)])
        d = posterior_to_dict(post)
        assert d["std"] == pytest.approx(math.sqrt(d["variance"]), rel=1e-6)

    def test_ci_keys(self):
        post = update(BETA_PRIOR, [_evidence(0.2)])
        d = posterior_to_dict(post)
        assert "lo" in d["credible_interval_95"]
        assert "hi" in d["credible_interval_95"]


class TestSaveSummary:
    def test_save_yaml(self, tmp_path: Path):
        post = update(BETA_PRIOR, [_evidence(0.15, weight=5.0)])
        out = save_summary(post, tmp_path / "out.yaml", prior=BETA_PRIOR, label="test")
        assert out.exists()
        import yaml

        data = yaml.safe_load(out.read_text())
        assert data["label"] == "test"
        assert "mean" in data

    def test_save_json(self, tmp_path: Path):
        post = update(NORMAL_PRIOR, [_evidence(0.07)])
        out = save_summary(post, tmp_path / "out.json", fmt="json")
        assert out.exists()
        import json

        data = json.loads(out.read_text())
        assert "mean" in data

    def test_creates_parent_dirs(self, tmp_path: Path):
        post = update(BETA_PRIOR, [])
        out = save_summary(post, tmp_path / "deep" / "nested" / "result.yaml")
        assert out.exists()


# ---------------------------------------------------------------------------
# Load priors from configs/priors.yaml
# ---------------------------------------------------------------------------


class TestLoadPriorsYaml:
    @pytest.mark.skipif(not CONFIGS_YAML.exists(), reason="configs/priors.yaml not found")
    def test_loads_all_entries(self):
        priors = load_priors_yaml(CONFIGS_YAML)
        assert len(priors) > 0

    @pytest.mark.skipif(not CONFIGS_YAML.exists(), reason="configs/priors.yaml not found")
    def test_beta_prior_valid(self):
        priors = load_priors_yaml(CONFIGS_YAML)
        p = priors["default_probability"]
        assert p.distribution == "beta"
        assert p.params["alpha"] > 0
        assert p.params["beta"] > 0

    @pytest.mark.skipif(not CONFIGS_YAML.exists(), reason="configs/priors.yaml not found")
    def test_normal_prior_valid(self):
        priors = load_priors_yaml(CONFIGS_YAML)
        p = priors["revenue_growth"]
        assert p.distribution == "normal"
        assert p.params["sigma"] > 0

    @pytest.mark.skipif(not CONFIGS_YAML.exists(), reason="configs/priors.yaml not found")
    def test_update_with_loaded_prior(self):
        """End-to-end: load prior → update with evidence → get PosteriorSummary."""
        priors = load_priors_yaml(CONFIGS_YAML)
        post = update(priors["default_probability"], [_evidence(0.05, weight=20.0)])
        assert 0.0 < post.mean < 1.0
        assert post.n_evidence == 1
