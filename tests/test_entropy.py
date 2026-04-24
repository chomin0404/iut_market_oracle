"""Tests for the Entropy Layer (T1000).

Coverage
--------
TestEntropyNormal       — entropy_normal edge cases and formula
TestEntropyBeta         — entropy_beta: uniform maximises, point mass limit
TestEntropyDispatch     — compute_entropy: Normal and Beta families
TestKLNormal            — kl_normal: zero when q==p, positive otherwise
TestKLBeta              — kl_beta: zero when q==p, positive otherwise
TestKLDispatch          — compute_kl: Normal and Beta families
TestEntropyRate         — rolling window first-differences
TestEntropyRateEdge     — edge cases: too short series, window larger than diffs
TestAlertGeneration     — KL and gradient alerts fire above threshold
TestRunDetection        — end-to-end run_detection with Normal posteriors
TestSaveEntropyReport   — JSON file is written and deserializable
TestEntropyReportSchema — EntropyReport Pydantic validation
TestEntropyAlertSchema  — EntropyAlert Pydantic validation
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import pytest

from entropy.detector import run_detection, save_entropy_report
from entropy.monitor import (
    compute_entropy,
    compute_kl,
    entropy_beta,
    entropy_normal,
    entropy_rate,
    kl_beta,
    kl_normal,
)
from schemas import (
    AlertType,
    EntropyAlert,
    EntropyReport,
    PosteriorSummary,
    PriorSpec,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EXP_ID = "exp-001"


def _normal_posterior(mean: float = 0.0, variance: float = 1.0) -> PosteriorSummary:
    lo = mean - 2.0 * variance**0.5
    hi = mean + 2.0 * variance**0.5
    return PosteriorSummary(
        mean=mean,
        variance=variance,
        credible_interval_95=(lo, hi),
        n_evidence=10,
    )


def _beta_posterior(alpha: float, beta_: float) -> PosteriorSummary:
    """Construct a PosteriorSummary consistent with Beta(α, β) via MoM."""
    mu = alpha / (alpha + beta_)
    var = alpha * beta_ / ((alpha + beta_) ** 2 * (alpha + beta_ + 1))
    lo = mu - 2.0 * var**0.5
    hi = mu + 2.0 * var**0.5
    return PosteriorSummary(
        mean=mu,
        variance=var,
        credible_interval_95=(lo, hi),
        n_evidence=5,
    )


def _normal_prior(mean: float = 0.0, std: float = 1.0) -> PriorSpec:
    return PriorSpec(distribution="normal", params={"mean": mean, "std": std})


def _beta_prior(alpha: float, beta_: float) -> PriorSpec:
    return PriorSpec(distribution="beta", params={"alpha": alpha, "beta": beta_})


# ---------------------------------------------------------------------------
# TestEntropyNormal
# ---------------------------------------------------------------------------


class TestEntropyNormal:
    def test_unit_variance(self) -> None:
        # H = 0.5 * ln(2πe)
        expected = 0.5 * math.log(2.0 * math.pi * math.e)
        assert math.isclose(entropy_normal(1.0), expected, rel_tol=1e-12)

    def test_larger_variance_gives_higher_entropy(self) -> None:
        assert entropy_normal(2.0) > entropy_normal(1.0)

    def test_smaller_variance_gives_lower_entropy(self) -> None:
        assert entropy_normal(0.1) < entropy_normal(1.0)

    def test_variance_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            entropy_normal(0.0)

    def test_negative_variance_raises(self) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            entropy_normal(-1.0)

    def test_very_small_variance(self) -> None:
        # Should return a large negative number (not raise).
        h = entropy_normal(1e-10)
        assert isinstance(h, float)


# ---------------------------------------------------------------------------
# TestEntropyBeta
# ---------------------------------------------------------------------------


class TestEntropyBeta:
    def test_uniform_has_zero_entropy(self) -> None:
        # Beta(1,1) = Uniform(0,1) → H = 0
        assert math.isclose(entropy_beta(1.0, 1.0), 0.0, abs_tol=1e-12)

    def test_concentrated_beta_has_lower_entropy_than_uniform(self) -> None:
        # More concentrated → lower entropy
        assert entropy_beta(10.0, 10.0) < entropy_beta(1.0, 1.0)

    def test_entropy_is_symmetric_in_params(self) -> None:
        # Beta(2,5) and Beta(5,2) have the same entropy (symmetry of Beta).
        assert math.isclose(
            entropy_beta(2.0, 5.0), entropy_beta(5.0, 2.0), rel_tol=1e-12
        )

    def test_entropy_beta_is_finite(self) -> None:
        assert math.isfinite(entropy_beta(2.0, 3.0))


# ---------------------------------------------------------------------------
# TestEntropyDispatch
# ---------------------------------------------------------------------------


class TestEntropyDispatch:
    def test_normal_family(self) -> None:
        p = _normal_posterior(variance=1.0)
        pr = _normal_prior()
        assert math.isclose(compute_entropy(p, pr), entropy_normal(1.0), rel_tol=1e-12)

    def test_gaussian_alias(self) -> None:
        p = _normal_posterior(variance=2.0)
        pr = PriorSpec(distribution="gaussian", params={"mean": 0.0, "std": 1.0})
        assert math.isclose(compute_entropy(p, pr), entropy_normal(2.0), rel_tol=1e-12)

    def test_beta_family(self) -> None:
        p = _beta_posterior(2.0, 5.0)
        pr = _beta_prior(2.0, 5.0)
        h = compute_entropy(p, pr)
        assert math.isfinite(h)

    def test_unsupported_family_raises(self) -> None:
        p = _normal_posterior()
        pr = PriorSpec(distribution="uniform", params={"low": 0.0, "high": 1.0})
        with pytest.raises(ValueError, match="Unsupported"):
            compute_entropy(p, pr)


# ---------------------------------------------------------------------------
# TestKLNormal
# ---------------------------------------------------------------------------


class TestKLNormal:
    def test_zero_when_identical(self) -> None:
        kl = kl_normal(0.0, 1.0, 0.0, 1.0)
        assert math.isclose(kl, 0.0, abs_tol=1e-12)

    def test_positive_when_different_mean(self) -> None:
        assert kl_normal(1.0, 1.0, 0.0, 1.0) > 0.0

    def test_positive_when_different_variance(self) -> None:
        assert kl_normal(0.0, 0.5, 0.0, 1.0) > 0.0

    def test_asymmetric(self) -> None:
        # KL is not symmetric: KL(q||p) ≠ KL(p||q) in general
        kl_qp = kl_normal(1.0, 1.0, 0.0, 2.0)
        kl_pq = kl_normal(0.0, 2.0, 1.0, 1.0)
        assert not math.isclose(kl_qp, kl_pq, rel_tol=1e-3)

    def test_zero_variance_raises(self) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            kl_normal(0.0, 0.0, 0.0, 1.0)

    def test_zero_prior_variance_raises(self) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            kl_normal(0.0, 1.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# TestKLBeta
# ---------------------------------------------------------------------------


class TestKLBeta:
    def test_zero_when_identical(self) -> None:
        kl = kl_beta(2.0, 5.0, 2.0, 5.0)
        assert math.isclose(kl, 0.0, abs_tol=1e-10)

    def test_positive_when_different(self) -> None:
        assert kl_beta(3.0, 5.0, 2.0, 5.0) > 0.0

    def test_asymmetric(self) -> None:
        kl_qp = kl_beta(3.0, 5.0, 2.0, 8.0)
        kl_pq = kl_beta(2.0, 8.0, 3.0, 5.0)
        assert not math.isclose(kl_qp, kl_pq, rel_tol=1e-3)


# ---------------------------------------------------------------------------
# TestKLDispatch
# ---------------------------------------------------------------------------


class TestKLDispatch:
    def test_normal_dispatch_zero(self) -> None:
        p = _normal_posterior(mean=0.0, variance=1.0)
        pr = _normal_prior(mean=0.0, std=1.0)
        assert math.isclose(compute_kl(p, pr), 0.0, abs_tol=1e-10)

    def test_normal_dispatch_positive(self) -> None:
        p = _normal_posterior(mean=2.0, variance=1.0)
        pr = _normal_prior(mean=0.0, std=1.0)
        assert compute_kl(p, pr) > 0.0

    def test_normal_variance_key(self) -> None:
        p = _normal_posterior(mean=0.0, variance=1.0)
        pr = PriorSpec(distribution="normal", params={"mean": 0.0, "variance": 1.0})
        assert math.isclose(compute_kl(p, pr), 0.0, abs_tol=1e-10)

    def test_normal_missing_variance_key_raises(self) -> None:
        p = _normal_posterior()
        pr = PriorSpec(distribution="normal", params={"mean": 0.0})
        with pytest.raises(KeyError):
            compute_kl(p, pr)

    def test_beta_dispatch_zero(self) -> None:
        alpha, beta_ = 3.0, 7.0
        p = _beta_posterior(alpha, beta_)
        pr = _beta_prior(alpha, beta_)
        kl = compute_kl(p, pr)
        # MoM round-trip may introduce small error → use loose tolerance.
        assert abs(kl) < 1e-4

    def test_beta_dispatch_positive(self) -> None:
        p = _beta_posterior(5.0, 5.0)
        pr = _beta_prior(2.0, 8.0)
        assert compute_kl(p, pr) > 0.0

    def test_unsupported_raises(self) -> None:
        p = _normal_posterior()
        pr = PriorSpec(distribution="uniform", params={"low": 0.0, "high": 1.0})
        with pytest.raises(ValueError, match="Unsupported"):
            compute_kl(p, pr)


# ---------------------------------------------------------------------------
# TestEntropyRate
# ---------------------------------------------------------------------------


class TestEntropyRate:
    def test_constant_series_gives_zero_rate(self) -> None:
        series = [1.0] * 10
        rates = entropy_rate(series, window=3)
        assert all(math.isclose(r, 0.0, abs_tol=1e-12) for r in rates)

    def test_linearly_increasing_series(self) -> None:
        # ΔH = 1.0 at each step; rolling mean should be 1.0
        series = list(range(10))  # 0,1,...,9
        rates = entropy_rate(series, window=3)
        assert all(math.isclose(r, 1.0, abs_tol=1e-12) for r in rates)

    def test_output_length(self) -> None:
        n = 10
        window = 4
        rates = entropy_rate(list(range(n)), window=window)
        # convolve valid mode: len(diffs) - window + 1 = (n-1) - window + 1
        expected_len = (n - 1) - window + 1
        assert len(rates) == expected_len

    def test_single_element_returns_empty(self) -> None:
        assert entropy_rate([1.0], window=3) == []

    def test_empty_returns_empty(self) -> None:
        assert entropy_rate([], window=3) == []

    def test_window_one(self) -> None:
        # Window=1: rolling mean of diffs == raw diffs
        series = [0.0, 2.0, 1.0, 3.0]
        rates = entropy_rate(series, window=1)
        diffs = [2.0, -1.0, 2.0]
        assert len(rates) == len(diffs)
        for r, d in zip(rates, diffs):
            assert math.isclose(r, d, abs_tol=1e-12)

    def test_window_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="window must be >= 1"):
            entropy_rate([1.0, 2.0], window=0)


# ---------------------------------------------------------------------------
# TestEntropyRateEdge
# ---------------------------------------------------------------------------


class TestEntropyRateEdge:
    def test_window_larger_than_diffs_returns_raw_diffs(self) -> None:
        # 3 values → 2 diffs; window=5 > 2 → returns raw diffs
        series = [1.0, 3.0, 2.0]
        rates = entropy_rate(series, window=5)
        assert len(rates) == 2
        assert math.isclose(rates[0], 2.0, abs_tol=1e-12)
        assert math.isclose(rates[1], -1.0, abs_tol=1e-12)

    def test_two_elements_window_one(self) -> None:
        rates = entropy_rate([0.0, 1.0], window=1)
        assert len(rates) == 1
        assert math.isclose(rates[0], 1.0, abs_tol=1e-12)


# ---------------------------------------------------------------------------
# TestAlertGeneration (unit: detector internals via run_detection)
# ---------------------------------------------------------------------------


class TestAlertGeneration:
    def _build_posteriors_with_shift(self, n: int = 10) -> list[PosteriorSummary]:
        """Normal posteriors; last three have a large mean shift → high KL."""
        posteriors = []
        for i in range(n):
            mean = 5.0 if i >= n - 3 else 0.0
            posteriors.append(_normal_posterior(mean=mean, variance=0.5))
        return posteriors

    def test_kl_alerts_fired_for_shifted_posteriors(self, tmp_path: Path) -> None:
        cfg = tmp_path / "thresholds.yaml"
        cfg.write_text(
            "kl_threshold: 1.0\nentropy_gradient_threshold: 999.0\nrolling_window: 1\n"
        )
        posteriors = self._build_posteriors_with_shift(10)
        prior = _normal_prior(mean=0.0, std=1.0)
        report = run_detection(posteriors, prior, _EXP_ID, config_path=cfg)
        kl_alerts = [a for a in report.alerts if a.alert_type == AlertType.KL_THRESHOLD]
        assert len(kl_alerts) >= 3

    def test_no_kl_alerts_when_threshold_high(self, tmp_path: Path) -> None:
        cfg = tmp_path / "thresholds.yaml"
        cfg.write_text(
            "kl_threshold: 999.0\nentropy_gradient_threshold: 999.0\nrolling_window: 1\n"
        )
        posteriors = [_normal_posterior() for _ in range(5)]
        prior = _normal_prior()
        report = run_detection(posteriors, prior, _EXP_ID, config_path=cfg)
        assert len(report.alerts) == 0

    def test_gradient_alerts_fire(self, tmp_path: Path) -> None:
        cfg = tmp_path / "thresholds.yaml"
        # Very low gradient threshold → should fire for a changing entropy series.
        cfg.write_text(
            "kl_threshold: 999.0\nentropy_gradient_threshold: 0.001\nrolling_window: 1\n"
        )
        # Increasing variance → increasing entropy.
        posteriors = [_normal_posterior(variance=float(i + 1)) for i in range(8)]
        prior = _normal_prior()
        report = run_detection(posteriors, prior, _EXP_ID, config_path=cfg)
        grad_alerts = [
            a for a in report.alerts if a.alert_type == AlertType.ENTROPY_GRADIENT
        ]
        assert len(grad_alerts) > 0

    def test_alerts_sorted_chronologically(self, tmp_path: Path) -> None:
        cfg = tmp_path / "thresholds.yaml"
        cfg.write_text(
            "kl_threshold: 0.0\nentropy_gradient_threshold: 0.0\nrolling_window: 1\n"
        )
        posteriors = [_normal_posterior(mean=float(i), variance=float(i + 1)) for i in range(6)]
        prior = _normal_prior()
        report = run_detection(posteriors, prior, _EXP_ID, config_path=cfg)
        steps = [a.triggered_at for a in report.alerts]
        assert steps == sorted(steps)


# ---------------------------------------------------------------------------
# TestRunDetection
# ---------------------------------------------------------------------------


class TestRunDetection:
    def test_series_lengths_match_posteriors(self, tmp_path: Path) -> None:
        cfg = tmp_path / "thresholds.yaml"
        cfg.write_text(
            "kl_threshold: 9999.0\nentropy_gradient_threshold: 9999.0\nrolling_window: 3\n"
        )
        n = 8
        posteriors = [_normal_posterior(variance=float(i + 1)) for i in range(n)]
        prior = _normal_prior()
        report = run_detection(posteriors, prior, _EXP_ID, config_path=cfg)
        assert len(report.entropy_series) == n
        assert len(report.kl_series) == n

    def test_entropy_series_increases_with_variance(self, tmp_path: Path) -> None:
        cfg = tmp_path / "thresholds.yaml"
        cfg.write_text(
            "kl_threshold: 9999.0\nentropy_gradient_threshold: 9999.0\nrolling_window: 1\n"
        )
        posteriors = [_normal_posterior(variance=float(i + 1)) for i in range(5)]
        prior = _normal_prior()
        report = run_detection(posteriors, prior, _EXP_ID, config_path=cfg)
        h = report.entropy_series
        assert all(h[i] < h[i + 1] for i in range(len(h) - 1))

    def test_kl_zero_when_posterior_equals_prior(self, tmp_path: Path) -> None:
        cfg = tmp_path / "thresholds.yaml"
        cfg.write_text(
            "kl_threshold: 9999.0\nentropy_gradient_threshold: 9999.0\nrolling_window: 1\n"
        )
        # Posterior == prior (mean=0, var=1)
        posteriors = [_normal_posterior(mean=0.0, variance=1.0) for _ in range(4)]
        prior = _normal_prior(mean=0.0, std=1.0)
        report = run_detection(posteriors, prior, _EXP_ID, config_path=cfg)
        assert all(math.isclose(kl, 0.0, abs_tol=1e-10) for kl in report.kl_series)

    def test_report_is_valid_entropy_report(self, tmp_path: Path) -> None:
        cfg = tmp_path / "thresholds.yaml"
        cfg.write_text(
            "kl_threshold: 9999.0\nentropy_gradient_threshold: 9999.0\nrolling_window: 2\n"
        )
        posteriors = [_normal_posterior() for _ in range(5)]
        prior = _normal_prior()
        report = run_detection(posteriors, prior, _EXP_ID, config_path=cfg)
        assert isinstance(report, EntropyReport)
        assert report.experiment_id == _EXP_ID


# ---------------------------------------------------------------------------
# TestSaveEntropyReport
# ---------------------------------------------------------------------------


class TestSaveEntropyReport:
    def test_file_is_created(self, tmp_path: Path) -> None:
        report = EntropyReport(
            experiment_id=_EXP_ID,
            entropy_series=[1.0, 2.0],
            kl_series=[0.1, 0.2],
        )
        out = save_entropy_report(report, output_dir=tmp_path)
        assert out.exists()

    def test_file_name_contains_experiment_id(self, tmp_path: Path) -> None:
        report = EntropyReport(
            experiment_id=_EXP_ID,
            entropy_series=[1.0],
            kl_series=[0.0],
        )
        out = save_entropy_report(report, output_dir=tmp_path)
        assert _EXP_ID in out.name

    def test_json_is_deserializable(self, tmp_path: Path) -> None:
        report = EntropyReport(
            experiment_id=_EXP_ID,
            entropy_series=[1.5, 2.0, 1.8],
            kl_series=[0.1, 0.3, 0.2],
            entropy_rate_series=[0.5, -0.2],
        )
        out = save_entropy_report(report, output_dir=tmp_path)
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["experiment_id"] == _EXP_ID
        assert len(data["entropy_series"]) == 3

    def test_output_dir_is_created_if_absent(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            new_dir = Path(td) / "nested" / "reports"
            assert not new_dir.exists()
            report = EntropyReport(
                experiment_id=_EXP_ID,
                entropy_series=[1.0],
                kl_series=[0.0],
            )
            save_entropy_report(report, output_dir=new_dir)
            assert new_dir.exists()


# ---------------------------------------------------------------------------
# TestEntropyReportSchema
# ---------------------------------------------------------------------------


class TestEntropyReportSchema:
    def test_valid_report(self) -> None:
        r = EntropyReport(
            experiment_id=_EXP_ID,
            entropy_series=[1.0, 2.0],
            kl_series=[0.1, 0.2],
        )
        assert r.experiment_id == _EXP_ID

    def test_kl_series_length_mismatch_raises(self) -> None:
        with pytest.raises(Exception):
            EntropyReport(
                experiment_id=_EXP_ID,
                entropy_series=[1.0, 2.0],
                kl_series=[0.1],  # wrong length
            )

    def test_invalid_experiment_id_raises(self) -> None:
        with pytest.raises(Exception):
            EntropyReport(
                experiment_id="bad-id",
                entropy_series=[1.0],
                kl_series=[0.0],
            )

    def test_default_empty_rate_and_alerts(self) -> None:
        r = EntropyReport(
            experiment_id=_EXP_ID,
            entropy_series=[1.0],
            kl_series=[0.0],
        )
        assert r.entropy_rate_series == []
        assert r.alerts == []


# ---------------------------------------------------------------------------
# TestEntropyAlertSchema
# ---------------------------------------------------------------------------


class TestEntropyAlertSchema:
    def test_valid_kl_alert(self) -> None:
        a = EntropyAlert(
            experiment_id=_EXP_ID,
            triggered_at=3,
            alert_type=AlertType.KL_THRESHOLD,
            metric_value=0.8,
            threshold=0.5,
        )
        assert a.alert_type == AlertType.KL_THRESHOLD

    def test_valid_gradient_alert(self) -> None:
        a = EntropyAlert(
            experiment_id=_EXP_ID,
            triggered_at=5,
            alert_type=AlertType.ENTROPY_GRADIENT,
            metric_value=0.15,
            threshold=0.1,
        )
        assert a.alert_type == AlertType.ENTROPY_GRADIENT

    def test_negative_step_raises(self) -> None:
        with pytest.raises(Exception):
            EntropyAlert(
                experiment_id=_EXP_ID,
                triggered_at=-1,
                alert_type=AlertType.KL_THRESHOLD,
                metric_value=1.0,
                threshold=0.5,
            )

    def test_invalid_experiment_id_raises(self) -> None:
        with pytest.raises(Exception):
            EntropyAlert(
                experiment_id="bad",
                triggered_at=0,
                alert_type=AlertType.KL_THRESHOLD,
                metric_value=1.0,
                threshold=0.5,
            )
