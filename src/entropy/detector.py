"""Entropy-based regime-change detection for the Entropy Layer (T1000).

Alert conditions
----------------
KL_THRESHOLD:
    KL divergence KL(posterior_t || prior) exceeds kl_threshold.
    Fired once per step when the condition holds.

ENTROPY_GRADIENT:
    Absolute rolling entropy rate |rate_t| exceeds
    entropy_gradient_threshold.
    Fired once per step when the condition holds.

Report pipeline integration
----------------------------
``run_detection`` runs both monitors end-to-end and returns an
``EntropyReport`` ready for serialisation.  ``save_entropy_report``
writes the report JSON to ``reports/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from entropy.monitor import compute_entropy, compute_kl, entropy_rate
from schemas import AlertType, EntropyAlert, EntropyReport, PosteriorSummary, PriorSpec

# Default config path (relative to repo root).
_DEFAULT_CONFIG = Path("configs/entropy_thresholds.yaml")


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def _load_thresholds(config_path: Path = _DEFAULT_CONFIG) -> dict:
    """Load entropy threshold configuration from YAML."""
    with config_path.open() as fh:
        cfg = yaml.safe_load(fh)
    return cfg


# ---------------------------------------------------------------------------
# Alert generators
# ---------------------------------------------------------------------------


def _kl_alerts(
    kl_series: list[float],
    threshold: float,
    experiment_id: str,
) -> list[EntropyAlert]:
    """Fire an EntropyAlert for every step where KL > threshold."""
    alerts: list[EntropyAlert] = []
    for step, kl_val in enumerate(kl_series):
        if kl_val > threshold:
            alerts.append(
                EntropyAlert(
                    experiment_id=experiment_id,
                    triggered_at=step,
                    alert_type=AlertType.KL_THRESHOLD,
                    metric_value=kl_val,
                    threshold=threshold,
                    message=(
                        f"KL divergence {kl_val:.4f} exceeds threshold {threshold:.4f} "
                        f"at step {step}"
                    ),
                )
            )
    return alerts


def _gradient_alerts(
    rate_series: list[float],
    threshold: float,
    experiment_id: str,
    step_offset: int,
) -> list[EntropyAlert]:
    """Fire an EntropyAlert for every rate step where |rate| > threshold.

    Parameters
    ----------
    step_offset:
        The rate_series starts at this step index in the original entropy
        series (accounts for window lag).
    """
    alerts: list[EntropyAlert] = []
    for i, rate_val in enumerate(rate_series):
        if abs(rate_val) > threshold:
            step = step_offset + i
            alerts.append(
                EntropyAlert(
                    experiment_id=experiment_id,
                    triggered_at=step,
                    alert_type=AlertType.ENTROPY_GRADIENT,
                    metric_value=rate_val,
                    threshold=threshold,
                    message=(
                        f"|ΔH| = {abs(rate_val):.4f} exceeds gradient threshold "
                        f"{threshold:.4f} at step {step}"
                    ),
                )
            )
    return alerts


# ---------------------------------------------------------------------------
# Public: run_detection
# ---------------------------------------------------------------------------


def run_detection(
    posteriors: list[PosteriorSummary],
    prior: PriorSpec,
    experiment_id: str,
    config_path: Path = _DEFAULT_CONFIG,
) -> EntropyReport:
    """Run full entropy detection pipeline and return an EntropyReport.

    Steps
    -----
    1. Compute H_t = entropy(posterior_t) for all t.
    2. Compute KL_t = KL(posterior_t || prior) for all t.
    3. Compute rolling entropy rate series.
    4. Collect KL_THRESHOLD and ENTROPY_GRADIENT alerts.
    5. Return validated EntropyReport.

    Parameters
    ----------
    posteriors:
        Ordered list of posterior summaries (one per observation step).
    prior:
        Prior specification (reference distribution for KL).
    experiment_id:
        Experiment identifier, e.g. ``"exp-001"``.
    config_path:
        Path to entropy_thresholds.yaml.

    Returns
    -------
    EntropyReport
    """
    cfg = _load_thresholds(config_path)
    kl_threshold: float = float(cfg["kl_threshold"])
    gradient_threshold: float = float(cfg["entropy_gradient_threshold"])
    window: int = int(cfg["rolling_window"])

    h_series: list[float] = [compute_entropy(p, prior) for p in posteriors]
    kl_series: list[float] = [compute_kl(p, prior) for p in posteriors]
    rate_series: list[float] = entropy_rate(h_series, window)

    # Step offset: rate_series[0] corresponds to entropy_series[window].
    step_offset = window

    alerts: list[EntropyAlert] = []
    alerts.extend(_kl_alerts(kl_series, kl_threshold, experiment_id))
    alerts.extend(
        _gradient_alerts(rate_series, gradient_threshold, experiment_id, step_offset)
    )
    # Sort alerts chronologically.
    alerts.sort(key=lambda a: a.triggered_at)

    return EntropyReport(
        experiment_id=experiment_id,
        entropy_series=h_series,
        kl_series=kl_series,
        entropy_rate_series=rate_series,
        alerts=alerts,
    )


# ---------------------------------------------------------------------------
# Public: save_entropy_report
# ---------------------------------------------------------------------------


def save_entropy_report(
    report: EntropyReport,
    output_dir: Path = Path("reports"),
) -> Path:
    """Serialise an EntropyReport to JSON in output_dir.

    File name: ``entropy_report_{experiment_id}.json``

    Parameters
    ----------
    report:
        Validated EntropyReport to persist.
    output_dir:
        Directory under which the file is written (created if absent).

    Returns
    -------
    Path
        Absolute path of the written file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"entropy_report_{report.experiment_id}.json"
    out_path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return out_path.resolve()
