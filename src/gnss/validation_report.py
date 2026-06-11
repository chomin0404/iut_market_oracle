"""Monte Carlo validation report generator for GNSS spoofing detection.

Runs the Fisher-combined MC validator (N=1,000,000 by default) over three
spoofing scenarios and saves a multi-page PDF report:

  Page 1: ROC curves (P_D vs P_fa) for all three scenarios
  Page 2: DET curves (P_miss vs P_fa) for all three scenarios
  Page 3: Summary table (AUC_ROC, AUC_DET, P_miss @ target, target met?)

Usage (CLI)::

    uv run python -m src.gnss.validation_report

Or from Python::

    from gnss.validation_report import generate_validation_report
    generate_validation_report(output_path="output/validation_report.pdf")

The report is saved to ``output/validation_report.pdf`` by default.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for PDF generation

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

if TYPE_CHECKING:
    from matplotlib.axes import Axes

from gnss.mc_validation import (
    MCValidationConfig,
    MCValidationResult,
    SpoofingScenario,
    run_mc_validation,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_PATH: str = "output/validation_report.pdf"
DEFAULT_N_TRIALS: int = 1_000_000

_SCENARIO_COLORS: dict[str, str] = {
    SpoofingScenario.SIMPLISTIC.value: "#1f77b4",  # blue
    SpoofingScenario.MEACONING.value: "#ff7f0e",  # orange
    SpoofingScenario.SOPHISTICATED.value: "#2ca02c",  # green
}

_SCENARIO_LABELS: dict[str, str] = {
    SpoofingScenario.SIMPLISTIC.value: "SIMPLISTIC",
    SpoofingScenario.MEACONING.value: "MEACONING",
    SpoofingScenario.SOPHISTICATED.value: "SOPHISTICATED",
}


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def _plot_roc(ax: Axes, result: MCValidationResult) -> None:
    """Plot ROC curves (P_D = 1 − P_miss vs P_fa) on ``ax``."""
    for name, sr in result.scenarios().items():
        det = sr.det
        pd = 1.0 - det.p_miss  # detection probability
        ax.plot(
            det.p_fa,
            pd,
            color=_SCENARIO_COLORS[name],
            label=f"{_SCENARIO_LABELS[name]} (AUC={det.auc_roc:.4f})",
            linewidth=1.5,
        )

    # Operating point line
    ax.axvline(
        result.config.p_fa_target,
        color="red",
        linestyle="--",
        linewidth=0.8,
        label=f"P_fa target = {result.config.p_fa_target:.0e}",
    )
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.5, label="Random")

    ax.set_xlabel("False Alarm Rate (P_fa)", fontsize=11)
    ax.set_ylabel("Detection Rate (P_D)", fontsize=11)
    ax.set_title("ROC Curve — GNSS Spoofing Detection", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)


def _plot_det(ax: Axes, result: MCValidationResult) -> None:
    """Plot DET curves (P_miss vs P_fa, log-log) on ``ax``."""
    for name, sr in result.scenarios().items():
        det = sr.det
        # Replace zeros to allow log scale
        p_fa = np.maximum(det.p_fa, 1e-7)
        p_miss = np.maximum(det.p_miss, 1e-7)
        ax.plot(
            p_fa,
            p_miss,
            color=_SCENARIO_COLORS[name],
            label=f"{_SCENARIO_LABELS[name]} (AUC={det.auc_det:.4f})",
            linewidth=1.5,
        )

    ax.axvline(
        result.config.p_fa_target,
        color="red",
        linestyle="--",
        linewidth=0.8,
        label=f"P_fa target = {result.config.p_fa_target:.0e}",
    )
    ax.axhline(
        result.config.p_miss_target,
        color="purple",
        linestyle="--",
        linewidth=0.8,
        label=f"P_miss target = {result.config.p_miss_target:.0e}",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("False Alarm Rate (P_fa)", fontsize=11)
    ax.set_ylabel("Miss Rate (P_miss = 1 − P_D)", fontsize=11)
    ax.set_title("DET Curve — GNSS Spoofing Detection (log-log)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)


def _plot_summary_table(ax: Axes, result: MCValidationResult) -> None:
    """Render a text summary table on ``ax``."""
    ax.axis("off")

    headers = [
        "Scenario",
        "AUC_ROC",
        "AUC_DET",
        f"P_miss@P_fa={result.config.p_fa_target:.0e}",
        f"P_fa@P_miss={result.config.p_miss_target:.0e}",
        "Target met?",
    ]
    rows: list[list[str]] = []
    for name, sr in result.scenarios().items():
        det = sr.det
        rows.append(
            [
                _SCENARIO_LABELS[name],
                f"{det.auc_roc:.5f}",
                f"{det.auc_det:.5f}",
                f"{det.p_miss_at_target_fa:.4f}",
                f"{det.p_fa_at_target_miss:.2e}",
                "YES" if sr.target_met else "NO",
            ]
        )

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)

    # Colour header row
    for j in range(len(headers)):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Colour target_met column
    for i, (_, sr) in enumerate(result.scenarios().items(), start=1):
        cell = table[i, len(headers) - 1]
        cell.set_facecolor("#C6EFCE" if sr.target_met else "#FFC7CE")

    ax.set_title(
        f"Validation Summary  (N = {result.config.n_trials:,} trials, "
        f"seed = {result.config.random_seed})",
        fontsize=12,
        pad=20,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def generate_validation_report(
    output_path: str | os.PathLike[str] = DEFAULT_OUTPUT_PATH,
    n_trials: int = DEFAULT_N_TRIALS,
    config: MCValidationConfig | None = None,
) -> Path:
    """Run MC validation and save a PDF report.

    Args:
        output_path: Destination PDF file path.
        n_trials:    Number of Monte Carlo trials per scenario (default 1,000,000).
        config:      Override the full :class:`MCValidationConfig`.  When supplied,
                     ``n_trials`` is ignored.

    Returns:
        Resolved :class:`Path` of the saved PDF.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if config is None:
        config = MCValidationConfig(n_trials=n_trials)

    result = run_mc_validation(config)

    with PdfPages(out) as pdf:
        # Page 1: ROC curves
        fig, ax = plt.subplots(figsize=(8, 6))
        _plot_roc(ax, result)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: DET curves
        fig, ax = plt.subplots(figsize=(8, 6))
        _plot_det(ax, result)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 3: Summary table
        fig, ax = plt.subplots(figsize=(10, 4))
        _plot_summary_table(ax, result)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # PDF metadata
        meta = pdf.infodict()
        meta["Title"] = "GNSS Spoofing Detection — MC Validation Report"
        meta["Subject"] = (
            f"Fisher-combined score: N={config.n_trials}, "
            f"scenarios={list(_SCENARIO_LABELS.values())}"
        )
        meta["Keywords"] = "GNSS, spoofing, RAIM, Monte Carlo, DET, ROC"

    return out.resolve()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate GNSS MC validation PDF report.")
    parser.add_argument(
        "--out",
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output PDF path (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=DEFAULT_N_TRIALS,
        help=f"MC trials per scenario (default: {DEFAULT_N_TRIALS:,})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    cfg = MCValidationConfig(n_trials=args.n_trials, random_seed=args.seed)
    saved = generate_validation_report(output_path=args.out, config=cfg)
    print(f"Report saved to: {saved}")
