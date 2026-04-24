"""Report generation pipeline.

Usage
-----
    uv run python -m src.report

    # Custom paths:
    uv run python -m src.report \
        --scenario-dir configs/scenarios \
        --reports-dir reports \
        --experiments-root experiments

Outputs (under reports/<exp-id>/)
----------------------------------
scenario_comparison.png   — bar chart comparing EV across scenarios
sensitivity_tornado.png   — tornado chart of ∂EV/∂p for the base scenario
results_table.csv         — one row per scenario with EV and key sensitivities
summary.md                — human-readable narrative summary

The run is also registered in experiments/registry.md with the output paths.
"""

from __future__ import annotations

# Ensure src/ is first on sys.path so project-level schemas.py is not
# shadowed by a same-named file at the repository root.
import sys
from pathlib import Path as _Path

_src_dir = str(_Path(__file__).parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

# Use non-interactive backend before any other matplotlib import.
import matplotlib

matplotlib.use("Agg")

import argparse
import csv
import io
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt

from experiments.tracker import create_experiment, update_experiment
from schemas import ScenarioResult
from valuation.scenario import load_assumption_yaml, run_scenario

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_SCENARIO_DIR = Path("configs/scenarios")
DEFAULT_REPORTS_DIR = Path("reports")
DEFAULT_EXPERIMENTS_ROOT = Path("experiments")

# Relative perturbation shown on the tornado chart (±TORNADO_DELTA × param value)
TORNADO_DELTA: float = 0.10  # ±10%

# Top-N parameters shown on the tornado chart
TORNADO_TOP_N: int = 6

# Colour scheme (bear → warm red, base → steel blue, bull → forest green)
SCENARIO_COLOURS: dict[str, str] = {
    "bear": "#c0392b",
    "base": "#2980b9",
    "bull": "#27ae60",
}
_DEFAULT_COLOUR = "#7f8c8d"


# ---------------------------------------------------------------------------
# Chart generators
# ---------------------------------------------------------------------------


def _scenario_comparison_chart(
    results: list[ScenarioResult],
    output_path: Path,
) -> None:
    """Horizontal bar chart comparing enterprise value by scenario."""
    names = [r.scenario_name for r in results]
    values = [r.value for r in results]
    colours = [SCENARIO_COLOURS.get(n, _DEFAULT_COLOUR) for n in names]

    fig, ax = plt.subplots(figsize=(7, max(3, len(names) * 1.2)))
    bars = ax.barh(names, values, color=colours, height=0.5)

    for bar, val in zip(bars, values):
        ax.text(
            val * 1.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:,.0f}",
            va="center",
            fontsize=9,
        )

    ax.set_xlabel("Enterprise Value (JPY millions)")
    ax.set_title("DCF Scenario Comparison")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _sensitivity_tornado_chart(
    result: ScenarioResult,
    output_path: Path,
    top_n: int = TORNADO_TOP_N,
) -> None:
    """Tornado chart: ∂EV/∂p × (±TORNADO_DELTA × |p|) for the base scenario.

    The bar width represents the approximate EV swing for a ±10% relative
    change in each parameter.  Parameters are ranked by |swing| descending.
    """
    # Approximate EV swing: ∂EV/∂p × (TORNADO_DELTA × |base_val|)
    # We only have the derivative; use it directly scaled by a unit perturbation.
    # Since sensitivities carry units (EV per unit param), we display the raw
    # derivative ranked by absolute magnitude — interpretable as "EV per unit p".
    sensitivities = result.sensitivity
    if not sensitivities:
        return

    items = sorted(sensitivities.items(), key=lambda kv: abs(kv[1]), reverse=True)
    items = items[:top_n]
    labels = [k for k, _ in items]
    values = [v for _, v in items]

    colours = ["#27ae60" if v >= 0 else "#c0392b" for v in values]

    fig, ax = plt.subplots(figsize=(8, max(3, len(labels) * 0.7)))
    ax.barh(labels, values, color=colours, height=0.5)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("∂EV / ∂param  (JPY millions per unit param)")
    ax.set_title(f"Sensitivity Tornado — '{result.scenario_name}' scenario")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Table generators
# ---------------------------------------------------------------------------

# Sensitivity parameters included in the CSV (deterministic column order)
_SENSITIVITY_COLS = [
    "initial_revenue",
    "revenue_growth",
    "ebit_margin",
    "tax_rate",
    "capex_rate",
    "discount_rate",
    "terminal_growth_rate",
]


def _results_csv(results: list[ScenarioResult]) -> str:
    """Return a CSV string with one row per scenario."""
    buf = io.StringIO()
    fieldnames = ["scenario", "ev_jpy_millions", "assumption_version"] + [
        f"dEV_d{col}" for col in _SENSITIVITY_COLS
    ]
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for r in results:
        row: dict = {
            "scenario": r.scenario_name,
            "ev_jpy_millions": round(r.value, 2),
            "assumption_version": r.assumption_version,
        }
        for col in _SENSITIVITY_COLS:
            row[f"dEV_d{col}"] = round(r.sensitivity.get(col, float("nan")), 4)
        writer.writerow(row)
    return buf.getvalue()


def _summary_markdown(
    results: list[ScenarioResult],
    exp_id: str,
    output_dir: Path,
    generated_at: datetime,
) -> str:
    """Return a Markdown summary string."""
    lines: list[str] = [
        f"# Report — {exp_id}",
        "",
        f"Generated: {generated_at.strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Enterprise Value by Scenario",
        "",
        "| Scenario | EV (JPY millions) |",
        "|----------|-------------------|",
    ]
    for r in sorted(results, key=lambda x: x.value):
        lines.append(f"| {r.scenario_name} | {r.value:,.0f} |")

    # Base scenario sensitivity top-3
    base_result = next((r for r in results if r.scenario_name == "base"), results[0])
    top3 = sorted(base_result.sensitivity.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]

    lines += [
        "",
        f"## Top Sensitivities — '{base_result.scenario_name}' scenario",
        "",
        "| Parameter | ∂EV/∂param (JPY millions / unit) |",
        "|-----------|----------------------------------|",
    ]
    for param, val in top3:
        lines.append(f"| {param} | {val:,.2f} |")

    lines += [
        "",
        "## Output Files",
        "",
        f"- `{output_dir}/scenario_comparison.png`",
        f"- `{output_dir}/sensitivity_tornado.png`",
        f"- `{output_dir}/results_table.csv`",
        f"- `{output_dir}/summary.md`",
        "",
        "> Assumptions are heuristic. Update with empirical evidence before reporting.",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_report(
    scenario_dir: str | Path = DEFAULT_SCENARIO_DIR,
    reports_dir: str | Path = DEFAULT_REPORTS_DIR,
    experiments_root: str | Path = DEFAULT_EXPERIMENTS_ROOT,
) -> dict[str, Path]:
    """Execute the full report pipeline.

    Parameters
    ----------
    scenario_dir:
        Directory containing ``*.yaml`` scenario files.
    reports_dir:
        Root directory for report output.  A sub-folder ``<exp_id>/`` is
        created inside it to hold all artifacts for this run.
    experiments_root:
        Root directory for experiment registry.

    Returns
    -------
    dict[str, Path]
        Mapping artifact name → absolute Path of generated file.
    """
    scenario_dir = Path(scenario_dir)
    reports_dir = Path(reports_dir)
    experiments_root = Path(experiments_root)
    generated_at = datetime.now(UTC)

    # 1. Load and run all scenarios
    results: list[ScenarioResult] = []
    for yaml_path in sorted(scenario_dir.glob("*.yaml")):
        assumption = load_assumption_yaml(yaml_path)
        results.append(run_scenario(assumption))

    if not results:
        raise FileNotFoundError(f"No scenario YAML files found in {scenario_dir}")

    # 2. Register experiment (get the ID before writing files)
    base_result = next((r for r in results if r.scenario_name == "base"), results[0])
    meta = create_experiment(
        title=f"DCF report — {generated_at.strftime('%Y-%m-%d')}",
        config_path=str(scenario_dir),
        tags=["dcf", "report", "automated"],
        summary=(
            f"Automated report: {len(results)} scenarios. "
            f"Base EV = {base_result.value:,.0f} JPY millions."
        ),
        experiments_root=experiments_root,
    )

    # 3. Create output directory for this run
    out_dir = reports_dir / meta.experiment_id
    out_dir.mkdir(parents=True, exist_ok=True)

    artifacts: dict[str, Path] = {}

    # 4. Charts
    comp_path = out_dir / "scenario_comparison.png"
    _scenario_comparison_chart(results, comp_path)
    artifacts["scenario_comparison"] = comp_path

    tornado_path = out_dir / "sensitivity_tornado.png"
    _sensitivity_tornado_chart(base_result, tornado_path)
    artifacts["sensitivity_tornado"] = tornado_path

    # 5. CSV table
    csv_path = out_dir / "results_table.csv"
    csv_path.write_text(_results_csv(results), encoding="utf-8")
    artifacts["results_table"] = csv_path

    # 6. Markdown summary
    md_path = out_dir / "summary.md"
    md_path.write_text(
        _summary_markdown(results, meta.experiment_id, out_dir, generated_at),
        encoding="utf-8",
    )
    artifacts["summary"] = md_path

    # 7. Update experiment metadata with output path
    update_experiment(
        meta.experiment_id,
        experiments_root=experiments_root,
        result_path=str(out_dir),
    )

    return artifacts


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate DCF valuation report from scenario YAML files."
    )
    parser.add_argument(
        "--scenario-dir",
        default=str(DEFAULT_SCENARIO_DIR),
        help="Directory containing scenario YAML files (default: configs/scenarios)",
    )
    parser.add_argument(
        "--reports-dir",
        default=str(DEFAULT_REPORTS_DIR),
        help="Root directory for report output (default: reports)",
    )
    parser.add_argument(
        "--experiments-root",
        default=str(DEFAULT_EXPERIMENTS_ROOT),
        help="Root directory for experiment registry (default: experiments)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    artifacts = run_report(
        scenario_dir=args.scenario_dir,
        reports_dir=args.reports_dir,
        experiments_root=args.experiments_root,
    )
    print("Report generated:")
    for name, path in artifacts.items():
        print(f"  {name:25s} {path}")
