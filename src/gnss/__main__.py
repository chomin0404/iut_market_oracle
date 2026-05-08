"""CLI entry point for the GNSS Resilience Twin MC simulation.

Usage:
    uv run python -m src.gnss [--n-mc N] [--seed S] [--out PATH]

Outputs:
    output/resilience_report.json  — ResilienceTwinReport + run metadata
    stdout                         — P_D / P_FA / AUC / per-class accuracy
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add src/ to sys.path so intra-project imports resolve correctly.
_src_dir = str(Path(__file__).parent.parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from gnss.resilience_twin import ResilienceTwinConfig, run_resilience_simulation  # noqa: E402

_DEFAULT_OUT = Path("output/resilience_report.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.gnss",
        description="GNSS Resilience Twin — Monte Carlo simulation",
    )
    parser.add_argument(
        "--n-mc",
        type=int,
        default=400,
        help="Total MC trials (default: 400, must be divisible by 4)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--out",
        type=Path,
        default=_DEFAULT_OUT,
        help=f"Output JSON path (default: {_DEFAULT_OUT})",
    )
    args = parser.parse_args(argv)

    if args.n_mc < 4 or args.n_mc % 4 != 0:
        parser.error("--n-mc must be >= 4 and divisible by 4")

    config = ResilienceTwinConfig(n_mc=args.n_mc, random_seed=args.seed)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    print(f"[gnss] run_id={run_id}  n_mc={config.n_mc}  seed={config.random_seed}")

    report = run_resilience_simulation(config=config)

    out_path: Path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "run_id": run_id,
        "config": {
            "n_mc": config.n_mc,
            "n_sats": config.n_sats,
            "n_epochs": config.n_epochs,
            "random_seed": config.random_seed,
        },
        "report": report.model_dump(mode="json"),
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"  P_D             = {report.p_detection:.4f}")
    print(f"  P_FA            = {report.p_false_alarm:.4f}")
    print(f"  AUC             = {report.auc:.4f}")
    print(f"  mean_confidence = {report.mean_confidence:.4f}")
    print("  per_class_accuracy:")
    for cls, acc in report.per_class_accuracy.items():
        marker = "" if acc >= 0.8 else "  ← below 0.80"
        print(f"    {cls:<22} {acc:.4f}{marker}")
    print(f"[gnss] saved → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
