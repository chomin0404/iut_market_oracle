"""CLI entry point for the growth-fragmentation drone swarm simulation.

Usage
-----
    uv run python -m src.fragmentation [options]

Options
-------
    --n-particles N     Initial sub-swarm count (default: 200)
    --seed S            Random seed (default: 42)
    --T T               Simulation end time (default: 10.0)
    --kappa0 K          Baseline fragmentation rate κ₀ (default: 1.0)
    --beta B            Loss efficiency β ∈ (0,1) (default: 0.9)
    --control-u U       Control input u ≥ 0 (default: 0.0)
    --out PATH          Output JSON path (default: artifacts/fragmentation/result.json)

Example
-------
    uv run python -m src.fragmentation --n-particles 200 --seed 42 --T 10.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src/ to sys.path so intra-project imports resolve correctly.
_src_dir = str(Path(__file__).parent.parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from fragmentation import FragConfig, run_pipeline  # noqa: E402

_DEFAULT_OUT = Path("artifacts/fragmentation/result.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.fragmentation",
        description="Growth-Fragmentation Drone Swarm — Gillespie simulation",
    )
    parser.add_argument("--n-particles", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--T", type=float, default=10.0)
    parser.add_argument("--kappa0", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.9)
    parser.add_argument("--control-u", type=float, default=0.0)
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    args = parser.parse_args(argv)

    config = FragConfig(
        n_particles=args.n_particles,
        seed=args.seed,
        T=args.T,
        kappa_0=args.kappa0,
        loss_efficiency=args.beta,
    )

    print(f"[fragmentation] N={config.n_particles}, T={config.T}, seed={config.seed}")
    print(
        f"[fragmentation] kappa0={config.kappa_0}, beta={config.loss_efficiency},"
        f" u={args.control_u}"
    )

    result = run_pipeline(config, control_u=args.control_u)

    # Print summary
    eigen = result.eigen
    print("\n--- Eigenanalysis ---")
    print(f"  Malthus lambda = {eigen.malthus_lambda:.6f}")
    print(f"  Spectral gap   = {eigen.spectral_gap:.4f}  (converged={eigen.converged})")

    n_final = result.trajectory[-1].n_particles
    print("\n--- Simulation ---")
    print(f"  Events logged : {len(result.trajectory) - 1}")
    print(f"  Final N       : {n_final}")
    print(f"  W2^2 cost     : {result.cost_w2:.6f}")
    print(f"  Mean size     : {result.score_components['mean_size']:.4f}")

    # Save output
    out_path: Path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        result.model_dump_json(indent=2),
        encoding="utf-8",
    )
    print(f"\n[fragmentation] saved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
