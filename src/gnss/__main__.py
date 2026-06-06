"""CLI entry point for the GNSS Resilience Twin.

Subcommands
-----------
(default)
    Resilience Twin Monte Carlo simulation (backward-compatible default).

    uv run python -m src.gnss [--n-mc N] [--seed S] [--out PATH]

ml train
    Train and evaluate an IsolationForest spoofing detector on MC-generated data.

    uv run python -m src.gnss ml train [--n-runs N] [--n-epochs E] [--n-sats N]
                                       [--seed S] [--out PATH]

ml predict
    Load a saved IsolationForest model and run inference on a JSONL dataset.

    uv run python -m src.gnss ml predict --model PATH --data PATH [--out PATH]
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

_DEFAULT_RESILIENCE_OUT = Path("output/resilience_report.json")
_DEFAULT_ML_TRAIN_OUT = Path("output/ml_train_report.json")
_DEFAULT_ML_MODEL_OUT = Path("output/if_detector.joblib")
_DEFAULT_ML_PREDICT_OUT = Path("output/ml_predict_report.json")


# ---------------------------------------------------------------------------
# Resilience Twin (default subcommand)
# ---------------------------------------------------------------------------


def _main_resilience(argv: list[str] | None) -> int:
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
        default=_DEFAULT_RESILIENCE_OUT,
        help=f"Output JSON path (default: {_DEFAULT_RESILIENCE_OUT})",
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


# ---------------------------------------------------------------------------
# ML subcommand: train
# ---------------------------------------------------------------------------


def _main_ml_train(argv: list[str] | None) -> int:
    from gnss.dataset import generate_full_dataset, records_to_arrays
    from gnss.ml_detector import IsolationForestDetector
    from gnss.spoof_sim import SimConfig

    parser = argparse.ArgumentParser(
        prog="python -m src.gnss ml train",
        description="Train IsolationForest spoofing detector and report metrics.",
    )
    parser.add_argument("--n-runs", type=int, default=200, help="Total MC runs (default: 200)")
    parser.add_argument("--n-epochs", type=int, default=40, help="Epochs per run (default: 40)")
    parser.add_argument("--n-sats", type=int, default=6, help="Visible satellites (default: 6)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.8,
        help="Fraction of runs for training (default: 0.8)",
    )
    parser.add_argument(
        "--target-far",
        type=float,
        default=1e-4,
        help="Target false-alarm rate for calibration (default: 1e-4)",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=_DEFAULT_ML_MODEL_OUT,
        help=f"Trained model output path (default: {_DEFAULT_ML_MODEL_OUT})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_DEFAULT_ML_TRAIN_OUT,
        help=f"Training report JSON path (default: {_DEFAULT_ML_TRAIN_OUT})",
    )
    args = parser.parse_args(argv)

    import numpy as np

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    print(
        f"[gnss ml train] run_id={run_id}  n_runs={args.n_runs}  "
        f"n_epochs={args.n_epochs}  seed={args.seed}"
    )

    config = SimConfig(n_epochs=args.n_epochs, n_sats=args.n_sats, random_seed=args.seed)
    records = generate_full_dataset(config=config, n_runs=args.n_runs)

    all_run_ids: list[str] = list(dict.fromkeys(r["run_id"] for r in records))
    n_train = max(1, int(len(all_run_ids) * args.train_fraction))
    train_ids: set[str] = set(all_run_ids[:n_train])
    test_ids: set[str] = set(all_run_ids[n_train:])

    train_recs = [r for r in records if r["run_id"] in train_ids]
    test_recs = [r for r in records if r["run_id"] in test_ids]

    X_train, y_train = records_to_arrays(train_recs, n_sats=args.n_sats)
    X_test, y_test = records_to_arrays(test_recs, n_sats=args.n_sats)
    X_test_genuine = X_test[y_test == 0]
    X_test_spoofed = X_test[y_test == 1]

    detector = IsolationForestDetector()
    detector.fit(X_train, y_train)
    threshold = detector.calibrate_threshold(X_test_genuine, target_far=args.target_far)

    alarm_genuine, _ = detector.predict(X_test_genuine)
    alarm_spoofed, _ = detector.predict(X_test_spoofed)
    dr = float(np.mean(alarm_spoofed))
    far = float(np.mean(alarm_genuine))

    model_path: Path = args.model_out
    model_path.parent.mkdir(parents=True, exist_ok=True)
    detector.save(model_path)

    out_path: Path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "config": {
            "n_runs": args.n_runs,
            "n_epochs": args.n_epochs,
            "n_sats": args.n_sats,
            "random_seed": args.seed,
            "train_fraction": args.train_fraction,
            "target_far": args.target_far,
        },
        "metrics": {
            "detection_rate": dr,
            "false_alarm_rate": far,
            "threshold": threshold,
            "n_train_genuine": int((y_train == 0).sum()),
            "n_train_spoofed": int((y_train == 1).sum()),
            "n_test_genuine": int(len(X_test_genuine)),
            "n_test_spoofed": int(len(X_test_spoofed)),
        },
        "model_path": str(model_path),
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"  detection_rate   = {dr:.4f}")
    print(f"  false_alarm_rate = {far:.6f}  (target <= {args.target_far:.1e})")
    print(f"  threshold        = {threshold:.6f}")
    print(f"  n_train_genuine  = {payload['metrics']['n_train_genuine']}")
    print(f"  n_train_spoofed  = {payload['metrics']['n_train_spoofed']}")
    print(f"[gnss ml train] model  → {model_path}")
    print(f"[gnss ml train] report → {out_path}")
    return 0


# ---------------------------------------------------------------------------
# ML subcommand: predict
# ---------------------------------------------------------------------------


def _main_ml_predict(argv: list[str] | None) -> int:
    from gnss.dataset import load_jsonl, records_to_arrays
    from gnss.ml_detector import IsolationForestDetector

    parser = argparse.ArgumentParser(
        prog="python -m src.gnss ml predict",
        description="Load a trained IsolationForest model and run inference on a JSONL dataset.",
    )
    parser.add_argument("--model", type=Path, required=True, help="Path to saved .joblib model")
    parser.add_argument("--data", type=Path, required=True, help="Path to JSONL dataset file")
    parser.add_argument(
        "--n-sats",
        type=int,
        default=6,
        help="Number of visible satellites (default: 6, must match training)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_DEFAULT_ML_PREDICT_OUT,
        help=f"Prediction report JSON path (default: {_DEFAULT_ML_PREDICT_OUT})",
    )
    args = parser.parse_args(argv)

    import numpy as np

    detector = IsolationForestDetector.load(args.model)
    records = load_jsonl(args.data)
    X, y = records_to_arrays(records, n_sats=args.n_sats)

    alarm, score = detector.predict(X)

    run_ids = [r["run_id"] for r in records]
    out_records = [
        {
            "run_id": run_ids[i],
            "epoch": records[i].get("epoch", i),
            "alarm": bool(alarm[i]),
            "score": float(score[i]),
            "label": int(y[i]),
        }
        for i in range(len(records))
    ]

    n_genuine = int((y == 0).sum())
    n_spoofed = int((y == 1).sum())
    dr = float(alarm[y == 1].mean()) if n_spoofed > 0 else float("nan")
    far = float(alarm[y == 0].mean()) if n_genuine > 0 else float("nan")

    out_path: Path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": str(args.model),
        "data": str(args.data),
        "summary": {
            "n_samples": len(records),
            "n_genuine": n_genuine,
            "n_spoofed": n_spoofed,
            "detection_rate": dr,
            "false_alarm_rate": far,
            "threshold": detector._threshold,
        },
        "predictions": out_records,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"  n_samples        = {len(records)}")
    print(f"  detection_rate   = {dr:.4f}" if not np.isnan(dr) else "  detection_rate   = n/a")
    print(f"  false_alarm_rate = {far:.6f}" if not np.isnan(far) else "  false_alarm_rate = n/a")
    print(f"[gnss ml predict] report → {out_path}")
    return 0


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]

    if args and args[0] == "ml":
        ml_argv = args[1:]
        if not ml_argv or ml_argv[0] not in ("train", "predict"):
            print("Usage: python -m src.gnss ml {train|predict} [options]", file=sys.stderr)
            return 2
        subcmd, rest = ml_argv[0], ml_argv[1:]
        if subcmd == "train":
            return _main_ml_train(rest)
        return _main_ml_predict(rest)

    return _main_resilience(list(args) if args else None)


if __name__ == "__main__":
    sys.exit(main())
