"""GNSS ML dataset: JSONL export from spoof_sim and loader.

Each JSONL line represents one epoch:
    {
        "run_id":      str,
        "epoch":       int,
        "doppler_dev": [float, ...],   # length = n_sats
        "m_t":         float,
        "chi_t":       float,
        "pvt_error":   float,
        "fisher_score": float,
        "label":       int             # 0 = genuine, 1 = spoofed
    }
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gnss.math_utils import init_constellation
from gnss.spoof_sim import SimConfig


def generate_full_dataset(
    config: SimConfig | None = None,
    *,
    n_runs: int = 400,
    output_path: Path | None = None,
) -> list[dict]:
    """Generate per-epoch feature records including per-satellite Doppler deviations.

    Runs the simulation from scratch to collect doppler_dev, m_t, chi_t per epoch.
    This is heavier than generate_dataset but provides the full feature vector.

    Args:
        config:      SimConfig; uses defaults if None.
        n_runs:      Total MC runs.
        output_path: If given, writes JSONL to this path.

    Returns:
        List of epoch-level dicts with full features.
    """
    if config is None:
        config = SimConfig()

    from gnss.constants import _INS_CLOCK_STD, _INS_VEL_STD
    from gnss.spoof_sim import (
        _build_features,
        _build_similarity_graph,
        _gen_genuine_measurements,
        _init_receiver,
        _inject_attack,
        _propagate_state,
        _sample_attack_window,
        fuse_score,
        percolation_stats,
        select_subset,
        wls_pvt,
    )

    rng = np.random.default_rng(config.random_seed)
    los = init_constellation(config.n_sats)

    records: list[dict] = []

    for mc in range(n_runs):
        attacked = mc % 2 == 0
        run_id = f"mc_{mc:04d}"

        vel, clock_drift = _init_receiver(rng)
        b_common = rng.normal(0.0, config.spoof_bias_std)

        if attacked:
            attack_start, attack_end = _sample_attack_window(
                config.n_epochs, config.dirichlet_alpha, rng
            )
        else:
            attack_start = attack_end = 0

        for t in range(config.n_epochs):
            vel, clock_drift = _propagate_state(vel, clock_drift, rng)
            vel_hat = vel + rng.normal(0.0, _INS_VEL_STD, size=3)
            clock_hat = clock_drift + rng.normal(0.0, _INS_CLOCK_STD)

            under_attack = attacked and (attack_start <= t < attack_end)

            meas = _gen_genuine_measurements(
                los,
                vel,
                clock_drift,
                vel_hat,
                clock_hat,
                config.doppler_noise_std,
                rng,
            )
            if under_attack:
                meas = _inject_attack(meas, b_common, config.spoof_diff_std, config.n_sats, rng)

            feats = _build_features(meas)
            G = _build_similarity_graph(feats, config.graph_sigma)
            m_t, chi_t = percolation_stats(G, config.doppler_noise_std)
            S = select_subset(G.W, config.subset_size)
            _, residuals = wls_pvt(los, meas, G.W, S)
            pvt_err = float(np.linalg.norm(residuals))
            score = fuse_score(m_t, chi_t, residuals, G.W, S, config)

            records.append(
                {
                    "run_id": run_id,
                    "epoch": t,
                    "doppler_dev": meas.tolist(),
                    "m_t": float(m_t),
                    "chi_t": float(chi_t),
                    "pvt_error": pvt_err,
                    "fisher_score": score,
                    "label": int(under_attack),
                }
            )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec) + "\n")

    return records


def load_jsonl(path: Path) -> list[dict]:
    """Load JSONL file produced by generate_full_dataset."""
    path = Path(path)
    records = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def records_to_arrays(
    records: list[dict],
    n_sats: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert list of epoch records to (X, y) arrays.

    Feature order: [doppler_dev × n_sats, m_t, chi_t, pvt_error]
    Shape: X = (N, n_sats + 3),  y = (N,) int {0, 1}

    Records without 'doppler_dev' use fisher_score + pvt_error only (2-dim fallback).
    """
    X_rows = []
    y_rows = []

    for rec in records:
        if "doppler_dev" in rec:
            row = list(rec["doppler_dev"])[:n_sats]
            row += [rec["m_t"], rec["chi_t"], rec["pvt_error"]]
        else:
            # Fallback for records without per-satellite Doppler (fisher_score + pvt_error only)
            row = [rec["fisher_score"], rec["pvt_error"]]
        X_rows.append(row)
        y_rows.append(rec["label"])

    return np.array(X_rows, dtype=np.float32), np.array(y_rows, dtype=np.int64)
