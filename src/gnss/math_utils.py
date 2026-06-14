"""Pure mathematical utilities shared across the GNSS pipeline (T1300 / T1500).

All functions here are stateless, have no I/O side-effects, and do not depend
on any simulation-specific logic.  This module is safe to import from both the
production scoring layer (resilience_twin, mvp) and the simulation layer
(spoof_sim, multi_sensor_sim).

Functions
---------
init_constellation   Unit LOS vectors via Fibonacci lattice (upper hemisphere)
build_graph          Gaussian-kernel similarity weight matrix
geometry_matrix      WLS Doppler geometry matrix H  shape (|S|, 4)
compute_roc          ROC curve (FPR, TPR) and scalar AUC
"""

from __future__ import annotations

import math

import numpy as np

from gnss.constants import _L1_FREQ, _SPEED_OF_LIGHT

# ROC resolution — algorithm parameter kept here since it belongs to compute_roc
_ROC_N_THRESHOLDS: int = 200


# ---------------------------------------------------------------------------
# Satellite geometry
# ---------------------------------------------------------------------------


def init_constellation(n_sats: int) -> np.ndarray:
    """Unit LOS vectors from receiver to satellites  shape (n_sats, 3).

    Placed on the upper hemisphere (z > 0) via a Fibonacci spiral so the
    geometry is deterministic and well-conditioned for any n_sats ≥ 4.
    """
    golden = (1.0 + math.sqrt(5.0)) / 2.0
    idx = np.arange(n_sats, dtype=float)
    # co-latitude in (0, π/2): ensure strictly positive elevation
    theta = np.arccos(1.0 - (idx + 0.5) / n_sats)
    phi = 2.0 * math.pi * idx / golden
    e = np.column_stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    return e  # (n_sats, 3)



# ---------------------------------------------------------------------------
# Similarity graph
# ---------------------------------------------------------------------------


def build_graph(doppler_dev: np.ndarray, sigma: float) -> np.ndarray:
    """Weight matrix of the satellite similarity graph  shape (n, n).

    w_{ij} = exp(−|Δf_i − Δf_j|² / σ²),  diagonal forced to zero.
    """
    diff = doppler_dev[:, None] - doppler_dev[None, :]  # (n, n)
    W = np.exp(-(diff**2) / (sigma**2))
    np.fill_diagonal(W, 0.0)
    return W



# ---------------------------------------------------------------------------
# WLS geometry matrix
# ---------------------------------------------------------------------------


def geometry_matrix(los: np.ndarray, S: list[int]) -> np.ndarray:
    """WLS geometry matrix H  shape (|S|, 4).

    Doppler observation equation (row i):
        Δf_i ≈ −(f_L1/c) · e_i · Δv − (f_L1/c) · Δb_dot
    Row: [−(f_L1/c) e_ix,  −(f_L1/c) e_iy,  −(f_L1/c) e_iz,  −(f_L1/c)]
    """
    scale = _L1_FREQ / _SPEED_OF_LIGHT
    return np.column_stack([-scale * los[S, :], -np.full(len(S), scale)])



# ---------------------------------------------------------------------------
# ROC / AUC
# ---------------------------------------------------------------------------


def compute_roc(
    scores: np.ndarray,
    labels: np.ndarray,
) -> tuple[list[float], list[float], float]:
    """Compute ROC curve (FPR, TPR) and AUC.

    Args:
        scores: (N,) detection scores.
        labels: (N,) binary labels (1 = attack, 0 = genuine).

    Returns:
        fpr_list, tpr_list (each length _ROC_N_THRESHOLDS), auc.
    """
    s_min, s_max = float(scores.min()), float(scores.max())
    if s_min >= s_max:
        return [0.0, 1.0], [0.0, 1.0], 0.5

    thresholds = np.linspace(s_min, s_max, _ROC_N_THRESHOLDS)
    pos = labels == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    # pred[k, i] = True iff scores[i] >= thresholds[k]  — shape (T, N)
    pred = scores[None, :] >= thresholds[:, None]
    tp_arr = (pred & pos[None, :]).sum(axis=1).astype(float)
    fp_arr = (pred & (~pos)[None, :]).sum(axis=1).astype(float)
    tpr_arr = tp_arr / max(n_pos, 1)
    fpr_arr = fp_arr / max(n_neg, 1)

    order = np.argsort(fpr_arr)
    auc = float(np.trapezoid(tpr_arr[order], fpr_arr[order]))
    return fpr_arr.tolist(), tpr_arr.tolist(), max(0.0, min(1.0, auc))


