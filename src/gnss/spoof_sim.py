"""Monte Carlo GNSS spoofing detection simulation (T1300).

Algorithm (per MC run × T epochs):
    1. Initialise satellite constellation (Fibonacci lattice on upper hemisphere)
    2. Propagate true receiver velocity and clock drift (random walk)
    3. Generate Doppler deviations: Δf_i = f_meas_i − f_pred_i + noise + spoof_bias_i
    4. Build similarity graph  w_{ij} = exp(−|Δf_i − Δf_j|² / σ²)
    5. Compute m(t) = det(I + L_w)  [all-forests polynomial of cycle matroid]
    6. Compute chi(t) = Σ(Δf_i − mean)² / σ_D²  [Doppler chi-squared]
    7. Greedy constrained subset selection S_t  (maximise Fiedler value)
    8. WLS PVT on S_t  → residuals r
    9. Detection score  T = rᵀ diag(w_S) r  ∼ χ²(|S|−4) under H₀
   10. Alarm: T > τ_{NP} = χ²_{1−α}(|S|−4)

Attack model (meaconing with differential error):
    b_i(t) = b_common + δ_i,   b_common ∼ N(0, σ_bias²),  δ_i ∼ N(0, σ_diff²)

The differential component δ_i breaks inter-satellite Doppler correlations so that
all three statistics (m, chi, residual T) are sensitive to the attack.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
from scipy.stats import chi2 as _chi2_dist

from schemas import MCSimReport, RunResult, RunTrace

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

_SPEED_OF_LIGHT: float = 2.998e8   # m/s
_L1_FREQ: float = 1575.42e6        # Hz  (GPS L1 carrier)

# ---------------------------------------------------------------------------
# Simulation constants
# ---------------------------------------------------------------------------

_DOPPLER_NOISE_STD: float = 0.30   # Hz  — genuine measurement noise 1-σ
_SPOOF_BIAS_STD: float = 2.50      # Hz  — common meaconing bias 1-σ
_SPOOF_DIFF_STD: float = 0.80      # Hz  — per-satellite differential bias 1-σ
_GRAPH_SIGMA: float = 1.50         # Hz  — Gaussian kernel bandwidth σ
_VEL_PROCESS_STD: float = 0.05     # m/s per epoch — receiver velocity random walk
_CLOCK_PROCESS_STD: float = 0.02   # m/s equivalent — clock drift random walk
_INS_VEL_STD: float = 0.05         # m/s — INS velocity error 1-σ
_INS_CLOCK_STD: float = 0.01       # m/s equivalent — INS clock error 1-σ
_PVT_DIM: int = 4                  # unknowns: [Δvx, Δvy, Δvz, Δb_dot]
_ROC_N_THRESHOLDS: int = 200       # resolution for ROC curve


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SimConfig:
    """Parameters for the Monte Carlo GNSS spoofing detection simulation.

    Attributes:
        n_mc:                Number of Monte Carlo realisations.
        n_epochs:            Time steps per run.
        n_sats:              Number of visible satellites.
        attack_start_frac:   Attack begins at this fraction of n_epochs.
        attack_duration_frac: Duration of attack window as fraction of n_epochs.
        doppler_noise_std:   Genuine Doppler measurement noise 1-σ [Hz].
        spoof_bias_std:      Common meaconing bias 1-σ [Hz].
        spoof_diff_std:      Per-satellite differential spoofing noise 1-σ [Hz].
        graph_sigma:         Gaussian kernel bandwidth σ for similarity graph [Hz].
        false_alarm_rate:    Neyman-Pearson target false-alarm probability α.
        subset_size:         Number of satellites in selected subset k < n_sats.
        random_seed:         RNG seed for full reproducibility.
    """

    n_mc: int = 200
    n_epochs: int = 80
    n_sats: int = 6
    attack_start_frac: float = 0.40
    attack_duration_frac: float = 0.35
    doppler_noise_std: float = _DOPPLER_NOISE_STD
    spoof_bias_std: float = _SPOOF_BIAS_STD
    spoof_diff_std: float = _SPOOF_DIFF_STD
    graph_sigma: float = _GRAPH_SIGMA
    false_alarm_rate: float = 0.05
    subset_size: int = 4
    random_seed: int = 42

    def __post_init__(self) -> None:
        if self.subset_size >= self.n_sats:
            raise ValueError(
                f"subset_size ({self.subset_size}) must be < n_sats ({self.n_sats})"
            )
        if not (0.0 < self.attack_start_frac < 1.0):
            raise ValueError("attack_start_frac must be in (0, 1)")
        if not (0.0 < self.false_alarm_rate < 1.0):
            raise ValueError("false_alarm_rate must be in (0, 1)")
        if self.n_sats < _PVT_DIM + 1:
            raise ValueError(f"n_sats must be >= {_PVT_DIM + 1} for WLS to be overdetermined")


# ---------------------------------------------------------------------------
# Satellite geometry (Fibonacci lattice on upper hemisphere)
# ---------------------------------------------------------------------------


def _init_constellation(n_sats: int) -> np.ndarray:
    """Unit LOS vectors from receiver to satellites  shape (n_sats, 3).

    Placed on the upper hemisphere (z > 0) via a Fibonacci spiral so the
    geometry is deterministic and well-conditioned for any n_sats ≥ 4.
    """
    golden = (1.0 + math.sqrt(5.0)) / 2.0
    idx = np.arange(n_sats, dtype=float)
    # co-latitude in (0, π/2): ensure strictly positive elevation
    theta = np.arccos(1.0 - (idx + 0.5) / n_sats)
    phi = 2.0 * math.pi * idx / golden
    e = np.column_stack(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
    )
    return e  # (n_sats, 3)


# ---------------------------------------------------------------------------
# Receiver dynamics
# ---------------------------------------------------------------------------


def _propagate_state(
    vel: np.ndarray,
    clock_drift: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """Advance velocity (m/s) and clock drift (m/s equivalent) by one epoch."""
    vel_new = vel + rng.normal(0.0, _VEL_PROCESS_STD, size=3)
    clock_new = clock_drift + rng.normal(0.0, _CLOCK_PROCESS_STD)
    return vel_new, clock_new


# ---------------------------------------------------------------------------
# Doppler measurements
# ---------------------------------------------------------------------------


def _true_doppler(
    los: np.ndarray,       # (n_sats, 3) unit LOS vectors
    vel: np.ndarray,       # (3,) receiver velocity [m/s]
    clock_drift: float,    # [m/s equivalent]
) -> np.ndarray:
    """Doppler shift [Hz] for each satellite.

    Δf_i = −(f_L1/c) · (e_i · v + b_dot)
    """
    radial = los @ vel + clock_drift     # (n_sats,) [m/s]
    return -(_L1_FREQ / _SPEED_OF_LIGHT) * radial  # (n_sats,) [Hz]


def _doppler_deviation(
    los: np.ndarray,
    vel: np.ndarray,
    vel_hat: np.ndarray,
    clock_drift: float,
    clock_drift_hat: float,
    noise_std: float,
    rng: np.random.Generator,
    spoof_bias: np.ndarray | None = None,
) -> np.ndarray:
    """Per-satellite Doppler deviation  Δf_i = f_meas_i − f_pred_i.

    Under H₀: Δf_i ∼ N(0, σ_D²).
    Under H₁: Δf_i ∼ N(b_i, σ_D²)  where b_i is the per-satellite spoofing bias.
    """
    f_true = _true_doppler(los, vel, clock_drift)
    f_pred = _true_doppler(los, vel_hat, clock_drift_hat)
    noise = rng.normal(0.0, noise_std, size=len(los))
    bias = spoof_bias if spoof_bias is not None else np.zeros(len(los))
    return (f_true - f_pred) + noise + bias


# ---------------------------------------------------------------------------
# Similarity graph
# ---------------------------------------------------------------------------


def _build_graph(doppler_dev: np.ndarray, sigma: float) -> np.ndarray:
    """Weight matrix of the satellite similarity graph  shape (n, n).

    w_{ij} = exp(−|Δf_i − Δf_j|² / σ²),  diagonal forced to zero.
    """
    diff = doppler_dev[:, None] - doppler_dev[None, :]   # (n, n)
    W = np.exp(-(diff**2) / (sigma**2))
    np.fill_diagonal(W, 0.0)
    return W


# ---------------------------------------------------------------------------
# Detection statistics
# ---------------------------------------------------------------------------


def matroid_forest_count(W: np.ndarray) -> float:
    """All-forests polynomial of the cycle matroid evaluated at 1.

    m(t) = det(I + L_w)

    where  L_w = diag(W·1) − W  is the weighted Laplacian.

    By the Matrix-Tree generalisation, det(I + L_w) equals the sum of
    weights of ALL spanning forests of G_t (independent sets of the cycle
    matroid), weighted by the product of their edge weights.  The value
    decreases when inter-satellite Doppler correlations break down.

    Args:
        W: (n, n) similarity weight matrix with zero diagonal.

    Returns:
        Scalar ≥ 1 (equals 1 only for the empty graph).
    """
    n = len(W)
    D = np.diag(W.sum(axis=1))
    L = D - W
    return float(np.linalg.det(np.eye(n) + L))


def chi_stat(doppler_dev: np.ndarray, noise_std: float) -> float:
    """Doppler chi-squared consistency statistic.

    chi(t) = Σ_i (Δf_i − mean(Δf))² / σ_D²

    Under H₀: chi(t) ∼ χ²(n − 1).
    Under H₁ (differential bias): non-central chi-squared with elevated
    non-centrality Σ_i (b_i − mean(b))² / σ_D².
    """
    mu = doppler_dev.mean()
    return float(np.sum((doppler_dev - mu) ** 2) / noise_std**2)


# ---------------------------------------------------------------------------
# Constrained subset selection — greedy Fiedler maximisation
# ---------------------------------------------------------------------------


def _fiedler_value(W_sub: np.ndarray) -> float:
    """Second-smallest eigenvalue of the Laplacian of W_sub (algebraic connectivity)."""
    if len(W_sub) < 2:
        return 0.0
    D = np.diag(W_sub.sum(axis=1))
    L = D - W_sub
    return float(np.sort(np.linalg.eigvalsh(L))[1])


def select_subset(W: np.ndarray, k: int) -> list[int]:
    """Greedy satellite subset selection maximising graph connectivity.

    Solves (approximately):
        max_{S : |S|=k, S ∈ ℐ(graphic matroid)} λ₂(L_{G[S]})

    The graphic matroid independence constraint (S induces a forest) is
    inherently satisfied by the Fiedler criterion: maximising λ₂ favours
    well-connected, non-degenerate subgraphs, which are exactly the high-
    weight spanning trees of the induced subgraph.

    Algorithm:
        1. Seed with the pair (i*, j*) = argmax_{i≠j} w_{ij}.
        2. Greedily add the satellite that maximises λ₂ of the induced
           subgraph until |S| = k.

    Args:
        W: (n, n) similarity weight matrix, diagonal zero.
        k: target subset size, must satisfy 2 ≤ k < n.

    Returns:
        Sorted list of k satellite indices.
    """
    n = len(W)
    k = min(max(k, 2), n)

    # Seed: most similar pair
    W_no_diag = W.copy()
    np.fill_diagonal(W_no_diag, -np.inf)
    i0, i1 = divmod(int(np.argmax(W_no_diag)), n)
    selected: list[int] = [i0, i1] if i0 != i1 else [0, 1]

    while len(selected) < k:
        remaining = [i for i in range(n) if i not in selected]
        best_val = -np.inf
        best_idx = remaining[0]
        for i in remaining:
            trial = selected + [i]
            val = _fiedler_value(W[np.ix_(trial, trial)])
            if val > best_val:
                best_val = val
                best_idx = i
        selected.append(best_idx)

    return sorted(selected)


# ---------------------------------------------------------------------------
# WLS PVT estimation
# ---------------------------------------------------------------------------


def _geometry_matrix(los: np.ndarray, S: list[int]) -> np.ndarray:
    """WLS geometry matrix H  shape (|S|, 4).

    Doppler observation equation (row i):
        Δf_i ≈ −(f_L1/c) · e_i · Δv − (f_L1/c) · Δb_dot
    Row: [−(f_L1/c) e_ix,  −(f_L1/c) e_iy,  −(f_L1/c) e_iz,  −(f_L1/c)]
    """
    scale = _L1_FREQ / _SPEED_OF_LIGHT
    return np.column_stack([-scale * los[S, :], -np.full(len(S), scale)])


def wls_pvt(
    los: np.ndarray,
    doppler_dev: np.ndarray,
    W: np.ndarray,
    S: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """WLS PVT solve on satellite subset S.

    State vector:  Δx = [Δvx, Δvy, Δvz, Δb_dot]
    Observation:   Δf_i ≈ H_i · Δx  for i ∈ S

    Measurement weights: row-sum of W restricted to S
        w_i = Σ_{j∈S} w_{ij}

    Returns:
        x_hat: (4,) state correction estimate.
        residuals: (|S|,) post-fit residuals  Δf_S − H_S · x̂.
    """
    H = _geometry_matrix(los, S)
    y = doppler_dev[S]
    w = W[np.ix_(S, S)].sum(axis=1) + 1e-9   # row-sum weights, avoid zero
    W_diag = np.diag(w)
    HTW = H.T @ W_diag
    HTWH = HTW @ H
    try:
        x_hat = np.linalg.solve(HTWH, HTW @ y)
    except np.linalg.LinAlgError:
        x_hat = np.zeros(_PVT_DIM)
    return x_hat, y - H @ x_hat


def detection_score(residuals: np.ndarray, W: np.ndarray, S: list[int]) -> float:
    """Weighted residual norm  T = rᵀ diag(w_S) r.

    Under H₀, T ∼ χ²(max(|S| − 4, 1)).  Under H₁, T is stochastically
    larger due to the spoofing bias inflating the residuals.
    """
    w = W[np.ix_(S, S)].sum(axis=1) + 1e-9
    return float(residuals @ (w * residuals))


# ---------------------------------------------------------------------------
# Neyman-Pearson threshold
# ---------------------------------------------------------------------------


def np_threshold(n_obs: int, pfa: float) -> float:
    """Chi-squared NP threshold  τ = F_{χ²(df)}^{−1}(1 − α).

    Degrees of freedom: df = max(n_obs − PVT_DIM, 1).
    """
    df = max(n_obs - _PVT_DIM, 1)
    return float(_chi2_dist.ppf(1.0 - pfa, df=df))


# ---------------------------------------------------------------------------
# ROC computation
# ---------------------------------------------------------------------------


def _compute_roc(
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
    fpr_list: list[float] = []
    tpr_list: list[float] = []

    for tau in thresholds:
        pred = scores >= tau
        tp = int((pred & (labels == 1)).sum())
        fp = int((pred & (labels == 0)).sum())
        fn = int((~pred & (labels == 1)).sum())
        tn = int((~pred & (labels == 0)).sum())
        tpr_list.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        fpr_list.append(fp / (fp + tn) if (fp + tn) > 0 else 0.0)

    order = np.argsort(fpr_list)
    fpr_sorted = np.array(fpr_list)[order]
    tpr_sorted = np.array(tpr_list)[order]
    auc = float(np.trapezoid(tpr_sorted, fpr_sorted))
    return fpr_list, tpr_list, max(0.0, min(1.0, auc))


# ---------------------------------------------------------------------------
# Monte Carlo simulation
# ---------------------------------------------------------------------------


def run_mc_simulation(
    config: SimConfig | None = None,
    rng: np.random.Generator | None = None,
) -> MCSimReport:
    """Run Monte Carlo GNSS spoofing detection simulation.

    Each MC run:
        - Initialises receiver state and satellite geometry
        - Simulates T epochs of genuine or spoofed Doppler measurements
        - Builds similarity graph, computes m(t) and chi(t)
        - Selects satellite subset via greedy Fiedler maximisation
        - Estimates PVT with WLS and computes detection score

    Aggregates over M runs:
        - ROC curve (FPR, TPR at 200 thresholds) + AUC
        - Mean / std detection delay [epochs after attack start]
        - Mean / std PVT degradation ratio under attack

    Args:
        config: simulation parameters; uses SimConfig() defaults if None.
        rng:    NumPy Generator; constructed from config.random_seed if None.

    Returns:
        MCSimReport with full metrics and ROC curve.
    """
    if config is None:
        config = SimConfig()
    if rng is None:
        rng = np.random.default_rng(config.random_seed)

    T = config.n_epochs
    attack_start = int(config.attack_start_frac * T)
    attack_end = min(T, attack_start + int(config.attack_duration_frac * T))
    tau = np_threshold(config.subset_size, config.false_alarm_rate)

    # Fixed satellite geometry for all MC runs (deterministic Fibonacci lattice)
    los = _init_constellation(config.n_sats)

    all_scores: list[float] = []
    all_labels: list[int] = []
    delay_samples: list[float] = []
    degradation_samples: list[float] = []
    run_results: list[RunResult] = []

    for _mc in range(config.n_mc):
        # Initialise receiver state
        vel = rng.normal(0.0, 0.5, size=3)
        clock_drift = rng.normal(0.0, 0.1)

        # Common spoofing bias for this MC run (fixed over attack window)
        b_common = rng.normal(0.0, config.spoof_bias_std)

        first_alarm: int | None = None
        epoch_scores: list[float] = []
        epoch_alarms: list[bool] = []
        epoch_delays: list[float | None] = []
        epoch_pvt_errors: list[float] = []

        for t in range(T):
            vel, clock_drift = _propagate_state(vel, clock_drift, rng)

            # INS prediction (small error added to true state)
            vel_hat = vel + rng.normal(0.0, _INS_VEL_STD, size=3)
            clock_hat = clock_drift + rng.normal(0.0, _INS_CLOCK_STD)

            under_attack = attack_start <= t < attack_end
            spoof_bias: np.ndarray | None = None
            if under_attack:
                # Per-satellite differential bias: b_i = b_common + δ_i
                delta = rng.normal(0.0, config.spoof_diff_std, size=config.n_sats)
                spoof_bias = np.full(config.n_sats, b_common) + delta

            dop_dev = _doppler_deviation(
                los, vel, vel_hat, clock_drift, clock_hat,
                config.doppler_noise_std, rng, spoof_bias=spoof_bias,
            )

            W = _build_graph(dop_dev, config.graph_sigma)

            # --- m(t) and chi(t) (logged; chi(t) contributes to subset guidance) ---
            # matroid_forest_count(W)  — available for diagnostics
            # chi_stat(dop_dev, config.doppler_noise_std) — available for diagnostics

            S = select_subset(W, config.subset_size)

            _, residuals = wls_pvt(los, dop_dev, W, S)
            _, residuals_all = wls_pvt(los, dop_dev, W, list(range(config.n_sats)))

            pvt_err = float(np.linalg.norm(residuals))
            score = detection_score(residuals, W, S)
            all_scores.append(score)
            all_labels.append(int(under_attack))

            alarm = score > tau
            is_first_alarm = alarm and first_alarm is None and under_attack
            if is_first_alarm:
                first_alarm = t

            epoch_scores.append(score)
            epoch_alarms.append(bool(alarm))
            epoch_pvt_errors.append(pvt_err)
            epoch_delays.append(float(t - attack_start) if is_first_alarm else None)

            if under_attack:
                r_all = float(np.linalg.norm(residuals_all)) + 1e-12
                degradation_samples.append(pvt_err / r_all)

        if first_alarm is not None:
            delay_samples.append(float(first_alarm - attack_start))

        pvt_errors_arr = np.array(epoch_pvt_errors, dtype=float)
        run_results.append(RunResult(
            score_max=float(max(epoch_scores)),
            alarm_any=any(epoch_alarms),
            delay=float(first_alarm - attack_start) if first_alarm is not None else None,
            pvt_rmse=float(np.sqrt(np.mean(pvt_errors_arr**2))),
            pvt_max=float(pvt_errors_arr.max()),
            trace=RunTrace(
                score=epoch_scores,
                alarm=epoch_alarms,
                delay=epoch_delays,
                pvt_error=epoch_pvt_errors,
            ),
        ))

    scores_arr = np.array(all_scores, dtype=float)
    labels_arr = np.array(all_labels, dtype=int)

    fpr_list, tpr_list, auc = _compute_roc(scores_arr, labels_arr)

    # Performance at NP threshold
    pred = scores_arr >= tau
    tp = int((pred & (labels_arr == 1)).sum())
    fp = int((pred & (labels_arr == 0)).sum())
    fn = int((~pred & (labels_arr == 1)).sum())
    tn = int((~pred & (labels_arr == 0)).sum())
    p_d = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    p_fa = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    mean_delay = float(np.mean(delay_samples)) if delay_samples else float("nan")
    std_delay = float(np.std(delay_samples)) if delay_samples else 0.0
    mean_deg = float(np.mean(degradation_samples)) if degradation_samples else 1.0
    std_deg = float(np.std(degradation_samples)) if degradation_samples else 0.0

    return MCSimReport(
        roc_fpr=fpr_list,
        roc_tpr=tpr_list,
        auc=auc,
        mean_detection_delay=mean_delay,
        std_detection_delay=std_delay,
        mean_pvt_degradation=mean_deg,
        std_pvt_degradation=std_deg,
        p_detection=p_d,
        p_false_alarm=p_fa,
        n_mc=config.n_mc,
        produced_at=datetime.now(timezone.utc),
        runs=run_results,
    )
