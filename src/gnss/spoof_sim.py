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
    9. Fused Fisher score  F = −2(ln p_T + ln p_chi + ln p_m) ∼ χ²(6) under H₀
   10. Alarm: F > τ_{NP} = χ²_{1−α}(6)

Attack model (meaconing with differential error):
    b_i(t) = b_common + δ_i,   b_common ∼ N(0, σ_bias²),  δ_i ∼ N(0, σ_diff²)

Attack window randomisation:
    [pre | attack | post] lengths are Dirichlet(α,α,α)-distributed fractions of T.

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

_SPEED_OF_LIGHT: float = 2.998e8  # m/s
_L1_FREQ: float = 1575.42e6  # Hz  (GPS L1 carrier)

# ---------------------------------------------------------------------------
# Simulation constants
# ---------------------------------------------------------------------------

_DOPPLER_NOISE_STD: float = 0.30  # Hz  — genuine measurement noise 1-σ
_SPOOF_BIAS_STD: float = 2.50  # Hz  — common meaconing bias 1-σ
_SPOOF_DIFF_STD: float = 0.80  # Hz  — per-satellite differential bias 1-σ
_GRAPH_SIGMA: float = 1.50  # Hz  — Gaussian kernel bandwidth σ
_VEL_PROCESS_STD: float = 0.05  # m/s per epoch — receiver velocity random walk
_CLOCK_PROCESS_STD: float = 0.02  # m/s equivalent — clock drift random walk
_INS_VEL_STD: float = 0.05  # m/s — INS velocity error 1-σ
_INS_CLOCK_STD: float = 0.01  # m/s equivalent — INS clock error 1-σ
_PVT_DIM: int = 4  # unknowns: [Δvx, Δvy, Δvz, Δb_dot]
_ROC_N_THRESHOLDS: int = 200  # resolution for ROC curve
_FISHER_DOF: int = 6  # χ²(6): 3 statistics × 2 df each (Fisher combination)
_EPS: float = 1e-300  # p-value floor to prevent log(0)
_DIRICHLET_ALPHA: float = 2.0  # symmetric Dirichlet concentration parameter


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SimConfig:
    """Parameters for the Monte Carlo GNSS spoofing detection simulation.

    Attributes:
        n_mc:              Number of Monte Carlo realisations.
        n_epochs:          Time steps per run.
        n_sats:            Number of visible satellites.
        doppler_noise_std: Genuine Doppler measurement noise 1-σ [Hz].
        spoof_bias_std:    Common meaconing bias 1-σ [Hz].
        spoof_diff_std:    Per-satellite differential spoofing noise 1-σ [Hz].
        graph_sigma:       Gaussian kernel bandwidth σ for similarity graph [Hz].
        false_alarm_rate:  Neyman-Pearson target false-alarm probability α.
        subset_size:       Number of satellites in selected subset k < n_sats.
        dirichlet_alpha:   Symmetric Dirichlet concentration for attack window.
        random_seed:       RNG seed for full reproducibility.
    """

    n_mc: int = 200
    n_epochs: int = 80
    n_sats: int = 6
    doppler_noise_std: float = _DOPPLER_NOISE_STD
    spoof_bias_std: float = _SPOOF_BIAS_STD
    spoof_diff_std: float = _SPOOF_DIFF_STD
    graph_sigma: float = _GRAPH_SIGMA
    false_alarm_rate: float = 0.05
    subset_size: int = 4
    dirichlet_alpha: float = _DIRICHLET_ALPHA
    random_seed: int = 42

    def __post_init__(self) -> None:
        if self.subset_size >= self.n_sats:
            raise ValueError(f"subset_size ({self.subset_size}) must be < n_sats ({self.n_sats})")
        if not (0.0 < self.false_alarm_rate < 1.0):
            raise ValueError("false_alarm_rate must be in (0, 1)")
        if self.n_sats < _PVT_DIM + 1:
            raise ValueError(f"n_sats must be >= {_PVT_DIM + 1} for WLS to be overdetermined")
        if self.dirichlet_alpha <= 0.0:
            raise ValueError("dirichlet_alpha must be positive")


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
    e = np.column_stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    return e  # (n_sats, 3)


# ---------------------------------------------------------------------------
# Receiver initialisation and attack-window sampling
# ---------------------------------------------------------------------------


def _init_receiver(rng: np.random.Generator) -> tuple[np.ndarray, float]:
    """Sample initial receiver velocity [m/s] and clock drift [m/s equiv]."""
    vel = rng.normal(0.0, 0.5, size=3)
    clock_drift = rng.normal(0.0, 0.1)
    return vel, clock_drift


def _sample_attack_window(T: int, alpha: float, rng: np.random.Generator) -> tuple[int, int]:
    """Sample [attack_start, attack_end) from Dirichlet(α,α,α) partition of [0,T).

    The three fractions [pre | attack | post] sum to 1.  At least one attacked
    epoch is guaranteed via  attack_end = max(attack_start + 1, ...).

    Returns:
        (attack_start, attack_end) with 0 ≤ attack_start < attack_end ≤ T.
    """
    fracs = rng.dirichlet([alpha, alpha, alpha])
    attack_start = int(fracs[0] * T)
    attack_end = max(attack_start + 1, int((fracs[0] + fracs[1]) * T))
    return attack_start, min(attack_end, T)


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
    los: np.ndarray,  # (n_sats, 3) unit LOS vectors
    vel: np.ndarray,  # (3,) receiver velocity [m/s]
    clock_drift: float,  # [m/s equivalent]
) -> np.ndarray:
    """Doppler shift [Hz] for each satellite.

    Δf_i = −(f_L1/c) · (e_i · v + b_dot)
    """
    radial = los @ vel + clock_drift  # (n_sats,) [m/s]
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


def _gen_genuine_measurements(
    los: np.ndarray,
    vel: np.ndarray,
    clock_drift: float,
    vel_hat: np.ndarray,
    clock_drift_hat: float,
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Genuine (no-attack) Doppler deviations."""
    return _doppler_deviation(los, vel, vel_hat, clock_drift, clock_drift_hat, noise_std, rng)


def _inject_attack(
    meas: np.ndarray,
    b_common: float,
    spoof_diff_std: float,
    n_sats: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Overlay meaconing bias onto genuine Doppler deviations.

    b_i = b_common + δ_i,   δ_i ∼ N(0, σ_diff²)
    """
    delta = rng.normal(0.0, spoof_diff_std, size=n_sats)
    return meas + b_common + delta


# ---------------------------------------------------------------------------
# Similarity graph
# ---------------------------------------------------------------------------


def _build_graph(doppler_dev: np.ndarray, sigma: float) -> np.ndarray:
    """Weight matrix of the satellite similarity graph  shape (n, n).

    w_{ij} = exp(−|Δf_i − Δf_j|² / σ²),  diagonal forced to zero.
    """
    diff = doppler_dev[:, None] - doppler_dev[None, :]  # (n, n)
    W = np.exp(-(diff**2) / (sigma**2))
    np.fill_diagonal(W, 0.0)
    return W


@dataclass
class SimilarityGraph:
    """Satellite similarity graph with its generating feature vector.

    Attributes:
        W:        (n, n) weight matrix with zero diagonal.
        features: (n,) Doppler deviation vector used to build W.
    """

    W: np.ndarray
    features: np.ndarray


def _build_features(meas: np.ndarray) -> np.ndarray:
    """Extract feature vector for the similarity graph (passthrough; extensible)."""
    return meas


def _build_similarity_graph(feats: np.ndarray, sigma: float) -> SimilarityGraph:
    """Build SimilarityGraph from feature vector and kernel bandwidth."""
    return SimilarityGraph(W=_build_graph(feats, sigma), features=feats)


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


def percolation_stats(G: SimilarityGraph, noise_std: float) -> tuple[float, float]:
    """Compute (m_t, chi_t) from a SimilarityGraph.

    Returns:
        m_t:   All-forests count  det(I + L_w).
        chi_t: Doppler chi-squared statistic.
    """
    return matroid_forest_count(G.W), chi_stat(G.features, noise_std)


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
    w = W[np.ix_(S, S)].sum(axis=1) + 1e-9  # row-sum weights, avoid zero
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
# Fisher's combined test
# ---------------------------------------------------------------------------


def _null_forest_count(n: int, doppler_noise_std: float, graph_sigma: float) -> float:
    """Analytical expected all-forests count under H₀.

    For a fully-symmetric n×n similarity matrix with all off-diagonal entries
    equal to E[w_ij] under H₀:

        Δf_i − Δf_j ∼ N(0, 2σ_D²)   ⟹   E[w_ij] = 1 / √(1 + 4σ_D²/σ²)

    Eigenvalues of (I + L) for this matrix: {1 (×1), 1 + n·w_null (×(n−1))}.

        m_null = (1 + n·w_null)^(n−1)
    """
    w_null = 1.0 / math.sqrt(1.0 + 4.0 * doppler_noise_std**2 / graph_sigma**2)
    return (1.0 + n * w_null) ** (n - 1)


def fuse_score(
    m_t: float,
    chi_t: float,
    residuals: np.ndarray,
    W: np.ndarray,
    S: list[int],
    config: SimConfig,
) -> float:
    """Fisher's combined detection score.

    F = −2(ln p_T + ln p_chi + ln p_m) ∼ χ²(6) under H₀

    Component p-values:
        p_T   = P(χ²(max(k−4, 1)) ≥ T_stat)   where T_stat = rᵀ diag(w_S) r
        p_chi = P(χ²(n−1)          ≥ chi_t)
        p_m   = P(χ²(1)             ≥ 2·max(log(m_null/m_t), 0))

    All p-values are floored at _EPS to prevent log(0).

    Args:
        m_t:      All-forests count at this epoch.
        chi_t:    Doppler chi-squared statistic.
        residuals: (k,) WLS post-fit residuals on subset S.
        W:        (n, n) full similarity matrix.
        S:        Selected satellite indices.
        config:   SimConfig (for n_sats, subset_size, noise/sigma params).

    Returns:
        Scalar Fisher score ≥ 0.
    """
    n, k = config.n_sats, config.subset_size
    T_stat = detection_score(residuals, W, S)
    p_T = max(float(_chi2_dist.sf(T_stat, df=max(k - _PVT_DIM, 1))), _EPS)
    p_chi = max(float(_chi2_dist.sf(chi_t, df=max(n - 1, 1))), _EPS)
    m_null = _null_forest_count(n, config.doppler_noise_std, config.graph_sigma)
    forest_stat = max(2.0 * math.log(max(m_null, 1.0) / max(m_t, 1.0)), 0.0)
    p_m = max(float(_chi2_dist.sf(forest_stat, df=1)), _EPS)
    return -2.0 * (math.log(p_T) + math.log(p_chi) + math.log(p_m))


# ---------------------------------------------------------------------------
# Neyman-Pearson threshold  (legacy — kept for external callers)
# ---------------------------------------------------------------------------


def np_threshold(n_obs: int, pfa: float) -> float:
    """Chi-squared NP threshold  τ = F_{χ²(df)}^{−1}(1 − α).

    Degrees of freedom: df = max(n_obs − PVT_DIM, 1).

    Note: run_mc_simulation now uses χ²(_FISHER_DOF) for the fused score.
    This function is retained for the legacy /simulate endpoint.
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
# Single-trial simulation
# ---------------------------------------------------------------------------


@dataclass
class _TrialSummary:
    """Internal result from a single simulate_trial call."""

    run_result: RunResult
    scores: list[float]
    labels: list[int]
    degradation: list[float]


def simulate_trial(
    config: SimConfig,
    attacked: bool,
    rng: np.random.Generator,
    los: np.ndarray,
    tau: float,
) -> _TrialSummary:
    """Simulate one Monte Carlo trial (T epochs).

    Args:
        config:   Simulation parameters.
        attacked: Whether this trial contains a meaconing attack.
        rng:      RNG state (mutated in place).
        los:      (n_sats, 3) fixed LOS unit vectors.
        tau:      NP alarm threshold for the fused Fisher score χ²(_FISHER_DOF).

    Returns:
        _TrialSummary with per-run aggregates, per-epoch trace, epoch-level
        scores/labels (for ROC), and degradation samples.
    """
    vel, clock_drift = _init_receiver(rng)
    b_common = rng.normal(0.0, config.spoof_bias_std)

    if attacked:
        attack_start, attack_end = _sample_attack_window(
            config.n_epochs, config.dirichlet_alpha, rng
        )
    else:
        attack_start = attack_end = 0

    first_alarm: int | None = None
    epoch_scores: list[float] = []
    epoch_alarms: list[bool] = []
    epoch_delays: list[float | None] = []
    epoch_pvt_errors: list[float] = []
    trial_scores: list[float] = []
    trial_labels: list[int] = []
    degradation: list[float] = []

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
        _, residuals_all = wls_pvt(los, meas, G.W, list(range(config.n_sats)))

        pvt_err = float(np.linalg.norm(residuals))
        score = fuse_score(m_t, chi_t, residuals, G.W, S, config)

        alarm = score > tau
        is_first_alarm = alarm and first_alarm is None and under_attack
        if is_first_alarm:
            first_alarm = t

        trial_scores.append(score)
        trial_labels.append(int(under_attack))
        epoch_scores.append(score)
        epoch_alarms.append(alarm)
        epoch_delays.append(float(t - attack_start) if is_first_alarm else None)
        epoch_pvt_errors.append(pvt_err)

        if under_attack:
            r_all = float(np.linalg.norm(residuals_all)) + 1e-12
            degradation.append(pvt_err / r_all)

    first_delay: float | None = (
        float(first_alarm - attack_start) if first_alarm is not None else None
    )
    pvt_arr = np.array(epoch_pvt_errors, dtype=float)
    run_result = RunResult(
        score_max=float(max(epoch_scores)),
        alarm_any=any(epoch_alarms),
        delay=first_delay,
        pvt_rmse=float(np.sqrt(np.mean(pvt_arr**2))),
        pvt_max=float(pvt_arr.max()),
        trace=RunTrace(
            score=epoch_scores,
            alarm=epoch_alarms,
            delay=epoch_delays,
            pvt_error=epoch_pvt_errors,
        ),
    )
    return _TrialSummary(
        run_result=run_result,
        scores=trial_scores,
        labels=trial_labels,
        degradation=degradation,
    )


# ---------------------------------------------------------------------------
# Monte Carlo simulation
# ---------------------------------------------------------------------------


def run_mc_simulation(
    config: SimConfig | None = None,
    rng: np.random.Generator | None = None,
) -> MCSimReport:
    """Run Monte Carlo GNSS spoofing detection simulation.

    Alternates attacked / genuine trials (mc % 2 == 0 → attacked).
    Each trial calls simulate_trial, which uses a Dirichlet-sampled attack
    window and the fused Fisher score  F ∼ χ²(6)  for alarm decisions.

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

    # NP threshold for the fused Fisher score  F ∼ χ²(_FISHER_DOF)
    tau = float(_chi2_dist.ppf(1.0 - config.false_alarm_rate, df=_FISHER_DOF))

    # Fixed satellite geometry for all MC runs (deterministic Fibonacci lattice)
    los = _init_constellation(config.n_sats)

    all_scores: list[float] = []
    all_labels: list[int] = []
    delay_samples: list[float] = []
    degradation_samples: list[float] = []
    run_results: list[RunResult] = []

    for mc in range(config.n_mc):
        attacked = mc % 2 == 0
        summary = simulate_trial(config, attacked=attacked, rng=rng, los=los, tau=tau)
        run_results.append(summary.run_result)
        all_scores.extend(summary.scores)
        all_labels.extend(summary.labels)
        degradation_samples.extend(summary.degradation)
        if summary.run_result.delay is not None:
            delay_samples.append(summary.run_result.delay)

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
