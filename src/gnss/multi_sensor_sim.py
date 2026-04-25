"""Multi-sensor GNSS spoofing detection simulation (T1350).

Sensors: pseudorange (PR), Doppler, angle-of-arrival (AoA), INS residuals.

Algorithm (per trial × T epochs):
    1. Generate satellite geometry via sinusoidal AoA / Doppler model
    2. Build fused multi-sensor measurements
    3. Gradual meaconing attack: mix = min(1, (t−t₀+1)/capture_len)
    4. Percolation graph on (AoA, Doppler, INS) Gaussian-kernel similarity
    5. Greedy AoA-diversity-constrained satellite subset selection
    6. Position error proxy from PR drift + Doppler/INS inconsistency
    7. Weighted detection score: s = w₁·m + w₂·clip(chi/χ₀) + w₃·clip(lor_dev)
    8. Alarm: s > threshold

Attack model (gradual meaconing):
    x(t) = (1−α)·x_genuine + α·x_spoof,   α = min(1, max(0, (t−t₀+1)/C))
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from gnss.spoof_sim import _compute_roc
from schemas import MSRunResult, MSRunTrace, MSSimReport

# ---------------------------------------------------------------------------
# Simulation constants
# ---------------------------------------------------------------------------

_PR_NOMINAL: float = 20_200_000.0      # m  — nominal pseudorange offset
_AoA_SIM_SCALE: float = 10.0           # deg — AoA term in similarity exponent
_DOPP_SIM_SCALE: float = 0.12          # Hz  — Doppler term in similarity exponent
_INS_SIM_SCALE: float = 1.5            # m/s — INS term in similarity exponent
_PERC_THRESHOLD: float = 0.35          # adjacency weight cutoff
_ANGULAR_GAP_MIN: float = 12.0         # deg — min AoA separation for subset
_MAX_SUBSET_SIZE: int = 5
_MIN_SUBSET_SIZE: int = 4
_CHI_SCALE: float = 0.35               # chi normalisation divisor
_AOA_DIVERSITY_SCALE: float = 90.0     # deg — Lorentz deviation denominator
_INS_LOCAL_SCALE: float = 4.0          # m/s — subset ranking: INS weight
_DOPP_LOCAL_SCALE: float = 0.8         # Hz  — subset ranking: Doppler weight
_DOPP_INCONS_SCALE: float = 50.0       # pos-error contribution from Doppler std
_INS_INCONS_SCALE: float = 10.0        # pos-error contribution from INS mean abs
_DRIFT_PER_EPOCH_DENOM: float = 25.0   # m  — spoofed INS drift denominator
_SPOOF_AoA_JITTER: float = 1.0         # deg — per-satellite AoA jitter under attack

# Geometry model: AoA sinusoid amplitude / frequency params
_AoA_ANIM_AMP: float = 8.0
_AoA_ANIM_PHASE_STEP: float = 0.3      # rad per satellite index
_DOPP_AMP: float = 0.6
_DOPP_ORTHO_AMP: float = 0.15
_DOPP_ORTHO_PHASE_STEP: float = 0.21   # rad per satellite index
_PR_GEOM_AMP: float = 15.0             # m


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class MultiSensorConfig:
    """Parameters for the multi-sensor GNSS spoofing detection simulation.

    Attributes:
        T:                  Total epochs per trial.
        dt:                 Epoch duration [s].
        n_sat:              Number of visible satellites (≥ 4).
        attack_start:       First attacked epoch (≥ 0).
        attack_end:         Last attacked epoch (inclusive, < T).
        capture_len:        Capture ramp length [epochs].
        n_nominal:          Number of genuine (label=0) MC trials.
        n_attack:           Number of attacked (label=1) MC trials.
        noise_pr:           PR measurement noise 1-σ [m].
        noise_dopp:         Doppler measurement noise 1-σ [Hz].
        noise_aoa:          AoA measurement noise 1-σ [deg].
        noise_ins:          INS residual noise 1-σ [m/s equiv].
        carryoff_rate:      PR drift rate under attack [m/epoch].
        spoof_aoa_center:   Mean AoA of spoofed signals [deg].
        score_weights:      (w_m, w_chi, w_lor_dev) — detection score weights.
        detect_threshold:   Alarm threshold on the weighted score.
        hazard_pos:         Pos-error proxy threshold for hazard assessment [m].
        random_seed:        RNG seed for reproducibility.
    """

    T: int = 200
    dt: float = 1.0
    n_sat: int = 8
    attack_start: int = 80
    attack_end: int = 140
    capture_len: int = 20
    n_nominal: int = 400
    n_attack: int = 400
    noise_pr: float = 2.0
    noise_dopp: float = 0.08
    noise_aoa: float = 3.0
    noise_ins: float = 1.5
    carryoff_rate: float = 4.0
    spoof_aoa_center: float = 30.0
    score_weights: tuple[float, float, float] = (0.55, 0.25, 0.20)
    detect_threshold: float = 0.62
    hazard_pos: float = 150.0
    random_seed: int = 42

    def __post_init__(self) -> None:
        if not (0 <= self.attack_start < self.attack_end <= self.T):
            raise ValueError(
                f"Must satisfy 0 <= attack_start ({self.attack_start}) < "
                f"attack_end ({self.attack_end}) <= T ({self.T})"
            )
        if self.capture_len < 1:
            raise ValueError("capture_len must be >= 1")
        if len(self.score_weights) != 3:  # type: ignore[arg-type]
            raise ValueError("score_weights must have exactly 3 elements")
        if any(w < 0.0 for w in self.score_weights):
            raise ValueError("score_weights must all be non-negative")
        if self.n_nominal < 1 or self.n_attack < 1:
            raise ValueError("n_nominal and n_attack must be >= 1")
        if self.n_sat < _MIN_SUBSET_SIZE:
            raise ValueError(f"n_sat must be >= {_MIN_SUBSET_SIZE}")


# ---------------------------------------------------------------------------
# Satellite geometry model
# ---------------------------------------------------------------------------


def _geometry_features(
    n_sat: int,
    t: int,
    T: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """True AoA [deg], Doppler [Hz], and PR bias [m] for each satellite.

    AoA:   base + A·sin(2π·t/T + k·φ_step)  [mod 360°]
    Dopp:  a·sin(AoA) + b·cos(2π·t/T + k·ψ_step)
    PRbias: A_pr · cos(AoA)
    """
    base = np.linspace(0.0, 360.0, n_sat, endpoint=False)
    phase = 2.0 * math.pi * t / T + np.arange(n_sat) * _AoA_ANIM_PHASE_STEP
    aoa_true = (base + _AoA_ANIM_AMP * np.sin(phase)) % 360.0
    dopp_true = (
        _DOPP_AMP * np.sin(np.deg2rad(aoa_true))
        + _DOPP_ORTHO_AMP * np.cos(2.0 * math.pi * t / T + np.arange(n_sat) * _DOPP_ORTHO_PHASE_STEP)
    )
    pr_bias_geom = _PR_GEOM_AMP * np.cos(np.deg2rad(aoa_true))
    return aoa_true, dopp_true, pr_bias_geom


# ---------------------------------------------------------------------------
# Measurement generation
# ---------------------------------------------------------------------------


@dataclass
class _Meas:
    """Raw multi-sensor measurements at one epoch."""

    pr: np.ndarray     # (n_sat,) pseudorange [m]
    dopp: np.ndarray   # (n_sat,) Doppler [Hz]
    aoa: np.ndarray    # (n_sat,) angle of arrival [deg]
    ins: np.ndarray    # (n_sat,) INS residuals [m/s equiv]
    mix: float         # spoofing mix fraction α ∈ [0, 1]


def build_measurements(
    t: int,
    attacked: bool,
    cfg: MultiSensorConfig,
    rng: np.random.Generator,
) -> _Meas:
    """Generate multi-sensor measurements at epoch t.

    Genuine:  x = x_true + noise.
    Attacked: x(t) = (1−α)·x_genuine + α·x_spoof,
              α = min(1, max(0, (t−t₀+1)/capture_len)).

    PR drift under attack: Δpr = carryoff_rate · max(0, t−t₀) [m].

    Args:
        t:        Epoch index.
        attacked: Whether this trial has an active attack.
        cfg:      Simulation parameters.
        rng:      NumPy Generator (mutated in place).

    Returns:
        _Meas with all sensor arrays and the current mix fraction.
    """
    aoa_true, dopp_true, pr_geom = _geometry_features(cfg.n_sat, t, cfg.T)

    pr = _PR_NOMINAL + pr_geom + rng.normal(0.0, cfg.noise_pr, cfg.n_sat)
    dopp = dopp_true + rng.normal(0.0, cfg.noise_dopp, cfg.n_sat)
    aoa = aoa_true + rng.normal(0.0, cfg.noise_aoa, cfg.n_sat)
    ins_res = rng.normal(0.0, cfg.noise_ins, cfg.n_sat)
    mix = 0.0

    if attacked and cfg.attack_start <= t <= cfg.attack_end:
        mix = min(1.0, max(0.0, (t - cfg.attack_start + 1) / cfg.capture_len))
        drift = cfg.carryoff_rate * max(0, t - cfg.attack_start)
        pr_spoof = _PR_NOMINAL + drift + rng.normal(0.0, cfg.noise_pr * 0.6, cfg.n_sat)
        dopp_spoof = 0.05 + rng.normal(0.0, cfg.noise_dopp * 0.5, cfg.n_sat)
        aoa_spoof = cfg.spoof_aoa_center + rng.normal(0.0, _SPOOF_AoA_JITTER, cfg.n_sat)
        ins_spoof = drift / _DRIFT_PER_EPOCH_DENOM + rng.normal(0.0, cfg.noise_ins * 0.7, cfg.n_sat)
        pr = (1.0 - mix) * pr + mix * pr_spoof
        dopp = (1.0 - mix) * dopp + mix * dopp_spoof
        aoa = (1.0 - mix) * aoa + mix * aoa_spoof
        ins_res = (1.0 - mix) * ins_res + mix * ins_spoof

    return _Meas(pr=pr, dopp=dopp, aoa=aoa, ins=ins_res, mix=mix)


# ---------------------------------------------------------------------------
# Angle utilities
# ---------------------------------------------------------------------------


def _wrap_angle_deg(a: float) -> float:
    """Wrap angle to (−180°, 180°]."""
    return (a + 180.0) % 360.0 - 180.0


# ---------------------------------------------------------------------------
# Percolation graph statistics
# ---------------------------------------------------------------------------


def ms_percolation_stats(
    meas: _Meas,
    n: int,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Percolation statistics on the multi-sensor satellite similarity graph.

    Edge weight:
        w_ij = exp(−(|ΔAoA|/10 + |ΔDopp|/0.12 + |ΔINS|/1.5))

    Adjacency: A_ij = 1  iff  w_ij > 0.35.

    Returns:
        max_comp:  Largest connected component size / n  ∈ (0, 1].
        chi:       (mean_deg + var_deg / (mean_deg + ε)) / n.
        W:         (n, n) weight matrix.
        A:         (n, n) adjacency matrix (zero diagonal).
    """
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            da = abs(_wrap_angle_deg(float(meas.aoa[i]) - float(meas.aoa[j])))
            dd = abs(float(meas.dopp[i]) - float(meas.dopp[j]))
            di = abs(float(meas.ins[i]) - float(meas.ins[j]))
            w = math.exp(-(da / _AoA_SIM_SCALE + dd / _DOPP_SIM_SCALE + di / _INS_SIM_SCALE))
            W[i, j] = W[j, i] = w

    A = (W > _PERC_THRESHOLD).astype(int)
    np.fill_diagonal(A, 0)

    # BFS to find largest connected component
    seen: set[int] = set()
    max_comp_size = 0
    for start in range(n):
        if start in seen:
            continue
        stack = [start]
        comp_size = 0
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            comp_size += 1
            stack.extend(int(v) for v in np.where(A[u] > 0)[0] if v not in seen)
        if comp_size > max_comp_size:
            max_comp_size = comp_size

    max_comp = max_comp_size / n
    deg = A.sum(axis=1).astype(float)
    deg_mean = float(deg.mean())
    chi = (deg_mean + float(deg.var()) / (deg_mean + 1e-6)) / n
    return max_comp, chi, W, A


# ---------------------------------------------------------------------------
# Satellite subset selection
# ---------------------------------------------------------------------------


def ms_select_subset(meas: _Meas, n: int) -> tuple[np.ndarray, float]:
    """Greedy satellite subset selection with AoA diversity constraint.

    Ranking: local_score_i = |INS_i| / 4 + |Dopp_i| / 0.8  (lower = more nominal).
    Accept satellite i iff its AoA differs from every already-selected satellite
    by ≥ 12°.  Stop at size 5; fall back to top-4 by rank if < 4 accepted.

    Returns:
        chosen:   Integer array of selected satellite indices.
        lor_dev:  Lorentz deviation = max(0, 1 − std(AoA[chosen]) / 90).
                  Near 0 → diverse geometry; near 1 → all AoAs clustered.
    """
    local = np.abs(meas.ins) / _INS_LOCAL_SCALE + np.abs(meas.dopp) / _DOPP_LOCAL_SCALE
    chosen: list[int] = []
    for idx in np.argsort(local):
        idx_int = int(idx)
        ok = all(
            abs(_wrap_angle_deg(float(meas.aoa[idx_int]) - float(meas.aoa[j]))) >= _ANGULAR_GAP_MIN
            for j in chosen
        )
        if ok:
            chosen.append(idx_int)
        if len(chosen) >= _MAX_SUBSET_SIZE:
            break

    if len(chosen) < _MIN_SUBSET_SIZE:
        chosen = [int(i) for i in np.argsort(local)[:_MIN_SUBSET_SIZE]]

    arr = np.array(chosen, dtype=int)
    diversity = float(np.std(meas.aoa[arr])) / _AOA_DIVERSITY_SCALE
    return arr, max(0.0, 1.0 - diversity)


# ---------------------------------------------------------------------------
# Position error proxy
# ---------------------------------------------------------------------------


def _estimate_position_error(meas: _Meas, subset: np.ndarray) -> float:
    """Position error proxy from PR drift and sensor inconsistency.

    proxy = |median(PR[S] − PR_nominal)| + std(Dopp[S])·50 + mean(|INS[S]|)·10
    """
    drift_proxy = float(np.median(meas.pr[subset] - _PR_NOMINAL))
    incons = (
        float(np.std(meas.dopp[subset])) * _DOPP_INCONS_SCALE
        + float(np.mean(np.abs(meas.ins[subset]))) * _INS_INCONS_SCALE
    )
    return abs(drift_proxy) + incons


# ---------------------------------------------------------------------------
# Single-trial simulation
# ---------------------------------------------------------------------------


@dataclass
class _TrialSummary:
    """Internal result from simulate_trial_ms."""

    run_result: MSRunResult
    score_max: float
    label: int  # 0 = genuine, 1 = attacked


def simulate_trial_ms(
    attacked: bool,
    cfg: MultiSensorConfig,
    rng: np.random.Generator,
) -> _TrialSummary:
    """Simulate one multi-sensor trial (T epochs).

    Args:
        attacked: Whether this trial includes a meaconing attack.
        cfg:      Simulation parameters.
        rng:      NumPy Generator (mutated in place).

    Returns:
        _TrialSummary with per-run MSRunResult and score_max for ROC.
    """
    w0, w1, w2 = cfg.score_weights
    first_alarm: int | None = None

    epoch_score: list[float] = []
    epoch_alarm: list[bool] = []
    epoch_mix: list[float] = []
    epoch_m: list[float] = []
    epoch_chi: list[float] = []
    epoch_lor_dev: list[float] = []
    epoch_pos_err: list[float] = []

    for t in range(cfg.T):
        meas = build_measurements(t, attacked, cfg, rng)
        m, chi, _W, _A = ms_percolation_stats(meas, cfg.n_sat)
        subset, lor_dev = ms_select_subset(meas, cfg.n_sat)
        pos_err = _estimate_position_error(meas, subset)

        score = w0 * m + w1 * min(1.0, chi / _CHI_SCALE) + w2 * min(1.0, lor_dev)
        alarm = score > cfg.detect_threshold

        if alarm and first_alarm is None:
            first_alarm = t

        epoch_score.append(score)
        epoch_alarm.append(alarm)
        epoch_mix.append(meas.mix)
        epoch_m.append(m)
        epoch_chi.append(chi)
        epoch_lor_dev.append(lor_dev)
        epoch_pos_err.append(pos_err)

    # Detection delay: first alarm at/after attack_start
    delay: int | None = None
    if attacked and first_alarm is not None and first_alarm >= cfg.attack_start:
        delay = first_alarm - cfg.attack_start

    # Hazard: pos-error exceeds threshold during attack window with no alarm
    hazard_no_alarm = 0
    if attacked:
        attack_errs = epoch_pos_err[cfg.attack_start : cfg.attack_end + 1]
        if max(attack_errs) > cfg.hazard_pos and delay is None:
            hazard_no_alarm = 1

    pvt_arr = np.array(epoch_pos_err, dtype=float)
    run_result = MSRunResult(
        score_max=float(max(epoch_score)),
        alarm_any=any(epoch_alarm),
        delay=delay,
        pvt_rmse=float(np.sqrt(np.mean(pvt_arr**2))),
        pvt_max=float(pvt_arr.max()),
        hazard_no_alarm=hazard_no_alarm,
        trace=MSRunTrace(
            score=epoch_score,
            alarm=epoch_alarm,
            mix=epoch_mix,
            m=epoch_m,
            chi=epoch_chi,
            lor_dev=epoch_lor_dev,
            pos_err=epoch_pos_err,
        ),
    )
    return _TrialSummary(
        run_result=run_result,
        score_max=float(max(epoch_score)),
        label=int(attacked),
    )


# ---------------------------------------------------------------------------
# Monte Carlo simulation
# ---------------------------------------------------------------------------


def run_ms_simulation(
    config: MultiSensorConfig | None = None,
    rng: np.random.Generator | None = None,
) -> MSSimReport:
    """Run Monte Carlo multi-sensor GNSS spoofing detection simulation.

    Runs n_nominal genuine trials followed by n_attack attacked trials.
    ROC curve computed from per-trial score_max vs label.
    P_FA / P_D computed from per-trial alarm_any at the fixed detect_threshold.

    Args:
        config: Simulation parameters; uses MultiSensorConfig() defaults if None.
        rng:    NumPy Generator; constructed from config.random_seed if None.

    Returns:
        MSSimReport with full metrics, ROC curve, and per-run results.
    """
    if config is None:
        config = MultiSensorConfig()
    if rng is None:
        rng = np.random.default_rng(config.random_seed)

    all_score_max: list[float] = []
    all_labels: list[int] = []
    nom_runs: list[MSRunResult] = []
    att_runs: list[MSRunResult] = []

    for _ in range(config.n_nominal):
        summary = simulate_trial_ms(False, config, rng)
        nom_runs.append(summary.run_result)
        all_score_max.append(summary.score_max)
        all_labels.append(0)

    for _ in range(config.n_attack):
        summary = simulate_trial_ms(True, config, rng)
        att_runs.append(summary.run_result)
        all_score_max.append(summary.score_max)
        all_labels.append(1)

    scores_arr = np.array(all_score_max, dtype=float)
    labels_arr = np.array(all_labels, dtype=int)

    fpr_list, tpr_list, auc_val = _compute_roc(scores_arr, labels_arr)

    # Metrics at fixed detect_threshold
    p_fa = float(np.mean([r.alarm_any for r in nom_runs]))
    p_d = float(np.mean([r.alarm_any for r in att_runs]))

    att_delays = [r.delay for r in att_runs if r.delay is not None]
    median_delay: float | None = float(np.median(att_delays)) if att_delays else None
    mean_delay: float | None = float(np.mean(att_delays)) if att_delays else None

    return MSSimReport(
        p_fa=p_fa,
        p_d=p_d,
        p_md=1.0 - p_d,
        median_delay=median_delay,
        mean_delay=mean_delay,
        auc=auc_val,
        nominal_rmse_mean=float(np.mean([r.pvt_rmse for r in nom_runs])),
        attack_rmse_mean=float(np.mean([r.pvt_rmse for r in att_runs])),
        attack_pvt_max_mean=float(np.mean([r.pvt_max for r in att_runs])),
        hazard_no_alarm_rate=float(np.mean([r.hazard_no_alarm for r in att_runs])),
        roc_fpr=fpr_list,
        roc_tpr=tpr_list,
        n_nominal=config.n_nominal,
        n_attack=config.n_attack,
        produced_at=datetime.now(timezone.utc),
        runs=nom_runs + att_runs,
    )
