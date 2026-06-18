"""Twin Core result schemas and experiment configuration (T800).

Defines the frozen dataclasses that carry per-epoch diagnostic output and
the mutable configuration dataclass for Monte Carlo experiments.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Default state-space model dimensions and noise parameters
# ---------------------------------------------------------------------------

DEFAULT_DT: float = 0.25  # time step [years]
DEFAULT_STATE_DIM: int = 3
DEFAULT_OBS_DIM: int = 3
DEFAULT_PROC_NOISE_STD: float = 0.05  # σ_w [√(1/year)]
DEFAULT_OBS_NOISE_STD: float = 0.10  # σ_v (observation noise)


# ---------------------------------------------------------------------------
# Per-epoch result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PosteriorState:
    """Output of BayesianStateFilter per epoch.

    mean     : (d,) posterior mean x̂_t
    cov      : (d,d) posterior covariance P_t
    innovation   : (m,) pre-update residual ν_t = y_t − H x̂_t⁻
    mahal    : normalised Mahalanobis distance √(νᵀ S⁻¹ ν / m)
    log_lik  : log p(y_t | y_{1:t−1})  (marginal likelihood contribution)
    """

    mean: np.ndarray  # (d,)
    cov: np.ndarray  # (d,d)
    innovation: np.ndarray  # (m,)
    mahal: float
    log_lik: float


@dataclass(frozen=True)
class RegimeState:
    """Output of RegimeTracker per epoch.

    regime_probs      : (K,) P(s_t = k | y_{1:t})
    regime            : int   MAP regime index
    regime_confidence : float max(regime_probs)
    transition        : bool  MAP regime changed from previous epoch
    """

    regime_probs: np.ndarray  # (K,)
    regime: int
    regime_confidence: float
    transition: bool


@dataclass(frozen=True)
class StructuralState:
    """Output of StructDepMonitor per epoch.

    fiedler_value     : λ₂ of the correlation-graph Laplacian
    graph_change_rate : ‖W_t − W_{t−1}‖_F / (‖W_{t−1}‖_F + ε)
    clustering_coeff  : mean clustering coefficient of thresholded graph
    fiedler_streak    : consecutive epochs with λ₂ < threshold
    alert             : True if streak ≥ thresh OR change_rate > thresh
    """

    fiedler_value: float
    graph_change_rate: float
    clustering_coeff: float
    fiedler_streak: int
    alert: bool


@dataclass(frozen=True)
class TwinCoreDiagnosis:
    """Per-epoch diagnostic output from TwinCore.

    anomaly_score : composite ∈ [0, 1]
    alert         : any sub-component raised an alert
    """

    t: int
    posterior: PosteriorState
    regime: RegimeState
    structure: StructuralState
    anomaly_score: float  # ∈ [0, 1]
    alert: bool


# ---------------------------------------------------------------------------
# Monte Carlo experiment configuration and results
# ---------------------------------------------------------------------------


@dataclass
class MCExperimentConfig:
    """Parameters for a TwinCore MC reproducibility experiment.

    n_trials   : number of independent Monte Carlo trials
    horizon    : number of forward simulation steps per trial
    state_dim  : d (state vector dimension)
    obs_dim    : m (observation dimension)
    proc_noise_std : σ_w  process noise 1-σ
    obs_noise_std  : σ_v  observation noise 1-σ
    base_seed  : RNG base seed; trial k uses base_seed + k
    x0_mean    : (d,) initial state mean; zeros if None
    x0_std     : initial state 1-σ for each component (scalar or (d,))
    dt         : time step for the transition matrix
    """

    n_trials: int = 100
    horizon: int = 40
    state_dim: int = DEFAULT_STATE_DIM
    obs_dim: int = DEFAULT_OBS_DIM
    proc_noise_std: float = DEFAULT_PROC_NOISE_STD
    obs_noise_std: float = DEFAULT_OBS_NOISE_STD
    base_seed: int = 42
    x0_mean: list[float] | None = None
    x0_std: float = 0.10
    dt: float = DEFAULT_DT


@dataclass(frozen=True)
class MCExperimentResult:
    """Result of a TwinCore MC experiment.

    posterior_means  : (n_trials, horizon+1, d) posterior mean trajectories
    anomaly_scores   : (n_trials, horizon) per-step anomaly scores
    regime_probs     : (n_trials, horizon, K) regime posteriors
    final_anomaly    : (n_trials,) anomaly score at t = horizon
    mean_traj        : (horizon+1, d) mean posterior trajectory across trials
    std_traj         : (horizon+1, d) std of posterior trajectory across trials
    """

    posterior_means: np.ndarray  # (n_trials, horizon+1, d)
    anomaly_scores: np.ndarray  # (n_trials, horizon)
    regime_probs: np.ndarray  # (n_trials, horizon, K)
    final_anomaly: np.ndarray  # (n_trials,)
    mean_traj: np.ndarray  # (horizon+1, d)
    std_traj: np.ndarray  # (horizon+1, d)
    config: MCExperimentConfig
