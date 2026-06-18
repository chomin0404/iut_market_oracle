"""Twin Core — orchestrator and Monte Carlo experiment runner (T800).

Architecture
------------

    BayesianStateFilter  ──►  RegimeTracker  ──►  StructDepMonitor
           │                        │                      │
     PosteriorState           RegimeState           StructuralState
           └──────────────────────┴──────────────────────►│
                                                    TwinCoreDiagnosis

    MCExperimentRunner replays the full TwinCore on synthetic observation
    sequences to characterise filter performance and uncertainty bounds.

Component responsibilities
--------------------------
BayesianStateFilter
    Linear-Gaussian Kalman filter.  Maintains the posterior p(x_t | y_{1:t})
    = N(x̂_t, P_t) through standard predict-update cycles.  Reports:
    · innovation ν_t = y_t − H x̂_t⁻   (pre-update residual)
    · Mahalanobis distance  d_t = √(νᵀ S⁻¹ ν / m)  where S = H P⁻ Hᵀ + R
    · log marginal likelihood log p(y_t | y_{1:t−1})

RegimeTracker
    Two-regime HMM forward (α) recursion over the scalar per-epoch
    Mahalanobis distance summary.  Emission model per regime k:
        log d_t² / m  ~ N(μ_k, σ_k²)
    Normal (k=0): μ₀ = 0, σ₀ = 0.50  — innovations ≈ chi²-consistent
    Stressed (k=1): μ₁ = 1, σ₁ = 0.80 — innovations ≈ 2.7× expected
    Detects transitions when the MAP regime changes between epochs.

StructDepMonitor
    Builds a weighted dependency graph from the Kalman posterior covariance:
        W_ij = |P_ij| / √(P_ii · P_jj)    (absolute correlation, i ≠ j)
        L    = diag(W·1) − W               (graph Laplacian)
        λ₂   = Fiedler value               (algebraic connectivity)
    Alert fires when:
    · consecutive low-connectivity epochs ≥ streak threshold, OR
    · Frobenius graph-change rate > change threshold

MCExperimentRunner
    For trial k = 0 … n_trials−1:
        rng_k = default_rng(base_seed + k)
        x_0   ~ N(x0_mean, x0_cov)
        Simulate  (x_1, …, x_horizon)  via  x_{t+1} = F x_t + w_t
        Observe   y_t = H x_t + v_t
        Run TwinCore filter on (y_1, …, y_horizon)
    Aggregates posterior trajectories and regime probability distributions.

State vector default (d = 3):
    x[0] : log-revenue       [log JPY millions]
    x[1] : annual growth rate [decimal]
    x[2] : log-volatility    [log decimal]

Anomaly score fusion
--------------------
    score_t = clip(w_M · tanh(d_t) + w_R · P(stressed_t) + w_S · alert_t, 0, 1)
    Default weights: w_M = 0.40, w_R = 0.40, w_S = 0.20
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from twin.components import (
    BayesianStateFilter,
    RegimeTracker,
    StructDepMonitor,
    _build_corr_graph,
    _fiedler_value,
    _mean_clustering,
)
from twin.schemas import (
    DEFAULT_DT,
    DEFAULT_OBS_DIM,
    DEFAULT_OBS_NOISE_STD,
    DEFAULT_PROC_NOISE_STD,
    DEFAULT_STATE_DIM,
    MCExperimentConfig,
    MCExperimentResult,
    PosteriorState,
    RegimeState,
    StructuralState,
    TwinCoreDiagnosis,
)

# Re-export everything that was previously public so existing imports still work.
__all__ = [
    "DEFAULT_DT",
    "DEFAULT_OBS_DIM",
    "DEFAULT_OBS_NOISE_STD",
    "DEFAULT_PROC_NOISE_STD",
    "DEFAULT_STATE_DIM",
    "BayesianStateFilter",
    "RegimeTracker",
    "StructDepMonitor",
    "_build_corr_graph",
    "_fiedler_value",
    "_mean_clustering",
    "PosteriorState",
    "RegimeState",
    "StructuralState",
    "TwinCoreDiagnosis",
    "MCExperimentConfig",
    "MCExperimentResult",
    "TwinCore",
    "run_mc_experiment",
]

# ---------------------------------------------------------------------------
# Anomaly score fusion weights
# ---------------------------------------------------------------------------

_ANOMALY_W_MAHAL: float = 0.40
_ANOMALY_W_REGIME: float = 0.40
_ANOMALY_W_STRUCT: float = 0.20


# ---------------------------------------------------------------------------
# TwinCore — orchestrator
# ---------------------------------------------------------------------------


class TwinCore:
    """Orchestrates BayesianStateFilter → RegimeTracker → StructDepMonitor per epoch.

    Produces a TwinCoreDiagnosis per observation with a composite anomaly score:

        score_t = clip(
            w_M · tanh(mahal_t) + w_R · P(stressed_t) + w_S · struct_alert_t,
            0, 1)

    Parameters
    ----------
    state_dim, obs_dim, proc_noise_std, obs_noise_std, dt :
        Passed through to BayesianStateFilter.
    transition_matrix, obs_matrix :
        Optional overrides for F and H.
    """

    def __init__(
        self,
        state_dim: int = DEFAULT_STATE_DIM,
        obs_dim: int = DEFAULT_OBS_DIM,
        proc_noise_std: float = DEFAULT_PROC_NOISE_STD,
        obs_noise_std: float = DEFAULT_OBS_NOISE_STD,
        dt: float = DEFAULT_DT,
        transition_matrix: np.ndarray | None = None,
        obs_matrix: np.ndarray | None = None,
        regime_transition: list[list[float]] | None = None,
    ) -> None:
        self._filter = BayesianStateFilter(
            state_dim=state_dim,
            obs_dim=obs_dim,
            proc_noise_std=proc_noise_std,
            obs_noise_std=obs_noise_std,
            dt=dt,
            transition_matrix=transition_matrix,
            obs_matrix=obs_matrix,
        )
        self._regime = RegimeTracker(
            transition=regime_transition,
        )
        self._struct = StructDepMonitor()
        self._t: int = 0

    def step(self, y: np.ndarray) -> TwinCoreDiagnosis:
        """Process one observation through the full TwinCore stack.

        Parameters
        ----------
        y : (obs_dim,) observation vector
        """
        # Layer 1: Bayesian state filter
        posterior = self._filter.step(y)

        # Layer 2: Regime tracking on Mahalanobis distance
        regime = self._regime.update(posterior.mahal)

        # Layer 3: Structural dependency graph on posterior covariance
        structure = self._struct.update(posterior.cov)

        # Fusion: composite anomaly score ∈ [0, 1]
        p_stressed = float(regime.regime_probs[1]) if len(regime.regime_probs) > 1 else 0.0
        score = float(
            np.clip(
                _ANOMALY_W_MAHAL * math.tanh(posterior.mahal)
                + _ANOMALY_W_REGIME * p_stressed
                + _ANOMALY_W_STRUCT * float(structure.alert),
                0.0,
                1.0,
            )
        )
        alert = bool(regime.transition or structure.alert or posterior.mahal > 3.0)

        t = self._t
        self._t += 1
        return TwinCoreDiagnosis(
            t=t,
            posterior=posterior,
            regime=regime,
            structure=structure,
            anomaly_score=score,
            alert=alert,
        )

    def reset(self) -> None:
        """Reset all sub-components to their initial states."""
        self._filter.reset()
        self._regime.reset()
        self._struct.reset()
        self._t = 0


# ---------------------------------------------------------------------------
# Monte Carlo experiment runner
# ---------------------------------------------------------------------------


def _default_transition_matrix(d: int, dt: float) -> np.ndarray:
    """Local-linear-trend transition matrix (matches twin/simulator.py convention)."""
    F = np.eye(d, dtype=float)
    if d >= 2:
        F[0, 1] = dt
    return F


def run_mc_experiment(
    config: MCExperimentConfig | None = None,
    **kwargs: Any,
) -> MCExperimentResult:
    """Run a reproducible Monte Carlo experiment through TwinCore.

    Each trial uses rng = default_rng(config.base_seed + trial_index) so the
    results are fully reproducible regardless of call order.

    Parameters
    ----------
    config :
        MCExperimentConfig instance.  If None, a default config is constructed
        from any keyword arguments provided (e.g. n_trials=50).
    **kwargs :
        Forwarded to MCExperimentConfig if config is None.

    Returns
    -------
    MCExperimentResult
    """
    if config is None:
        config = MCExperimentConfig(**kwargs)

    d = config.state_dim
    m = config.obs_dim
    horizon = config.horizon
    n_trials = config.n_trials
    K = 2  # regime count (fixed at 2)

    F = _default_transition_matrix(d, config.dt)
    H = np.eye(m, d)
    Q = config.proc_noise_std**2 * np.eye(d)
    R = config.obs_noise_std**2 * np.eye(m)

    x0_mean = np.array(config.x0_mean, dtype=float) if config.x0_mean is not None else np.zeros(d)

    # Pre-allocate output arrays
    post_means = np.zeros((n_trials, horizon + 1, d))
    anomaly_arr = np.zeros((n_trials, horizon))
    regime_arr = np.zeros((n_trials, horizon, K))

    for k in range(n_trials):
        rng = np.random.default_rng(config.base_seed + k)

        # Sample initial state
        x = x0_mean + rng.normal(scale=config.x0_std, size=d)
        post_means[k, 0, :] = x  # record x0 as initial "posterior"

        # Instantiate a fresh TwinCore for each trial
        core = TwinCore(
            state_dim=d,
            obs_dim=m,
            proc_noise_std=config.proc_noise_std,
            obs_noise_std=config.obs_noise_std,
            dt=config.dt,
        )

        for t in range(horizon):
            # Transition
            w = rng.multivariate_normal(np.zeros(d), Q)
            x = F @ x + w

            # Observe
            v = rng.multivariate_normal(np.zeros(m), R)
            y = H @ x + v

            # Run TwinCore step
            diag = core.step(y)

            post_means[k, t + 1, :] = diag.posterior.mean
            anomaly_arr[k, t] = diag.anomaly_score
            regime_arr[k, t, :] = diag.regime.regime_probs

    mean_traj = post_means.mean(axis=0)  # (horizon+1, d)
    std_traj = post_means.std(axis=0)  # (horizon+1, d)
    final_anomaly = anomaly_arr[:, -1] if horizon > 0 else np.zeros(n_trials)

    return MCExperimentResult(
        posterior_means=post_means,
        anomaly_scores=anomaly_arr,
        regime_probs=regime_arr,
        final_anomaly=final_anomaly,
        mean_traj=mean_traj,
        std_traj=std_traj,
        config=config,
    )
