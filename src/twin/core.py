"""Twin Core — Bayesian state space, regime switching, structural dependency
graph monitoring, and Monte Carlo reproducible experiment runner (T800).

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
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

_EPS: float = 1e-12

# Default state-space model (d = 3, m = 3 full observation)
DEFAULT_DT: float = 0.25  # time step [years]
DEFAULT_STATE_DIM: int = 3
DEFAULT_OBS_DIM: int = 3
DEFAULT_PROC_NOISE_STD: float = 0.05  # σ_w [√(1/year)]
DEFAULT_OBS_NOISE_STD: float = 0.10  # σ_v (observation noise)

# RegimeTracker — 2-regime HMM parameters
# Transition matrix rows: [P(stay in 0), P(transition)], [P(transition), P(stay in 1)]
_REGIME_TRANSITION: list[list[float]] = [
    [0.95, 0.05],  # normal   → [stay, stressed]
    [0.10, 0.90],  # stressed → [switch back, stay]
]
_REGIME_LOG_MU: tuple[float, float] = (0.0, 1.0)  # emission log-mean per regime
_REGIME_LOG_SIGMA: tuple[float, float] = (0.50, 0.80)  # emission log-std per regime

# StructDepMonitor — graph connectivity thresholds
_STRUCT_FIEDLER_LOW_THRESH: float = 0.10  # λ₂ below this → low-connectivity epoch
_STRUCT_STREAK_THRESH: int = 3  # consecutive low-connectivity epochs to alert
_STRUCT_CHANGE_THRESH: float = 0.50  # Frobenius change rate to alert

# Anomaly score fusion weights: Mahalanobis, Regime-stress, Structural-alert
_ANOMALY_W_MAHAL: float = 0.40
_ANOMALY_W_REGIME: float = 0.40
_ANOMALY_W_STRUCT: float = 0.20


# ---------------------------------------------------------------------------
# Result dataclasses
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


# ---------------------------------------------------------------------------
# BayesianStateFilter — linear-Gaussian Kalman filter
# ---------------------------------------------------------------------------


class BayesianStateFilter:
    """Linear-Gaussian Kalman filter tracking posterior p(x_t | y_{1:t}) = N(x̂_t, P_t).

    State transition:    x_{t+1} = F x_t + w_t,  w_t ~ N(0, Q)
    Observation model:   y_t     = H x_t + v_t,  v_t ~ N(0, R)

    F defaults to a local-linear-trend matrix (matches twin/simulator.py convention):
        F[0, 1] = dt,  all other entries on diagonal = 1
    H defaults to I_d (full observation of all state components).
    Q = σ_w² · I_d (isotropic process noise)
    R = σ_v² · I_m (isotropic observation noise)
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
    ) -> None:
        d, m = state_dim, obs_dim
        self._d = d
        self._m = m

        if transition_matrix is not None:
            if transition_matrix.shape != (d, d):
                raise ValueError(
                    f"transition_matrix must be ({d},{d}), got {transition_matrix.shape}"
                )
            self._F = transition_matrix.copy()
        else:
            self._F = np.eye(d)
            if d >= 2:
                self._F[0, 1] = dt  # local-linear trend

        if obs_matrix is not None:
            if obs_matrix.shape != (m, d):
                raise ValueError(f"obs_matrix must be ({m},{d}), got {obs_matrix.shape}")
            self._H = obs_matrix.copy()
        else:
            self._H = np.eye(m, d)  # observe first min(m,d) components

        self._Q: np.ndarray = proc_noise_std**2 * np.eye(d)
        self._R: np.ndarray = obs_noise_std**2 * np.eye(m)

        # Initial posterior: x̂_0 = 0, P_0 = I_d
        self._x_hat: np.ndarray = np.zeros(d)
        self._P: np.ndarray = np.eye(d)

    @property
    def posterior_mean(self) -> np.ndarray:
        return self._x_hat.copy()

    @property
    def posterior_cov(self) -> np.ndarray:
        return self._P.copy()

    def step(self, y: np.ndarray) -> PosteriorState:
        """Run one predict-update cycle.

        Parameters
        ----------
        y : (m,) observation vector

        Returns
        -------
        PosteriorState with posterior at this step.
        """
        y = np.asarray(y, dtype=float)
        if y.shape != (self._m,):
            raise ValueError(f"observation must be ({self._m},), got {y.shape}")

        F, H, Q, R = self._F, self._H, self._Q, self._R
        I_d = np.eye(self._d)

        # --- Predict ---
        x_pred = F @ self._x_hat
        P_pred = F @ self._P @ F.T + Q

        # --- Innovation and covariance ---
        nu = y - H @ x_pred  # (m,) innovation
        S = H @ P_pred @ H.T + R  # (m,m) innovation covariance

        # Numerical stabilise S
        S = 0.5 * (S + S.T)

        # Log-likelihood: log N(ν; 0, S)
        sign_det, log_det = np.linalg.slogdet(S)
        if sign_det > 0:
            try:
                S_inv_nu = np.linalg.solve(S, nu)
                quad = float(nu @ S_inv_nu)
            except np.linalg.LinAlgError:
                S_inv_nu = np.zeros(self._m)
                quad = float(nu @ nu)
        else:
            S_inv_nu = np.zeros(self._m)
            quad = float(nu @ nu)
            log_det = 0.0
        log_lik = -0.5 * (self._m * math.log(2.0 * math.pi) + float(log_det) + quad)

        # Normalised Mahalanobis distance √(νᵀ S⁻¹ ν / m)
        mahal = math.sqrt(max(quad / self._m, 0.0))

        # --- Update (Joseph form for numerical stability) ---
        try:
            K_T = np.linalg.solve(S, H @ P_pred)  # (m, d) — avoids explicit S⁻¹
        except np.linalg.LinAlgError:
            K_T = np.zeros((self._m, self._d))
        K = K_T.T  # (d, m) Kalman gain

        x_upd = x_pred + K @ nu
        KH = K @ H
        P_upd = (I_d - KH) @ P_pred @ (I_d - KH).T + K @ R @ K.T  # Joseph form

        self._x_hat = x_upd
        self._P = P_upd

        return PosteriorState(
            mean=x_upd.copy(),
            cov=P_upd.copy(),
            innovation=nu.copy(),
            mahal=mahal,
            log_lik=log_lik,
        )

    def reset(self) -> None:
        """Reset filter to initial state (x̂ = 0, P = I)."""
        self._x_hat = np.zeros(self._d)
        self._P = np.eye(self._d)


# ---------------------------------------------------------------------------
# RegimeTracker — 2-regime HMM forward filter
# ---------------------------------------------------------------------------


class RegimeTracker:
    """Bayesian HMM forward (α) recursion over per-epoch Mahalanobis distance.

    Emission model (log-normal on normalised Mahalanobis²):
        obs_t = log(mahal_t² + ε)  (or 0 for first epoch)
        p(obs_t | s_t = k) = N(obs_t; μ_k, σ_k²)

    Regime 0 (normal):   μ₀ = 0.0, σ₀ = 0.50  → mahal ≈ chi²-consistent
    Regime 1 (stressed): μ₁ = 1.0, σ₁ = 0.80  → mahal ≈ 2.7× expected

    Parameters
    ----------
    K : int
        Number of regimes (currently only K=2 is supported by the default
        emission constants; a larger K requires custom emission_params).
    emission_params : list of (mu, sigma) tuples
        Override per-regime emission (log-normal) parameters.
    """

    def __init__(
        self,
        K: int = 2,
        transition: list[list[float]] | None = None,
        emission_params: list[tuple[float, float]] | None = None,
    ) -> None:
        self._K = K
        Pi = np.array(transition if transition is not None else _REGIME_TRANSITION[:K], dtype=float)
        # Normalise rows to be row-stochastic
        self._Pi: np.ndarray = Pi / Pi.sum(axis=1, keepdims=True)

        if emission_params is not None:
            self._mu = np.array([p[0] for p in emission_params])
            self._sigma = np.array([p[1] for p in emission_params])
        else:
            self._mu = np.array(_REGIME_LOG_MU[:K])
            self._sigma = np.array(_REGIME_LOG_SIGMA[:K])

        # Uniform initial regime prior
        self._alpha: np.ndarray = np.ones(K) / K
        self._prev_regime: int = 0

    @property
    def regime_probs(self) -> np.ndarray:
        """Current regime posterior (K,)."""
        return self._alpha.copy()

    def update(self, mahal: float) -> RegimeState:
        """Update regime posterior with current Mahalanobis distance.

        Parameters
        ----------
        mahal : normalised Mahalanobis distance (output of BayesianStateFilter)

        Returns
        -------
        RegimeState
        """
        # Emission: log-normal observation log(mahal² + ε)
        obs = math.log(mahal**2 + _EPS)

        # Log emission probabilities per regime: log N(obs; μ_k, σ_k²)
        log_emit = -0.5 * ((obs - self._mu) / self._sigma) ** 2 - np.log(self._sigma)

        # α-recursion: predict then update
        alpha_pred = self._Pi.T @ self._alpha  # (K,) predicted regime probs
        alpha_pred = np.clip(alpha_pred, _EPS, None)
        log_alpha = np.log(alpha_pred) + log_emit
        log_alpha -= log_alpha.max()  # subtract max for stability
        alpha_new = np.exp(log_alpha)
        alpha_new /= alpha_new.sum()

        self._alpha = alpha_new
        regime = int(np.argmax(alpha_new))
        transition = regime != self._prev_regime
        self._prev_regime = regime

        return RegimeState(
            regime_probs=alpha_new.copy(),
            regime=regime,
            regime_confidence=float(alpha_new[regime]),
            transition=transition,
        )

    def reset(self) -> None:
        """Reset to uniform prior."""
        self._alpha = np.ones(self._K) / self._K
        self._prev_regime = 0


# ---------------------------------------------------------------------------
# StructDepMonitor — correlation-graph Fiedler value monitor
# ---------------------------------------------------------------------------


def _build_corr_graph(P: np.ndarray) -> np.ndarray:
    """Build absolute-correlation dependency graph from covariance P.

    W_ij = |P_ij| / √(P_ii · P_jj)  for i ≠ j,  W_ii = 0.
    """
    diag_std = np.sqrt(np.maximum(np.diag(P), _EPS))
    W = np.abs(P) / (np.outer(diag_std, diag_std) + _EPS)
    np.fill_diagonal(W, 0.0)
    return W


def _fiedler_value(W: np.ndarray) -> float:
    """Return λ₂ (Fiedler value) of the graph Laplacian L = D − W."""
    L = np.diag(W.sum(axis=1)) - W
    ev = np.sort(np.linalg.eigvalsh(L))
    return float(ev[1]) if len(ev) > 1 else 0.0


def _mean_clustering(W: np.ndarray, thresh: float = 0.30) -> float:
    """Mean clustering coefficient of the thresholded adjacency graph."""
    A = (W > thresh).astype(float)
    np.fill_diagonal(A, 0.0)
    n = W.shape[0]
    degree = A.sum(axis=1)
    cc_sum = 0.0
    n_counted = 0
    for i in range(n):
        d_i = int(degree[i])
        if d_i < 2:
            continue
        nbrs = np.where(A[i] > 0)[0]
        e = sum(A[a, b] for a in nbrs for b in nbrs if a < b)
        cc_sum += e / (d_i * (d_i - 1) / 2)
        n_counted += 1
    return float(cc_sum / max(n_counted, 1))


class StructDepMonitor:
    """Structural dependency graph monitor on the Kalman posterior covariance.

    Builds a weighted graph from absolute correlations of the state posterior,
    tracks Fiedler value (algebraic connectivity) and Frobenius change rate
    across consecutive epochs.
    """

    def __init__(
        self,
        streak_thresh: int = _STRUCT_STREAK_THRESH,
        change_thresh: float = _STRUCT_CHANGE_THRESH,
        fiedler_low_thresh: float = _STRUCT_FIEDLER_LOW_THRESH,
    ) -> None:
        self._streak_thresh = streak_thresh
        self._change_thresh = change_thresh
        self._fiedler_low = fiedler_low_thresh
        self._streak: int = 0
        self._prev_W: np.ndarray | None = None

    def update(self, P: np.ndarray) -> StructuralState:
        """Compute structural state from the current Kalman posterior covariance.

        Parameters
        ----------
        P : (d,d) posterior covariance matrix
        """
        W = _build_corr_graph(P)
        fiedler = _fiedler_value(W)
        cc = _mean_clustering(W)

        # Consecutive low-connectivity streak
        if fiedler < self._fiedler_low:
            self._streak += 1
        else:
            self._streak = 0

        # Frobenius change rate
        if self._prev_W is not None:
            frob_prev = float(np.linalg.norm(self._prev_W, "fro"))
            frob_diff = float(np.linalg.norm(W - self._prev_W, "fro"))
            change_rate = frob_diff / (frob_prev + _EPS)
        else:
            change_rate = 0.0
        self._prev_W = W.copy()

        alert = self._streak >= self._streak_thresh or change_rate > self._change_thresh
        return StructuralState(
            fiedler_value=fiedler,
            graph_change_rate=change_rate,
            clustering_coeff=cc,
            fiedler_streak=self._streak,
            alert=alert,
        )

    def reset(self) -> None:
        """Reset monitor state."""
        self._streak = 0
        self._prev_W = None


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
# MCExperimentRunner — reproducible Monte Carlo harness
# ---------------------------------------------------------------------------


def _default_transition_matrix(d: int, dt: float) -> np.ndarray:
    """Local-linear-trend transition matrix (matches twin/simulator.py convention)."""
    F = np.eye(d, dtype=float)
    if d >= 2:
        F[0, 1] = dt
    return F


def run_mc_experiment(
    config: MCExperimentConfig | None = None,
    **kwargs: object,
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
        config = MCExperimentConfig(**kwargs)  # type: ignore[arg-type]

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
