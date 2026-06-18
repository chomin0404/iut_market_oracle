"""Twin Core algorithmic components (T800).

BayesianStateFilter  — linear-Gaussian Kalman filter
RegimeTracker        — 2-regime HMM forward filter
StructDepMonitor     — correlation-graph Fiedler value monitor
"""

from __future__ import annotations

import math

import numpy as np

from twin.schemas import (
    DEFAULT_DT,
    DEFAULT_OBS_DIM,
    DEFAULT_OBS_NOISE_STD,
    DEFAULT_PROC_NOISE_STD,
    DEFAULT_STATE_DIM,
    PosteriorState,
    RegimeState,
    StructuralState,
)

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

_EPS: float = 1e-12

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


# ---------------------------------------------------------------------------
# StructDepMonitor helpers
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
