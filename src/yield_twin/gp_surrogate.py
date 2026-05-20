"""Gaussian Process surrogate for process yield prediction (T1600).

Kernel:
    k(x, x') = σ_f² · exp(−½ Σ_d (x_d − x'_d)² / l_d²)   [ARD RBF]

Additive white noise (nugget):
    K_obs = K_XX + σ_n² · Iₙ

Hyperparameter fitting:
    Minimise negative log marginal likelihood (NLML) via L-BFGS-B
    with multiple random restarts for global convergence.

Prediction:
    μ(x*) = k(x*, X) K_obs⁻¹ y
    σ²(x*) = k(x*, x*) − v(x*)ᵀ v(x*)    where v = L⁻¹ k(X, x*)

LOO cross-validation (O(n²) using Cholesky identity):
    ê_i = α_i / [K_obs⁻¹]_{ii},   σ̂²_i = 1 / [K_obs⁻¹]_{ii}
    R²_LOO = 1 − Σ ê_i² / Σ (y_i − ȳ)²

All inputs are assumed normalised to [0, 1]^d by the caller.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.linalg import LinAlgError, cho_factor, cho_solve, solve_triangular
from scipy.optimize import minimize
from scipy.stats import norm as _norm_dist

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GP_NOISE_FLOOR: float = 1e-6  # minimum σ_n² added to diagonal (jitter)
_GP_N_RESTARTS: int = 5  # random restarts for NLML optimisation
_GP_LOG_THETA_LO: float = -5.0  # lower bound on log(hyperparameter)
_GP_LOG_THETA_HI: float = 5.0  # upper bound on log(hyperparameter)
_EI_XI: float = 0.01  # exploration bonus ξ in EI

_TWO_PI: float = 2.0 * math.pi
_LOG_2PI: float = math.log(_TWO_PI)

_EPS: float = 1e-9  # numerical floor


# ---------------------------------------------------------------------------
# Internal kernel
# ---------------------------------------------------------------------------


def _rbf_kernel(
    X1: np.ndarray,
    X2: np.ndarray,
    length_scales: np.ndarray,
    signal_var: float,
) -> np.ndarray:
    """ARD RBF kernel matrix, shape (n1, n2).

    k(x, x') = σ_f² · exp(−½ Σ_d (x_d − x'_d)² / l_d²)

    Args:
        X1:            (n1, d)
        X2:            (n2, d)
        length_scales: (d,)  positive per-dimension scales
        signal_var:    σ_f² > 0
    """
    diff = X1[:, None, :] - X2[None, :, :]  # (n1, n2, d)
    sq = np.sum((diff / length_scales) ** 2, axis=2)  # (n1, n2)
    return signal_var * np.exp(-0.5 * sq)


def _nlml(
    log_theta: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
) -> float:
    """Negative log marginal likelihood as a function of log-hyperparameters.

    log_theta = [log σ_f, log l_1, …, log l_d, log σ_n]
    """
    d = X.shape[1]
    signal_var = math.exp(log_theta[0]) ** 2  # σ_f²
    length_scales = np.exp(log_theta[1 : d + 1])  # (d,)
    noise_var = math.exp(log_theta[d + 1]) ** 2  # σ_n²

    n = len(y)
    K = _rbf_kernel(X, X, length_scales, signal_var)
    K[np.diag_indices(n)] += noise_var + _GP_NOISE_FLOOR

    try:
        cf = cho_factor(K, lower=True, check_finite=False)
    except LinAlgError:
        return 1e10

    alpha = cho_solve(cf, y, check_finite=False)
    # log|K| = 2 Σ log diag(L)
    log_det = 2.0 * np.sum(np.log(np.abs(np.diag(cf[0]))))
    return float(0.5 * (y @ alpha) + 0.5 * log_det + 0.5 * n * _LOG_2PI)


# ---------------------------------------------------------------------------
# Frozen hyperparameter container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GPHyperparams:
    """Fitted GP hyperparameters."""

    signal_var: float  # σ_f²  — output variance
    noise_var: float  # σ_n²  — observation noise variance
    length_scales: tuple[float, ...]  # per-dimension (d,)

    def as_dict(self, factor_names: list[str]) -> dict[str, float]:
        """Export as named dict (keys: signal_var, noise_var, length_scale_<name>)."""
        out: dict[str, float] = {
            "signal_var": self.signal_var,
            "noise_var": self.noise_var,
        }
        for name, ls in zip(factor_names, self.length_scales):
            out[f"length_scale_{name}"] = ls
        return out


# ---------------------------------------------------------------------------
# GP Surrogate
# ---------------------------------------------------------------------------


class GPSurrogate:
    """Gaussian Process regression surrogate with ARD RBF kernel.

    Inputs must be normalised to [0, 1]^d by the caller.
    Outputs y ∈ ℝ (process yields in [0, 1] work directly).
    """

    def __init__(self, n_restarts: int = _GP_N_RESTARTS) -> None:
        self._n_restarts = n_restarts
        self._X: np.ndarray | None = None
        self._y: np.ndarray | None = None
        self._hyperparams: GPHyperparams | None = None
        # Cached Cholesky factor and solve for fast prediction
        self._cf: tuple[np.ndarray, bool] | None = None
        self._alpha: np.ndarray | None = None

    # ── Public interface ────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> None:
        """Fit GP hyperparameters by maximising log marginal likelihood.

        Args:
            X:   (n, d) normalised inputs ∈ [0, 1]^d
            y:   (n,)   target values (yields ∈ [0, 1])
            rng: Random generator for restart initialisation
        """
        n, d = X.shape
        # Parameter layout: [log σ_f, log l_1, …, log l_d, log σ_n]
        n_params = d + 2
        bounds = [(float(_GP_LOG_THETA_LO), float(_GP_LOG_THETA_HI))] * n_params

        best_nlml = float("inf")
        best_theta: np.ndarray | None = None

        for i in range(self._n_restarts):
            if i == 0:
                # Warm start: σ_f=1, l_d=0.5, σ_n=0.1
                x0 = np.zeros(n_params)
                x0[1 : d + 1] = math.log(0.5)
                x0[d + 1] = math.log(0.1)
            else:
                x0 = rng.uniform(_GP_LOG_THETA_LO, _GP_LOG_THETA_HI, size=n_params)

            try:
                res = minimize(
                    _nlml,
                    x0,
                    args=(X, y),
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={"maxiter": 200, "ftol": 1e-10},
                )
            except (ValueError, np.linalg.LinAlgError):
                continue

            if res.fun < best_nlml:
                best_nlml = res.fun
                best_theta = res.x

        if best_theta is None:
            best_theta = np.zeros(n_params)
        assert best_theta is not None

        signal_var = math.exp(best_theta[0]) ** 2
        length_scales = tuple(float(v) for v in np.exp(best_theta[1 : d + 1]))
        noise_var = math.exp(best_theta[d + 1]) ** 2

        self._hyperparams = GPHyperparams(
            signal_var=signal_var,
            noise_var=noise_var,
            length_scales=length_scales,
        )
        self._X = X.copy()
        self._y = y.copy()
        self._update_cache()

    def predict(self, X_star: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """GP posterior mean and std at X_star.

        Args:
            X_star: (m, d) normalised query points

        Returns:
            mu:    (m,) posterior mean
            sigma: (m,) posterior standard deviation  ≥ 0
        """
        if self._X is None or self._alpha is None or self._hyperparams is None or self._cf is None:
            m = len(X_star)
            return np.zeros(m), np.ones(m) * float("inf")

        hp = self._hyperparams
        ls = np.array(hp.length_scales)

        k_star = _rbf_kernel(X_star, self._X, ls, hp.signal_var)  # (m, n)
        mu = k_star @ self._alpha  # (m,)

        # σ²(x*) = k(x*,x*) − ‖L⁻¹ k(X, x*)‖²
        v = solve_triangular(self._cf[0], k_star.T, lower=True, check_finite=False)  # (n, m)
        k_star_diag = np.full(len(X_star), hp.signal_var)
        var = np.maximum(k_star_diag - np.sum(v**2, axis=0), 0.0)

        return mu, np.sqrt(var)

    def expected_improvement(
        self,
        X_star: np.ndarray,
        y_best: float,
        xi: float = _EI_XI,
    ) -> np.ndarray:
        """Expected Improvement acquisition function (maximisation).

        EI(x) = (μ(x) − y_best − ξ) · Φ(Z) + σ(x) · φ(Z)
        where Z = (μ(x) − y_best − ξ) / σ(x)

        Args:
            X_star: (m, d) normalised candidate points
            y_best: Current best observed yield
            xi:     Exploration bonus ξ ≥ 0

        Returns:
            (m,) EI values  ≥ 0
        """
        mu, sigma = self.predict(X_star)
        imp = mu - y_best - xi
        mask = sigma > _EPS
        ei = np.zeros(len(X_star))
        z = np.where(mask, imp / np.where(mask, sigma, 1.0), 0.0)
        ei[mask] = imp[mask] * _norm_dist.cdf(z[mask]) + sigma[mask] * _norm_dist.pdf(z[mask])
        return np.maximum(ei, 0.0)

    def loocv_r2(self) -> float | None:
        """GP leave-one-out cross-validated R² using the Cholesky identity.

        LOO residual:   ê_i = α_i / [K⁻¹]_{ii}
        LOO variance:   σ̂²_i = 1 / [K⁻¹]_{ii}
        R²_LOO = 1 − Σ ê_i² / Σ (y_i − ȳ)²

        Returns None if fewer than 3 observations or degenerate y.
        """
        X = self._X
        y = self._y
        alpha = self._alpha
        if X is None or len(X) < 3 or alpha is None or y is None:
            return None

        var_y = float(np.var(y))
        if var_y < _EPS:
            return None

        # K_inv diagonal via Cholesky solve on identity columns
        n = len(y)
        try:
            K_inv = cho_solve(self._cf, np.eye(n), check_finite=False)
        except LinAlgError:
            return None

        K_inv_diag = np.diag(K_inv)
        safe = K_inv_diag > _EPS
        loo_residuals = np.where(safe, alpha / K_inv_diag, y - y.mean())
        r2 = float(1.0 - np.mean(loo_residuals**2) / var_y)
        return float(np.clip(r2, -1.0, 1.0))

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def hyperparams(self) -> GPHyperparams | None:
        return self._hyperparams

    @property
    def X_train(self) -> np.ndarray | None:
        return self._X

    @property
    def y_train(self) -> np.ndarray | None:
        return self._y

    @property
    def n_obs(self) -> int:
        return len(self._X) if self._X is not None else 0

    # ── Internal helpers ────────────────────────────────────────────────────

    def _update_cache(self) -> None:
        """Recompute Cholesky factor and alpha after fit."""
        hp = self._hyperparams
        assert hp is not None
        X, y = self._X, self._y
        assert X is not None and y is not None

        ls = np.array(hp.length_scales)
        K = _rbf_kernel(X, X, ls, hp.signal_var)
        K[np.diag_indices(len(y))] += hp.noise_var + _GP_NOISE_FLOOR

        try:
            self._cf = cho_factor(K, lower=True, check_finite=False)
            self._alpha = cho_solve(self._cf, y, check_finite=False)
        except LinAlgError:
            self._cf = None
            self._alpha = np.zeros(len(y))
