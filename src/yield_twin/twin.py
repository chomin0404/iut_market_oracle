"""Process Yield Twin — surrogate + Bayesian optimisation + D-Optimal DOE (T1600).

Architecture
------------
At each step the twin:
  1. Fits a GP surrogate (ARD RBF) to all observed (x, yield) pairs.
  2. Generates a Latin-Hypercube candidate set in [0,1]^d.
  3. Evaluates Expected Improvement (EI) on the GP posterior.
  4. Evaluates D-Optimal leverage d(x) = φ(x)ᵀ M⁻¹ φ(x) for a
     quadratic polynomial model matrix Φ ∈ ℝ^{n × p}.
  5. Fuses EI and D-leverage with a data-driven weight α(n):
       α(n) = clip((n − n_min) / n_min, 0, 1)   where n_min = max(2d+1, 5)
       score(x) = α · EI_norm(x) + (1−α) · d_norm(x)
  6. Returns the candidate with the highest fused score.

Acquisition modes:
  "doe_explore"  — n < n_min: D-optimal dominates (explore design space)
  "fused"        — n_min ≤ n < 2·n_min: weighted blend
  "ei_exploit"   — n ≥ 2·n_min: EI dominates (exploit surrogate)

Quadratic model basis for D-Optimal (p terms):
  φ(x) = [1, x₁, …, x_d, x₁², x₁x₂, …, x_d²]   p = 1 + 2d + d(d−1)/2
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from schemas import DOERecommendation, ExperimentPoint, FactorSpec, YieldTwinReport
from yield_twin.gp_surrogate import GPSurrogate

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LHS_N_CANDIDATES: int = 2000  # Latin Hypercube candidate points
_DOE_LAMBDA_REG: float = 1e-6  # regularisation for information matrix M
_MIN_OBS_FACTOR: int = 2  # n_min = _MIN_OBS_FACTOR * d + 1

_EPS: float = 1e-9


# ---------------------------------------------------------------------------
# Quadratic basis for D-Optimal DOE
# ---------------------------------------------------------------------------


def _quadratic_basis(x: np.ndarray) -> np.ndarray:
    """Quadratic polynomial basis vector φ(x) ∈ ℝ^p.

    Terms: [1, x₁, …, x_d, x₁², x₁x₂, …, x_d²]  (upper-triangle order)
    p = 1 + d + d(d+1)/2

    Args:
        x: (d,) normalised input ∈ [0, 1]^d
    """
    phi = [1.0]
    phi.extend(float(xi) for xi in x)
    for i in range(len(x)):
        for j in range(i, len(x)):
            phi.append(float(x[i] * x[j]))
    return np.array(phi)


def _basis_matrix(X_norm: np.ndarray) -> np.ndarray:
    """Build model matrix Φ ∈ ℝ^{n × p} from normalised design X ∈ [0,1]^{n×d}."""
    return np.vstack([_quadratic_basis(row) for row in X_norm])


def _d_leverages(
    X_candidates: np.ndarray,
    M_inv: np.ndarray,
) -> np.ndarray:
    """D-Optimal leverage scores φ(x)ᵀ M⁻¹ φ(x) for all candidates.

    Args:
        X_candidates: (m, d) normalised candidate points
        M_inv:        (p, p) inverse information matrix (regularised)

    Returns:
        (m,) non-negative leverage scores
    """
    m = len(X_candidates)
    leverages = np.zeros(m)
    for i in range(m):
        phi = _quadratic_basis(X_candidates[i])
        leverages[i] = float(phi @ M_inv @ phi)
    return np.maximum(leverages, 0.0)


def _build_info_matrix_inv(X_norm: np.ndarray, d: int) -> np.ndarray:
    """Build regularised inverse Fisher information matrix (M + λI)⁻¹.

    M = ΦᵀΦ,  Φ ∈ ℝ^{n×p}
    Regularised so M is always invertible, even when n < p.
    """
    p = 1 + d + d * (d + 1) // 2
    if len(X_norm) == 0:
        return np.eye(p) / _DOE_LAMBDA_REG
    Phi = _basis_matrix(X_norm)
    M = Phi.T @ Phi + _DOE_LAMBDA_REG * np.eye(p)
    return np.linalg.inv(M)


# ---------------------------------------------------------------------------
# Latin Hypercube Sampling
# ---------------------------------------------------------------------------


def _lhs_candidates(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Latin Hypercube Sample of n points in [0, 1]^d.

    Each dimension is divided into n equal strata; one point is drawn
    uniformly from each stratum, then dimensions are independently shuffled.

    Args:
        n: Number of sample points
        d: Number of dimensions

    Returns:
        (n, d) array with entries in [0, 1]
    """
    X = np.zeros((n, d))
    for j in range(d):
        perm = rng.permutation(n)
        X[:, j] = (perm + rng.uniform(size=n)) / n
    return X


# ---------------------------------------------------------------------------
# Factor normalisation helpers
# ---------------------------------------------------------------------------


def _normalise(x_phys: np.ndarray, factor_specs: list[FactorSpec]) -> np.ndarray:
    """Map physical units → [0, 1]^d."""
    lo = np.array([fs.low for fs in factor_specs])
    hi = np.array([fs.high for fs in factor_specs])
    return (x_phys - lo) / (hi - lo)


def _denormalise(x_norm: np.ndarray, factor_specs: list[FactorSpec]) -> np.ndarray:
    """Map [0, 1]^d → physical units."""
    lo = np.array([fs.low for fs in factor_specs])
    hi = np.array([fs.high for fs in factor_specs])
    return lo + x_norm * (hi - lo)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class YieldTwinConfig:
    """Parameters for the Process Yield Twin.

    Attributes:
        factor_specs:     Factor definitions (name, low, high) — required.
        n_candidates:     Candidate points per recommendation cycle.
        gp_n_restarts:    GP hyperparameter optimisation random restarts.
        ei_xi:            Exploration bonus ξ for Expected Improvement.
        doe_lambda_reg:   Regularisation for D-Optimal information matrix.
        random_seed:      RNG seed.
    """

    factor_specs: list[FactorSpec]
    n_candidates: int = _LHS_N_CANDIDATES
    gp_n_restarts: int = 5
    ei_xi: float = 0.01
    doe_lambda_reg: float = _DOE_LAMBDA_REG
    random_seed: int = 42

    def __post_init__(self) -> None:
        if not self.factor_specs:
            raise ValueError("factor_specs must not be empty")
        if self.n_candidates < 2:
            raise ValueError("n_candidates must be >= 2")

    @property
    def n_factors(self) -> int:
        return len(self.factor_specs)

    @property
    def n_min_ei(self) -> int:
        """Minimum observations before EI fully activates."""
        return max(_MIN_OBS_FACTOR * self.n_factors + 1, 5)


# ---------------------------------------------------------------------------
# Process Yield Twin
# ---------------------------------------------------------------------------


class ProcessYieldTwin:
    """Digital twin for process yield optimisation.

    Combines GP surrogate (ARD RBF) with Bayesian optimisation (EI)
    and sequential D-Optimal DOE to recommend the next experiment.

    Usage::

        twin = ProcessYieldTwin(config)
        twin.observe({"temp": 180.0, "pressure": 2.5}, yield_obs=0.82)
        report = twin.report()
        print(report.recommendation.factors)
    """

    def __init__(self, config: YieldTwinConfig) -> None:
        self._config = config
        self._rng = np.random.default_rng(config.random_seed)
        self._surrogate = GPSurrogate(n_restarts=config.gp_n_restarts)
        # Normalised observations
        self._X_norm: list[np.ndarray] = []
        self._y_obs: list[float] = []

    # ── Observation ─────────────────────────────────────────────────────────

    def observe(self, factors: dict[str, float], yield_obs: float) -> None:
        """Record an experiment result.

        Args:
            factors:   Factor name → value in physical units.
            yield_obs: Observed yield ∈ [0, 1].
        """
        x_phys = self._dict_to_array(factors)
        x_norm = _normalise(x_phys, self._config.factor_specs)
        x_norm = np.clip(x_norm, 0.0, 1.0)
        self._X_norm.append(x_norm)
        self._y_obs.append(float(yield_obs))
        self._refit()

    def observe_batch(self, points: list[ExperimentPoint]) -> None:
        """Record multiple experiment results."""
        for pt in points:
            if pt.yield_obs is not None:
                self.observe(pt.factors, pt.yield_obs)

    # ── Recommendation ───────────────────────────────────────────────────────

    def recommend(self) -> DOERecommendation:
        """Recommend the next experiment to run.

        Generates a Latin-Hypercube candidate set, scores each candidate
        with a fused EI + D-leverage criterion, and returns the best.
        """
        cfg = self._config
        d = cfg.n_factors
        n_obs = len(self._y_obs)
        n_min = cfg.n_min_ei

        candidates = _lhs_candidates(cfg.n_candidates, d, self._rng)  # (m, d)

        # ── D-Optimal leverage scores ────────────────────────────────────────
        X_norm_arr = np.array(self._X_norm) if self._X_norm else np.empty((0, d))
        M_inv = _build_info_matrix_inv(X_norm_arr, d)
        d_scores = _d_leverages(candidates, M_inv)

        # ── EI scores ───────────────────────────────────────────────────────
        y_best = max(self._y_obs) if self._y_obs else 0.0
        if self._surrogate.n_obs >= 2:
            ei_scores = self._surrogate.expected_improvement(candidates, y_best, xi=cfg.ei_xi)
        else:
            ei_scores = np.zeros(len(candidates))

        # ── Fusion weight α(n) ──────────────────────────────────────────────
        if n_obs < n_min:
            alpha = 0.0
            mode = "doe_explore"
        elif n_obs < 2 * n_min:
            alpha = (n_obs - n_min) / n_min
            mode = "fused"
        else:
            alpha = 1.0
            mode = "ei_exploit"

        # Normalise each score to [0, 1]
        ei_norm = _safe_normalise(ei_scores)
        d_norm = _safe_normalise(d_scores)
        fused = alpha * ei_norm + (1.0 - alpha) * d_norm

        best_idx = int(np.argmax(fused))
        x_best_norm = candidates[best_idx]
        x_best_phys = _denormalise(x_best_norm, cfg.factor_specs)

        mu_best, sigma_best = self._surrogate.predict(x_best_norm[None, :])

        return DOERecommendation(
            factors=self._array_to_dict(x_best_phys),
            expected_improvement=float(ei_scores[best_idx]),
            d_leverage=float(d_scores[best_idx]),
            fusion_score=float(fused[best_idx]),
            predicted_yield=float(np.clip(mu_best[0], 0.0, 1.0)),
            predicted_std=float(sigma_best[0]),
            acquisition_mode=mode,
            n_observations=n_obs,
        )

    # ── Report ───────────────────────────────────────────────────────────────

    def report(self) -> YieldTwinReport:
        """Generate the full optimisation report."""
        recommendation = self.recommend()
        n_obs = len(self._y_obs)

        best_yield: float | None = None
        best_factors: dict[str, float] | None = None
        if self._y_obs:
            best_idx = int(np.argmax(self._y_obs))
            best_yield = float(self._y_obs[best_idx])
            x_best_phys = _denormalise(self._X_norm[best_idx], self._config.factor_specs)
            best_factors = self._array_to_dict(x_best_phys)

        gp_hp: dict[str, float] = {}
        if self._surrogate.hyperparams is not None:
            gp_hp = self._surrogate.hyperparams.as_dict(
                [fs.name for fs in self._config.factor_specs]
            )

        return YieldTwinReport(
            n_observations=n_obs,
            best_yield_observed=best_yield,
            best_factors=best_factors,
            surrogate_loocv_r2=self._surrogate.loocv_r2(),
            recommendation=recommendation,
            gp_hyperparams=gp_hp,
            factor_specs=self._config.factor_specs,
        )

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _refit(self) -> None:
        """Refit GP surrogate on all observations (when ≥ 2 data points)."""
        n = len(self._y_obs)
        if n < 2:
            return
        X = np.array(self._X_norm)  # (n, d)
        y = np.array(self._y_obs)  # (n,)
        self._surrogate.fit(X, y, self._rng)

    def _dict_to_array(self, factors: dict[str, float]) -> np.ndarray:
        return np.array([factors[fs.name] for fs in self._config.factor_specs])

    def _array_to_dict(self, x: np.ndarray) -> dict[str, float]:
        return {fs.name: float(x[i]) for i, fs in enumerate(self._config.factor_specs)}


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _safe_normalise(v: np.ndarray) -> np.ndarray:
    """Normalise array to [0, 1]; return uniform 0.5 if constant."""
    vmin, vmax = v.min(), v.max()
    if vmax - vmin < _EPS:
        return np.full_like(v, 0.5)
    return (v - vmin) / (vmax - vmin)


# ---------------------------------------------------------------------------
# Convenience: run one full recommendation cycle from a batch of points
# ---------------------------------------------------------------------------


def recommend_next_experiment(
    factor_specs: list[FactorSpec],
    observations: list[ExperimentPoint],
    *,
    random_seed: int = 42,
    n_candidates: int = _LHS_N_CANDIDATES,
) -> YieldTwinReport:
    """One-shot function: fit twin on observations, return next recommendation.

    Args:
        factor_specs:  Factor definitions.
        observations:  Past experiments with observed yields.
        random_seed:   RNG seed.
        n_candidates:  LHS candidate set size.

    Returns:
        YieldTwinReport containing the recommendation and surrogate diagnostics.
    """
    config = YieldTwinConfig(
        factor_specs=factor_specs,
        random_seed=random_seed,
        n_candidates=n_candidates,
    )
    twin = ProcessYieldTwin(config)
    twin.observe_batch(observations)
    return twin.report()
