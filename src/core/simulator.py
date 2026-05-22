"""Copula-based Monte Carlo simulator.

Supported copulas: gaussian, student_t, clayton, independent
Supported marginal distributions: all scipy.stats distributions listed in _DIST_MAP.

Algorithm (Gaussian copula):
    1. Draw Z ~ N(0, I)  shape (n_samples, n_vars)
    2. Correlate: X = Z @ chol(corr_matrix).T
    3. Map to uniform: U = Phi(X)  via standard-normal CDF
    4. Apply inverse CDF of each marginal: Y_i = F_i^{-1}(U_i)

Algorithm (Student-T copula):
    1-2. Same as Gaussian copula to get correlated Z
    3. Draw W ~ chi2(df); T = Z / sqrt(W/df)  (multivariate t)
    4. Map to uniform: U = t_{df}.cdf(T_i)
    5. Apply inverse CDF of each marginal

Algorithm (Clayton copula, frailty method):
    1. V ~ Gamma(1/theta, scale=1)
    2. E_i ~ Exp(1) iid for i=1,...,d
    3. U_i = (1 + E_i / V)^{-1/theta}
    4. Apply inverse CDF of each marginal

Algorithm (independent copula):
    U ~ Uniform(0, 1) drawn independently
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.stats as st

# Supported distributions and their scipy.stats equivalents
_DIST_MAP: dict[str, Any] = {
    "normal": st.norm,
    "lognormal": st.lognorm,
    "t": st.t,
    "uniform": st.uniform,
    "gev": st.genextreme,
    "expon": st.expon,
    "beta": st.beta,
    "gamma": st.gamma,
    "weibull": st.weibull_min,
}

MAX_N_SAMPLES = 100_000


@dataclass
class MCSimulationResult:
    """Result of a Monte Carlo simulation.

    Attributes:
        samples: Simulated values, shape (n_vars, n_samples).
        n_samples: Number of samples drawn.
        seed_used: Random seed used; None if non-deterministic.
    """

    samples: np.ndarray  # shape (n_vars, n_samples)
    n_samples: int
    seed_used: int | None


def _make_frozen(dist_spec: dict[str, Any]) -> Any:
    """Build a frozen scipy.stats distribution from a spec dict.

    Args:
        dist_spec: {"name": str, "params": {kwarg: value, ...}}

    Returns:
        Frozen scipy.stats distribution with .ppf() available.

    Raises:
        ValueError: If the distribution name is not supported.
    """
    name = dist_spec["name"]
    if name not in _DIST_MAP:
        supported = sorted(_DIST_MAP.keys())
        raise ValueError(f"Unsupported distribution: {name!r}. Supported: {supported}")
    params = dist_spec.get("params", {})
    return _DIST_MAP[name](**params)


def simulate_gaussian_copula(
    n_vars: int,
    n_samples: int,
    distributions: list[dict[str, Any]],
    corr_matrix: list[list[float]],
    seed: int | None = None,
) -> np.ndarray:
    """Sample from a Gaussian copula with specified marginal distributions.

    Args:
        n_vars: Number of variables. Must match len(distributions) and corr_matrix size.
        n_samples: Number of MC samples to draw.
        distributions: List of dicts [{"name": ..., "params": {...}}].
        corr_matrix: n_vars x n_vars positive-definite correlation matrix.
        seed: Random seed for reproducibility (None = non-deterministic).

    Returns:
        np.ndarray of shape (n_samples, n_vars) with samples from the joint distribution.

    Raises:
        ValueError: If corr_matrix is not positive definite or distributions are invalid.
    """
    rng = np.random.default_rng(seed)
    corr = np.asarray(corr_matrix, dtype=float)

    if corr.shape != (n_vars, n_vars):
        raise ValueError(f"corr_matrix shape {corr.shape} does not match n_vars={n_vars}")

    # Cholesky decomposition — fails if matrix is not positive definite
    try:
        chol = np.linalg.cholesky(corr)
    except np.linalg.LinAlgError as exc:
        raise ValueError("corr_matrix must be symmetric positive definite.") from exc

    # Step 1-3: correlated uniforms via Gaussian copula
    z = rng.standard_normal((n_samples, n_vars))
    correlated = z @ chol.T  # shape (n_samples, n_vars)
    uniforms = st.norm.cdf(correlated)  # shape (n_samples, n_vars); values in (0, 1)

    # Step 4: apply each marginal's inverse CDF
    result = np.empty((n_samples, n_vars), dtype=float)
    for i, dist_spec in enumerate(distributions):
        frozen = _make_frozen(dist_spec)
        result[:, i] = frozen.ppf(uniforms[:, i])

    return result


class MonteCarloSimulator:
    """Copula-based Monte Carlo simulator supporting multiple copula families.

    Supported copulas: gaussian, student_t, clayton, independent.
    Supported marginals: all distributions in _DIST_MAP.
    """

    def simulate(
        self,
        n_vars: int,
        n_samples: int,
        distributions: list[dict[str, Any]],
        copula: dict[str, Any],
        seed: int | None = None,
    ) -> MCSimulationResult:
        """Run a Monte Carlo simulation.

        Args:
            n_vars: Number of variables.
            n_samples: Number of samples (max 100_000).
            distributions: Marginal distribution specs, one per variable.
                Each spec: {"name": str, "params": {kwarg: value, ...}}.
            copula: Copula configuration dict.
                gaussian:  {"type": "gaussian", "corr_matrix": [[...]]}
                student_t: {"type": "student_t", "corr_matrix": [[...]], "df": float}
                clayton:   {"type": "clayton", "theta": float}
                independent: {"type": "independent"}
            seed: Random seed for reproducibility.

        Returns:
            SimulationResult with samples of shape (n_vars, n_samples).

        Raises:
            ValueError: If n_samples > 100_000 or copula type is unsupported.
        """
        if n_samples > MAX_N_SAMPLES:
            raise ValueError(f"n_samples={n_samples} exceeds the limit of {MAX_N_SAMPLES}.")

        rng = np.random.default_rng(seed)
        copula_type = copula.get("type", "gaussian")

        if copula_type == "independent":
            uniforms = rng.uniform(size=(n_samples, n_vars))
        elif copula_type == "gaussian":
            corr = np.asarray(copula["corr_matrix"], dtype=float)
            uniforms = self._gaussian_uniforms(n_vars, n_samples, corr, rng)
        elif copula_type == "student_t":
            corr = np.asarray(copula["corr_matrix"], dtype=float)
            df = float(copula["df"])
            uniforms = self._student_t_uniforms(n_vars, n_samples, corr, df, rng)
        elif copula_type == "clayton":
            theta = float(copula["theta"])
            uniforms = self._clayton_uniforms(n_vars, n_samples, theta, rng)
        else:
            raise ValueError(
                f"Unsupported copula type: {copula_type!r}. "
                "Supported: gaussian, student_t, clayton, independent."
            )

        # Apply each marginal's inverse CDF
        result = np.empty((n_samples, n_vars), dtype=float)
        for i, dist_spec in enumerate(distributions):
            frozen = _make_frozen(dist_spec)
            result[:, i] = frozen.ppf(uniforms[:, i])

        return MCSimulationResult(
            samples=result.T,  # shape (n_vars, n_samples)
            n_samples=n_samples,
            seed_used=seed,
        )

    # ------------------------------------------------------------------
    # Private copula helpers — all return uniforms of shape (n_samples, n_vars)
    # ------------------------------------------------------------------

    @staticmethod
    def _gaussian_uniforms(
        n_vars: int,
        n_samples: int,
        corr: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Correlated uniforms via Gaussian copula (Cholesky decomposition)."""
        if corr.shape != (n_vars, n_vars):
            raise ValueError(f"corr_matrix shape {corr.shape} does not match n_vars={n_vars}")
        try:
            chol = np.linalg.cholesky(corr)
        except np.linalg.LinAlgError as exc:
            raise ValueError("corr_matrix must be symmetric positive definite.") from exc

        z = rng.standard_normal((n_samples, n_vars))
        correlated = z @ chol.T  # (n_samples, n_vars)
        return st.norm.cdf(correlated)

    @staticmethod
    def _student_t_uniforms(
        n_vars: int,
        n_samples: int,
        corr: np.ndarray,
        df: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Correlated uniforms via Student-T copula.

        Algorithm:
            X = Z @ chol.T (correlated normal)
            T = X / sqrt(W / df),  W ~ chi2(df)
            U_i = t_{df}.cdf(T_i)
        """
        if corr.shape != (n_vars, n_vars):
            raise ValueError(f"corr_matrix shape {corr.shape} does not match n_vars={n_vars}")
        try:
            chol = np.linalg.cholesky(corr)
        except np.linalg.LinAlgError as exc:
            raise ValueError("corr_matrix must be symmetric positive definite.") from exc

        z = rng.standard_normal((n_samples, n_vars))
        correlated = z @ chol.T  # (n_samples, n_vars)
        chi2 = rng.chisquare(df, size=n_samples)  # (n_samples,)
        t_samples = correlated / np.sqrt(chi2 / df)[:, None]  # (n_samples, n_vars)
        return st.t.cdf(t_samples, df=df)

    @staticmethod
    def _clayton_uniforms(
        n_vars: int,
        n_samples: int,
        theta: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Correlated uniforms via Clayton copula (frailty method).

        The Clayton Archimedean copula generator is phi(t) = t^{-theta} - 1.
        Frailty algorithm (valid for any d >= 1, theta > 0):
            V   ~ Gamma(shape=1/theta, scale=1)
            E_i ~ Exp(1)  iid,  i = 1,...,d
            U_i = (1 + E_i / V)^{-1/theta}

        Reference: Nelsen (2006), Introduction to Copulas, Ch. 4.
        """
        if theta <= 0:
            raise ValueError(f"Clayton copula requires theta > 0, got {theta}.")

        # V shape: (n_samples,); E shape: (n_samples, n_vars)
        v = rng.gamma(shape=1.0 / theta, scale=1.0, size=n_samples)
        e = rng.exponential(scale=1.0, size=(n_samples, n_vars))
        return (1.0 + e / v[:, None]) ** (-1.0 / theta)
