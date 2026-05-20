"""Copula-based Monte Carlo simulator.

Supported copulas: gaussian
Supported marginal distributions: all scipy.stats distributions listed in _DIST_MAP.

Algorithm (Gaussian copula):
    1. Draw Z ~ N(0, I)  shape (n_samples, n_vars)
    2. Correlate: X = Z @ chol(corr_matrix).T
    3. Map to uniform: U = Phi(X)  via standard-normal CDF
    4. Apply inverse CDF of each marginal: Y_i = F_i^{-1}(U_i)
"""

from __future__ import annotations

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
}


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
