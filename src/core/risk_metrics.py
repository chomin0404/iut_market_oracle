"""Risk metrics for Monte Carlo simulation outputs.

All functions operate on 1-D NumPy arrays of samples (losses or returns).
Bootstrap uses only NumPy — no scipy dependency.
"""

from __future__ import annotations

import numpy as np

# Bootstrap percentile constants for 95% CI
_BOOT_LOWER = 0.025
_BOOT_UPPER = 0.975


def compute_var(samples: np.ndarray, alpha: float) -> float:
    """Value at Risk at confidence level alpha.

    Args:
        samples: 1-D array of samples.
        alpha: Confidence level, e.g. 0.95.

    Returns:
        The alpha-quantile of the sample distribution.
    """
    return float(np.quantile(samples, alpha))


def compute_es(samples: np.ndarray, alpha: float) -> float:
    """Expected Shortfall (CVaR) at confidence level alpha.

    ES is the mean of samples that exceed VaR at alpha.

    Args:
        samples: 1-D array of samples.
        alpha: Confidence level, e.g. 0.95.

    Returns:
        Mean of samples >= VaR. Falls back to VaR if no tail samples exist.
    """
    var = compute_var(samples, alpha)
    tail = samples[samples >= var]
    if len(tail) == 0:
        return var
    return float(tail.mean())


def compute_exceedance_curve(
    samples: np.ndarray,
    thresholds: list[float],
) -> list[float]:
    """Probability of exceeding each threshold P(X > t).

    Args:
        samples: 1-D array of samples.
        thresholds: List of threshold values.

    Returns:
        List of empirical exceedance probabilities, one per threshold.
    """
    n = len(samples)
    return [float((samples > t).sum() / n) for t in thresholds]


def compute_confidence_band(
    samples: np.ndarray,
    thresholds: list[float],
    bootstrap_n: int = 500,
) -> dict[str, list[float]]:
    """Bootstrap 95% confidence band for the exceedance curve.

    Uses non-parametric bootstrap with replacement. NumPy-only implementation.

    Args:
        samples: 1-D array of samples.
        thresholds: List of threshold values.
        bootstrap_n: Number of bootstrap resamples.

    Returns:
        Dict {"lower": [...], "upper": [...]} with 2.5th and 97.5th percentile bounds.
    """
    rng = np.random.default_rng(seed=None)
    n = len(samples)
    n_thresh = len(thresholds)
    thresh_arr = np.asarray(thresholds, dtype=float)

    boot_probs = np.empty((bootstrap_n, n_thresh))
    for i in range(bootstrap_n):
        resample = rng.choice(samples, size=n, replace=True)
        # Vectorised comparison: shape (n_thresh,)
        boot_probs[i] = (resample[:, None] > thresh_arr[None, :]).sum(axis=0) / n

    lower = np.quantile(boot_probs, _BOOT_LOWER, axis=0).tolist()
    upper = np.quantile(boot_probs, _BOOT_UPPER, axis=0).tolist()
    return {"lower": lower, "upper": upper}
