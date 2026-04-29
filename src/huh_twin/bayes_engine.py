"""Bayesian engine for the HuhTwin digital-twin framework.

Provides Normal-Normal conjugate updates, regime posterior inference,
and YAML-driven prior configuration loading.

Design note
-----------
A second Normal-Normal implementation exists in ``src/bayesian/updater.py``
(the API/web layer).  That version accepts ``Evidence`` Pydantic schemas with
precision-weighted observations (``weight = 1/σ_likelihood²``).  This module
uses a count-based interface (``observations: list[float]`` + known
``observation_std``) suited to HuhTwin's internal simulation loop.  The two
are intentionally kept separate to avoid coupling the web schema layer to the
simulation internals.

Mathematical conventions
------------------------
Normal-Normal conjugate update
    Prior     : N(μ₀, σ₀²)
    Likelihood: xᵢ ~ N(μ, σ_obs²)   (σ_obs known)
    Posterior precision : τ_n = 1/σ₀² + n/σ_obs²
    Posterior mean      : μ_n = (μ₀/σ₀² + Σxᵢ/σ_obs²) / τ_n
    Posterior std       : σ_n = 1/√τ_n

Regime posterior (Bayes' theorem, discrete)
    p(r | data) ∝ p(data | r) · p(r)
    Normalized so that Σ_r p(r | data) = 1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GaussianPrior:
    mean: float
    std: float


@dataclass(frozen=True)
class PriorConfig:
    priors: dict[str, GaussianPrior]
    default_observation_std: float
    observation_std_overrides: dict[str, float]
    regime_priors: dict[str, float]


@dataclass(frozen=True)
class UpdateSummary:
    prior_mean: float
    prior_std: float
    posterior_mean: float
    posterior_std: float
    observation_count: int


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def normal_normal_update(
    prior_mean: float,
    prior_std: float,
    observations: list[float],
    observation_std: float,
) -> tuple[float, float]:
    """Normal-Normal conjugate update with known observation variance.

    Parameters
    ----------
    prior_mean:
        μ₀ — prior mean.
    prior_std:
        σ₀ — prior standard deviation (> 0).
    observations:
        Observed data points. Empty list returns the prior unchanged.
    observation_std:
        σ_obs — known observation standard deviation (> 0).

    Returns
    -------
    (posterior_mean, posterior_std)
    """
    if not observations:
        return prior_mean, prior_std

    tau0 = 1.0 / prior_std**2  # prior precision
    tau_obs = 1.0 / observation_std**2  # observation precision
    n = len(observations)

    tau_n = tau0 + n * tau_obs  # posterior precision
    mu_n = (prior_mean * tau0 + sum(observations) * tau_obs) / tau_n
    sigma_n = math.sqrt(1.0 / tau_n)

    return mu_n, sigma_n


def regime_posterior(
    prior_weights: dict[str, float],
    likelihoods: dict[str, float],
) -> dict[str, float]:
    """Compute normalized posterior weights over regimes.

    p(regime | data) ∝ p(data | regime) · p(regime)

    Parameters
    ----------
    prior_weights:
        Mapping regime → prior probability (must sum to 1, not validated).
    likelihoods:
        Mapping regime → p(data | regime).

    Returns
    -------
    dict[str, float]
        Posterior weights normalized to sum exactly to 1.
    """
    unnormalized = {regime: prior_weights[regime] * likelihoods[regime] for regime in prior_weights}
    total = sum(unnormalized.values())
    return {regime: w / total for regime, w in unnormalized.items()}


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_prior_config(config_path: Path) -> PriorConfig:
    """Load PriorConfig from a YAML file.

    Expected YAML structure::

        priors:
          growth_rate:
            mean: 0.10
            std: 0.03

        observation_model:
          default_std: 0.05
          overrides:
            growth_rate: 0.02

        regimes:
          normal: 0.60
          high_growth: 0.25
          disruption: 0.15

    Parameters
    ----------
    config_path:
        Path to the YAML configuration file.
    """
    raw: dict = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    priors = {
        name: GaussianPrior(mean=float(spec["mean"]), std=float(spec["std"]))
        for name, spec in raw["priors"].items()
    }

    obs_model: dict = raw.get("observation_model", {})
    default_std = float(obs_model.get("default_std", 0.05))
    overrides = {k: float(v) for k, v in obs_model.get("overrides", {}).items()}

    regime_priors = {k: float(v) for k, v in raw.get("regimes", {}).items()}

    return PriorConfig(
        priors=priors,
        default_observation_std=default_std,
        observation_std_overrides=overrides,
        regime_priors=regime_priors,
    )


# ---------------------------------------------------------------------------
# Batch update
# ---------------------------------------------------------------------------


def update_named_priors(
    config: PriorConfig,
    observations_by_name: dict[str, list[float]],
) -> dict[str, UpdateSummary]:
    """Apply Normal-Normal conjugate updates for each named prior.

    Parameters
    ----------
    config:
        Prior configuration loaded via :func:`load_prior_config`.
    observations_by_name:
        Mapping prior-name → list of observed values.

    Returns
    -------
    dict[str, UpdateSummary]
        Posterior summary for each name in *observations_by_name*.
    """
    summaries: dict[str, UpdateSummary] = {}

    for name, observations in observations_by_name.items():
        prior = config.priors[name]
        obs_std = config.observation_std_overrides.get(name, config.default_observation_std)

        posterior_mean, posterior_std = normal_normal_update(
            prior_mean=prior.mean,
            prior_std=prior.std,
            observations=observations,
            observation_std=obs_std,
        )

        summaries[name] = UpdateSummary(
            prior_mean=prior.mean,
            prior_std=prior.std,
            posterior_mean=posterior_mean,
            posterior_std=posterior_std,
            observation_count=len(observations),
        )

    return summaries
