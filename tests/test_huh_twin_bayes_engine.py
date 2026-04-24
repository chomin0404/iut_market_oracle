from pathlib import Path

from huh_twin.bayes_engine import (
    load_prior_config,
    normal_normal_update,
    regime_posterior,
    update_named_priors,
)


def test_normal_normal_update_moves_mean_toward_observations() -> None:
    posterior_mean, posterior_std = normal_normal_update(
        prior_mean=0.10,
        prior_std=0.03,
        observations=[0.14, 0.15, 0.13],
        observation_std=0.02,
    )

    assert 0.10 < posterior_mean < 0.15
    assert posterior_std < 0.03


def test_normal_normal_update_without_observations_returns_prior() -> None:
    posterior_mean, posterior_std = normal_normal_update(
        prior_mean=0.10,
        prior_std=0.03,
        observations=[],
        observation_std=0.02,
    )

    assert posterior_mean == 0.10
    assert posterior_std == 0.03


def test_regime_posterior_normalizes_to_one() -> None:
    posterior = regime_posterior(
        prior_weights={
            "normal": 0.60,
            "high_growth": 0.25,
            "disruption": 0.15,
        },
        likelihoods={
            "normal": 0.5,
            "high_growth": 0.9,
            "disruption": 0.2,
        },
    )

    assert abs(sum(posterior.values()) - 1.0) < 1e-12
    assert posterior["high_growth"] > posterior["disruption"]


def test_load_prior_config_from_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "priors.yaml"
    config_path.write_text(
        """
priors:
  growth_rate:
    mean: 0.10
    std: 0.03
  discount_rate:
    mean: 0.10
    std: 0.02

observation_model:
  default_std: 0.05
  overrides:
    growth_rate: 0.02

regimes:
  normal: 0.60
  high_growth: 0.25
  disruption: 0.15
""".strip()
    )

    config = load_prior_config(config_path)

    assert config.priors["growth_rate"].mean == 0.10
    assert config.priors["discount_rate"].std == 0.02
    assert config.default_observation_std == 0.05
    assert config.observation_std_overrides["growth_rate"] == 0.02
    assert abs(sum(config.regime_priors.values()) - 1.0) < 1e-12


def test_update_named_priors_returns_posterior_summaries(tmp_path: Path) -> None:
    config_path = tmp_path / "priors.yaml"
    config_path.write_text(
        """
priors:
  growth_rate:
    mean: 0.10
    std: 0.03
  discount_rate:
    mean: 0.10
    std: 0.02

observation_model:
  default_std: 0.05
  overrides:
    growth_rate: 0.02
    discount_rate: 0.01

regimes:
  normal: 0.60
  high_growth: 0.25
  disruption: 0.15
""".strip()
    )

    config = load_prior_config(config_path)
    summaries = update_named_priors(
        config=config,
        observations_by_name={
            "growth_rate": [0.12, 0.13, 0.11],
            "discount_rate": [0.095, 0.09, 0.092],
        },
    )

    growth = summaries["growth_rate"]
    discount = summaries["discount_rate"]

    assert growth.observation_count == 3
    assert 0.10 < growth.posterior_mean < 0.13
    assert growth.posterior_std < growth.prior_std

    assert discount.observation_count == 3
    assert 0.09 < discount.posterior_mean < 0.10
    assert discount.posterior_std < discount.prior_std
