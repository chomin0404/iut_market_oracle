"""T1100 Regime Switching / Market Evolution and T1200 Matroid schemas."""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field, model_validator


class RegimeSwitchResult(BaseModel):
    """Output of a 2-state Markov regime-switching price simulation (T1100).

    prices:
        Simulated asset price series of length n_steps.
        prices[0] is the initial price; prices[t] = prices[t-1] * (1 + ret_t).
    regimes:
        Regime label at each step: 0 = normal (Laplace returns),
        1 = volatile (Cauchy returns, clipped).
    """

    n_steps: int = Field(..., ge=1)
    prices: list[float] = Field(..., min_length=1)
    regimes: list[int] = Field(..., min_length=1)
    p_stay_normal: float = Field(..., gt=0.0, lt=1.0, description="P(regime=0 | prev=0)")
    p_stay_volatile: float = Field(..., gt=0.0, lt=1.0, description="P(regime=1 | prev=1)")
    produced_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def lengths_consistent(self) -> RegimeSwitchResult:
        if len(self.prices) != self.n_steps:
            raise ValueError(
                f"prices length ({len(self.prices)}) must equal n_steps ({self.n_steps})"
            )
        if len(self.regimes) != self.n_steps:
            raise ValueError(
                f"regimes length ({len(self.regimes)}) must equal n_steps ({self.n_steps})"
            )
        return self


class MarketEvolutionResult(BaseModel):
    """Output of a Gamma-Poisson market evolution simulation (T1100).

    Customer arrivals per step follow a Negative Binomial (Gamma-Poisson) mixture:
        lambda_t ~ Gamma(alpha, scale=1/beta)
        k_t | lambda_t ~ Poisson(lambda_t)

    Market capture is modulated by a logistic sigmoid adoption curve.

    new_customers:
        Customer arrivals k_t at each step.
    cumulative_base:
        Cumulative sum of new_customers up to step t.
    sigmoid_factor:
        Logistic sigmoid values sigma(t) mapped over [-5, 5].
    market_capture:
        cumulative_base[t] * sigmoid_factor[t] — market capture index.
    """

    n_steps: int = Field(..., ge=1)
    new_customers: list[int] = Field(..., min_length=1)
    cumulative_base: list[float] = Field(..., min_length=1)
    sigmoid_factor: list[float] = Field(..., min_length=1)
    market_capture: list[float] = Field(..., min_length=1)
    gamma_alpha: float = Field(..., gt=0.0, description="Gamma shape parameter alpha")
    gamma_beta: float = Field(..., gt=0.0, description="Gamma rate parameter beta (scale=1/beta)")
    produced_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def lengths_consistent(self) -> MarketEvolutionResult:
        for name, lst in [
            ("new_customers", self.new_customers),
            ("cumulative_base", self.cumulative_base),
            ("sigmoid_factor", self.sigmoid_factor),
            ("market_capture", self.market_capture),
        ]:
            if len(lst) != self.n_steps:
                raise ValueError(f"{name} length ({len(lst)}) must equal n_steps ({self.n_steps})")
        return self


class MatroidLogConcavityResult(BaseModel):
    """Output of a matroid characteristic-polynomial log-concavity computation (T1200).

    Models the coefficients of a rank-generating polynomial:
        b_k = C(n, k) * rank_weight^k * corank_weight^(n-k),  k = 0…n
    After normalisation these form a probability mass function over subset sizes k.

    June Huh (2022 Fields Medal) proved that the characteristic polynomial of any
    matroid has log-concave coefficients.  This schema captures that property.

    log_concavity_checks:
        Boolean per interior index k=1…n-1: True iff b_k² >= b_{k-1} * b_{k+1}.
        Length = n_assets - 1.
    is_log_concave:
        True iff all entries in log_concavity_checks are True.
    """

    n_assets: int = Field(..., ge=1)
    rank_weight: float = Field(..., gt=0.0, description="Weight per rank unit (alpha)")
    corank_weight: float = Field(..., gt=0.0, description="Weight per corank unit (beta)")
    subset_sizes: list[int] = Field(..., min_length=1, description="k = 0, 1, …, n_assets")
    probability_mass: list[float] = Field(..., min_length=1, description="Normalised b_k")
    log_probability: list[float] = Field(..., min_length=1, description="log(b_k + eps)")
    log_concavity_checks: list[bool] = Field(..., description="b_k² >= b_{k-1}*b_{k+1} for k=1…n-1")
    is_log_concave: bool
    produced_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def lengths_consistent(self) -> MatroidLogConcavityResult:
        expected = self.n_assets + 1
        for name, lst in [
            ("subset_sizes", self.subset_sizes),
            ("probability_mass", self.probability_mass),
            ("log_probability", self.log_probability),
        ]:
            if len(lst) != expected:
                raise ValueError(f"{name} length ({len(lst)}) must equal n_assets + 1 ({expected})")
        if len(self.log_concavity_checks) != self.n_assets - 1:
            raise ValueError(
                f"log_concavity_checks length ({len(self.log_concavity_checks)}) "
                f"must equal n_assets - 1 ({self.n_assets - 1})"
            )
        return self
