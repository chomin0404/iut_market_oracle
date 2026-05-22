"""T900 Exit Strategy Engine schemas."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator


class ExitType(str, Enum):
    """Exit route classification."""

    IPO = "ipo"
    MA = "ma"
    SECONDARY = "secondary"
    WIND_DOWN = "wind_down"


class ExitOption(BaseModel):
    """Specification of one exit route with timing and value estimates.

    Timing model: triangular distribution over [timing_earliest, timing_latest]
    with mode at timing_expected (all in years from now).

    value_by_scenario:
        Mapping of scenario name → enterprise value in the same unit as
        floor_value (e.g. JPY millions).  At least one scenario required.

    floor_value:
        Minimum net deal value (analogous to a put strike).
        Payoff per scenario = max(V_s - floor_value, 0).

    discount_rate:
        Annual WACC used to discount future payoffs to present value.
    """

    name: str = Field(..., min_length=1)
    exit_type: ExitType
    timing_earliest: float = Field(..., ge=0.0, description="Years from now")
    timing_expected: float = Field(..., ge=0.0, description="Years from now (mode)")
    timing_latest: float = Field(..., ge=0.0, description="Years from now")
    value_by_scenario: dict[str, float] = Field(..., description="scenario name → enterprise value")
    floor_value: float = Field(default=0.0, ge=0.0, description="Minimum deal value")
    discount_rate: float = Field(..., gt=0.0, description="Annual WACC")

    @model_validator(mode="after")
    def timing_ordered(self) -> ExitOption:
        if not (self.timing_earliest <= self.timing_expected <= self.timing_latest):
            raise ValueError(
                "timing must satisfy earliest <= expected <= latest, got "
                f"({self.timing_earliest}, {self.timing_expected}, {self.timing_latest})"
            )
        return self

    @field_validator("value_by_scenario")
    @classmethod
    def non_empty_scenarios(cls, v: dict[str, float]) -> dict[str, float]:
        if not v:
            raise ValueError("value_by_scenario must contain at least one scenario")
        return v


class ExitValueSummary(BaseModel):
    """Pricing output for one exit option (T900).

    scenario_payoffs:
        max(V_s - floor, 0) per scenario (before discounting).
    scenario_pvs:
        Present value of each scenario payoff, discounted at timing_expected.
    expected_value:
        Probability-weighted mean of scenario_pvs.
    sensitivity:
        Central-difference ∂EV/∂p for discount_rate, timing_expected, floor_value.
    """

    option_name: str = Field(..., min_length=1)
    exit_type: ExitType
    scenario_payoffs: dict[str, float]
    scenario_pvs: dict[str, float]
    expected_value: float
    sensitivity: dict[str, float] = Field(default_factory=dict)
    produced_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TimingDistribution(BaseModel):
    """Discretised exit-timing probability distribution (T900).

    Derived from a triangular distribution over [earliest, latest] with
    mode at expected.  Probabilities sum to 1.0 up to floating-point error.

    time_steps:
        Discrete time points in years from now.
    probabilities:
        P(exit at t_k) for each step; normalised to sum to 1.0.
    expected_timing:
        Probability-weighted mean timing E[T] = Σ_k t_k · P(t_k).
    """

    option_name: str = Field(..., min_length=1)
    time_steps: list[float] = Field(..., min_length=1)
    probabilities: list[float] = Field(..., min_length=1)
    expected_timing: float = Field(..., ge=0.0)

    @model_validator(mode="after")
    def steps_and_probs_aligned(self) -> TimingDistribution:
        if len(self.time_steps) != len(self.probabilities):
            raise ValueError(
                f"time_steps length ({len(self.time_steps)}) must equal "
                f"probabilities length ({len(self.probabilities)})"
            )
        total = sum(self.probabilities)
        if not (0.999 <= total <= 1.001):
            raise ValueError(f"probabilities must sum to ~1.0, got {total:.6f}")
        return self
