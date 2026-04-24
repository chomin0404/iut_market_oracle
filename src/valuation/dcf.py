from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class DCFInputs:
    initial_fcf: float
    growth_rate: float
    discount_rate: float
    forecast_years: int = 5
    terminal_growth_rate: float = 0.03

    def __post_init__(self) -> None:
        if self.initial_fcf <= 0:
            raise ValueError("initial_fcf must be positive.")
        if self.discount_rate <= -1.0:
            raise ValueError("discount_rate must be greater than -1.")
        if self.forecast_years <= 0:
            raise ValueError("forecast_years must be positive.")
        if self.terminal_growth_rate >= self.discount_rate:
            raise ValueError("terminal_growth_rate must be less than discount_rate.")


@dataclass(slots=True)
class DCFResult:
    projected_fcfs: list[float]
    discounted_fcfs: list[float]
    terminal_value: float
    discounted_terminal_value: float
    enterprise_value: float


def project_fcfs(initial_fcf: float, growth_rate: float, years: int) -> list[float]:
    if initial_fcf <= 0:
        raise ValueError("initial_fcf must be positive.")
    if years <= 0:
        raise ValueError("years must be positive.")

    fcfs: list[float] = []
    fcf = initial_fcf
    for _ in range(years):
        fcf = fcf * (1.0 + growth_rate)
        fcfs.append(fcf)
    return fcfs


def discount_cash_flows(cash_flows: list[float], discount_rate: float) -> list[float]:
    if discount_rate <= -1.0:
        raise ValueError("discount_rate must be greater than -1.")

    discounted: list[float] = []
    for t, cf in enumerate(cash_flows, start=1):
        discounted.append(cf / ((1.0 + discount_rate) ** t))
    return discounted


def gordon_terminal_value(
    final_year_fcf: float,
    discount_rate: float,
    terminal_growth_rate: float,
) -> float:
    if final_year_fcf <= 0:
        raise ValueError("final_year_fcf must be positive.")
    if terminal_growth_rate >= discount_rate:
        raise ValueError("terminal_growth_rate must be less than discount_rate.")

    return final_year_fcf * (1.0 + terminal_growth_rate) / (discount_rate - terminal_growth_rate)


def dcf_valuation(inputs: DCFInputs) -> DCFResult:
    projected_fcfs = project_fcfs(
        initial_fcf=inputs.initial_fcf,
        growth_rate=inputs.growth_rate,
        years=inputs.forecast_years,
    )
    discounted_fcfs = discount_cash_flows(projected_fcfs, inputs.discount_rate)

    terminal_value = gordon_terminal_value(
        final_year_fcf=projected_fcfs[-1],
        discount_rate=inputs.discount_rate,
        terminal_growth_rate=inputs.terminal_growth_rate,
    )
    discounted_terminal_value = terminal_value / (
        (1.0 + inputs.discount_rate) ** inputs.forecast_years
    )
    enterprise_value = sum(discounted_fcfs) + discounted_terminal_value

    return DCFResult(
        projected_fcfs=projected_fcfs,
        discounted_fcfs=discounted_fcfs,
        terminal_value=terminal_value,
        discounted_terminal_value=discounted_terminal_value,
        enterprise_value=enterprise_value,
    )


def reverse_dcf_implied_growth(
    target_enterprise_value: float,
    initial_fcf: float,
    discount_rate: float,
    forecast_years: int = 5,
    terminal_growth_rate: float = 0.03,
    low: float = -0.50,
    high: float = 0.80,
    tolerance: float = 1e-8,
    max_iter: int = 200,
) -> float:
    if target_enterprise_value <= 0:
        raise ValueError("target_enterprise_value must be positive.")
    if initial_fcf <= 0:
        raise ValueError("initial_fcf must be positive.")
    if terminal_growth_rate >= discount_rate:
        raise ValueError("terminal_growth_rate must be less than discount_rate.")

    def objective(growth_rate: float) -> float:
        result = dcf_valuation(
            DCFInputs(
                initial_fcf=initial_fcf,
                growth_rate=growth_rate,
                discount_rate=discount_rate,
                forecast_years=forecast_years,
                terminal_growth_rate=terminal_growth_rate,
            )
        )
        return result.enterprise_value - target_enterprise_value

    f_low = objective(low)
    f_high = objective(high)

    if f_low == 0:
        return low
    if f_high == 0:
        return high
    if f_low * f_high > 0:
        raise ValueError("Bisection bounds do not bracket a solution.")

    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        f_mid = objective(mid)

        if abs(f_mid) < tolerance:
            return mid

        if f_low * f_mid <= 0:
            high = mid
            f_high = f_mid
        else:
            low = mid
            f_low = f_mid

    return 0.5 * (low + high)
