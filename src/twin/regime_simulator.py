"""Probabilistic regime-switching and market-evolution simulators (T1100).

Regime Switching Model
----------------------
Uses a 2-state discrete-time Markov chain:

    Transition matrix:
        P = [[ p_stay_normal,    1 - p_stay_normal  ],
             [ 1 - p_stay_volatile,  p_stay_volatile ]]

Return distributions per regime:
    State 0 (normal):   ret_t ~ Laplace(mu=0.001, b=0.01)
    State 1 (volatile): ret_t ~ Cauchy(x0=-0.005, gamma=0.02),
                        clipped to [-0.15, 0.15] for simulation stability

Price evolution:
    price_t = price_{t-1} * (1 + ret_t)

Market Evolution Model
----------------------
Customer arrivals follow a Gamma-Poisson (Negative Binomial) mixture:
    lambda_t ~ Gamma(alpha, scale=1/beta)
    k_t | lambda_t ~ Poisson(lambda_t)

Market capture is the cumulative customer base weighted by logistic adoption:
    sigma(t) = 1 / (1 + exp(-t)),   t in linspace(-5, 5, n_steps)
    market_capture_t = cumsum(k)[t] * sigma(t)

Note: floating-point accumulation in price[t] = price[0] * prod(1+ret_i) over
long horizons may accumulate O(horizon * eps_mach) relative error.
"""

from __future__ import annotations

import numpy as np

from schemas import MarketEvolutionResult, RegimeSwitchResult

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Regime 0 (normal): Laplace return distribution
_LAPLACE_LOC: float = 0.001  # daily drift [decimal]
_LAPLACE_SCALE: float = 0.01  # scale b [decimal]

# Regime 1 (volatile): Cauchy return distribution
_CAUCHY_LOC: float = -0.005  # location x_0 [decimal]
_CAUCHY_SCALE: float = 0.02  # scale gamma [decimal]
_RETURN_CLIP_LO: float = -0.15  # lower clip to prevent price explosion
_RETURN_CLIP_HI: float = 0.15  # upper clip

# Sigmoid mapping range for market adoption curve
_SIGMOID_T_MIN: float = -5.0
_SIGMOID_T_MAX: float = 5.0


# ---------------------------------------------------------------------------
# Simulation functions
# ---------------------------------------------------------------------------


def simulate_regime_switching(
    n_steps: int,
    initial_price: float,
    p_stay_normal: float,
    p_stay_volatile: float,
    rng: np.random.Generator,
) -> RegimeSwitchResult:
    """Simulate asset prices under a 2-state Markov regime-switching model.

    All random draws are taken from ``rng`` in a fixed order to ensure
    reproducibility: uniform (regime transitions), Laplace (normal returns),
    Cauchy (volatile returns, via normal/chi2 ratio).

    Parameters
    ----------
    n_steps:
        Number of simulation steps (>= 1).  prices[0] is the initial price.
    initial_price:
        Starting asset price (must be > 0).
    p_stay_normal:
        P(regime=0 at t+1 | regime=0 at t); denoted p_00 in the transition matrix.
    p_stay_volatile:
        P(regime=1 at t+1 | regime=1 at t); denoted p_11 in the transition matrix.
    rng:
        Seeded NumPy random generator.  Caller must supply, e.g.
        ``np.random.default_rng(42)``.

    Returns
    -------
    RegimeSwitchResult
        prices and regimes series, each of length n_steps.

    Raises
    ------
    ValueError
        If n_steps < 1, initial_price <= 0, or transition probabilities
        are not in (0, 1).
    """
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    if initial_price <= 0.0:
        raise ValueError(f"initial_price must be > 0, got {initial_price}")
    if not (0.0 < p_stay_normal < 1.0):
        raise ValueError(f"p_stay_normal must be in (0, 1), got {p_stay_normal}")
    if not (0.0 < p_stay_volatile < 1.0):
        raise ValueError(f"p_stay_volatile must be in (0, 1), got {p_stay_volatile}")

    regimes = np.zeros(n_steps, dtype=int)
    prices = np.empty(n_steps, dtype=float)
    prices[0] = initial_price

    # Pre-draw all random variates for deterministic ordering
    u = rng.uniform(size=n_steps)
    laplace_draws = rng.laplace(loc=_LAPLACE_LOC, scale=_LAPLACE_SCALE, size=n_steps)
    # Cauchy via standard_cauchy: x = loc + scale * Z where Z ~ Cauchy(0,1)
    cauchy_draws = np.clip(
        rng.standard_cauchy(size=n_steps) * _CAUCHY_SCALE + _CAUCHY_LOC,
        _RETURN_CLIP_LO,
        _RETURN_CLIP_HI,
    )

    for t in range(1, n_steps):
        prev_regime = regimes[t - 1]
        if prev_regime == 0:
            regimes[t] = 0 if u[t] < p_stay_normal else 1
        else:
            regimes[t] = 1 if u[t] < p_stay_volatile else 0

        ret = laplace_draws[t] if regimes[t] == 0 else cauchy_draws[t]
        prices[t] = prices[t - 1] * (1.0 + ret)

    return RegimeSwitchResult(
        n_steps=n_steps,
        prices=prices.tolist(),
        regimes=regimes.tolist(),
        p_stay_normal=p_stay_normal,
        p_stay_volatile=p_stay_volatile,
    )


def simulate_market_evolution(
    n_steps: int,
    gamma_alpha: float,
    gamma_beta: float,
    rng: np.random.Generator,
) -> MarketEvolutionResult:
    """Simulate market size evolution via Gamma-Poisson mixture and sigmoid adoption.

    Model:
        lambda_t ~ Gamma(alpha, scale=1/beta)
        k_t | lambda_t ~ Poisson(lambda_t)
        sigma(t) = 1 / (1 + exp(-t)),  t in linspace(-5, 5, n_steps)
        market_capture_t = cumsum(k)[t] * sigma(t)

    Parameters
    ----------
    n_steps:
        Number of simulation steps (>= 1).
    gamma_alpha:
        Shape parameter alpha of the Gamma mixing distribution (> 0).
    gamma_beta:
        Rate parameter beta; the scale is 1/beta (> 0).
    rng:
        Seeded NumPy random generator.

    Returns
    -------
    MarketEvolutionResult

    Raises
    ------
    ValueError
        If n_steps < 1, gamma_alpha <= 0, or gamma_beta <= 0.
    """
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    if gamma_alpha <= 0.0:
        raise ValueError(f"gamma_alpha must be > 0, got {gamma_alpha}")
    if gamma_beta <= 0.0:
        raise ValueError(f"gamma_beta must be > 0, got {gamma_beta}")

    # lambda_t ~ Gamma(alpha, scale=1/beta)
    lambdas = rng.gamma(shape=gamma_alpha, scale=1.0 / gamma_beta, size=n_steps)
    # k_t | lambda_t ~ Poisson(lambda_t)
    new_customers = rng.poisson(lam=lambdas)
    cumulative = np.cumsum(new_customers).astype(float)

    t_vals = np.linspace(_SIGMOID_T_MIN, _SIGMOID_T_MAX, n_steps)
    sigmoid = 1.0 / (1.0 + np.exp(-t_vals))
    market_capture = cumulative * sigmoid

    return MarketEvolutionResult(
        n_steps=n_steps,
        new_customers=new_customers.tolist(),
        cumulative_base=cumulative.tolist(),
        sigmoid_factor=sigmoid.tolist(),
        market_capture=market_capture.tolist(),
        gamma_alpha=gamma_alpha,
        gamma_beta=gamma_beta,
    )


# ---------------------------------------------------------------------------
# Visualisation helpers  (return Figure; caller decides where to save)
# ---------------------------------------------------------------------------


def plot_regime_switching(result: RegimeSwitchResult):  # -> matplotlib.figure.Figure
    """Return a matplotlib Figure for the regime-switching price simulation.

    Two panels:
        Top: asset price series with volatile-regime steps highlighted in red.
        Bottom: regime label (0/1) over time.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    times = list(range(result.n_steps))
    prices = result.prices
    regimes = result.regimes

    fig, (ax_price, ax_regime) = plt.subplots(
        2, 1, figsize=(12, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    ax_price.plot(times, prices, color="steelblue", linewidth=1.0, label="Asset Price")

    # Highlight volatile regime segments
    vol_times = [t for t in times if regimes[t] == 1]
    vol_prices = [prices[t] for t in vol_times]
    if vol_times:
        ax_price.scatter(
            vol_times, vol_prices, color="crimson", s=8, alpha=0.5, label="Volatile Regime"
        )

    ax_price.set_ylabel("Price")
    ax_price.set_title("Asset Price — 2-State Markov Regime Switching (Laplace / Cauchy)")
    ax_price.legend(fontsize=8)

    ax_regime.step(times, regimes, color="darkorange", linewidth=0.8)
    ax_regime.set_yticks([0, 1])
    ax_regime.set_yticklabels(["Normal (0)", "Volatile (1)"])
    ax_regime.set_xlabel("Time Steps")
    ax_regime.set_ylabel("Regime")

    fig.tight_layout()
    return fig


def plot_market_evolution(result: MarketEvolutionResult):  # -> matplotlib.figure.Figure
    """Return a matplotlib Figure for the Gamma-Poisson market evolution.

    Two panels:
        Top: market capture index (cumulative customers * sigmoid).
        Bottom: new customer arrivals per step.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    times = list(range(result.n_steps))

    fig, (ax_cap, ax_cust) = plt.subplots(
        2, 1, figsize=(12, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    ax_cap.plot(
        times,
        result.market_capture,
        color="seagreen",
        linewidth=1.5,
        label="Market Capture Index",
    )
    ax_cap.set_ylabel("Market Capture")
    ax_cap.set_title(
        f"Market Evolution — Gamma-Poisson Mixture + Sigmoid Adoption "
        f"(α={result.gamma_alpha}, β={result.gamma_beta})"
    )
    ax_cap.legend(fontsize=8)

    ax_cust.bar(times, result.new_customers, color="cadetblue", alpha=0.6, width=1.0)
    ax_cust.set_xlabel("Time Steps")
    ax_cust.set_ylabel("New Customers")

    fig.tight_layout()
    return fig
