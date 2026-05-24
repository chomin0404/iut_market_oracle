"""Unit tests for twin/regime_simulator.py — including plot helpers (T1100).

matplotlib is configured to use the Agg (non-interactive) backend so that
tests run headlessly on CI without a display server.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # must precede pyplot import  # noqa: E402

import numpy as np
import pytest

from twin.regime_simulator import (
    plot_market_evolution,
    plot_regime_switching,
    simulate_market_evolution,
    simulate_regime_switching,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture()
def regime_result(rng):
    return simulate_regime_switching(
        n_steps=50,
        initial_price=100.0,
        p_stay_normal=0.95,
        p_stay_volatile=0.90,
        rng=rng,
    )


@pytest.fixture()
def market_result(rng):
    return simulate_market_evolution(
        n_steps=50,
        gamma_alpha=2.0,
        gamma_beta=1.0,
        rng=rng,
    )


# ---------------------------------------------------------------------------
# plot_regime_switching
# ---------------------------------------------------------------------------


class TestPlotRegimeSwitching:
    def test_returns_figure(self, regime_result) -> None:
        import matplotlib.figure

        fig = plot_regime_switching(regime_result)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_figure_has_two_axes(self, regime_result) -> None:
        fig = plot_regime_switching(regime_result)
        assert len(fig.axes) == 2

    def test_price_axes_has_data(self, regime_result) -> None:
        fig = plot_regime_switching(regime_result)
        ax_price = fig.axes[0]
        # Main price line should have as many points as n_steps
        lines = ax_price.get_lines()
        assert len(lines) >= 1
        assert len(lines[0].get_xdata()) == regime_result.n_steps

    def test_regime_axes_has_step_data(self, regime_result) -> None:
        fig = plot_regime_switching(regime_result)
        ax_regime = fig.axes[1]
        lines = ax_regime.get_lines()
        assert len(lines) >= 1

    def test_all_volatile_regime(self, rng) -> None:
        """p_stay_volatile=0.999 → almost all steps volatile; scatter should appear."""
        result = simulate_regime_switching(
            n_steps=200,
            initial_price=100.0,
            p_stay_normal=0.001,  # quickly leaves normal
            p_stay_volatile=0.999,
            rng=rng,
        )
        import matplotlib.figure

        fig = plot_regime_switching(result)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_no_volatile_steps(self) -> None:
        """p_stay_normal=0.999 → mostly normal; vol_times list is empty."""
        rng2 = np.random.default_rng(999)
        result = simulate_regime_switching(
            n_steps=30,
            initial_price=50.0,
            p_stay_normal=0.999,
            p_stay_volatile=0.001,
            rng=rng2,
        )
        import matplotlib.figure

        fig = plot_regime_switching(result)
        assert isinstance(fig, matplotlib.figure.Figure)


# ---------------------------------------------------------------------------
# plot_market_evolution
# ---------------------------------------------------------------------------


class TestPlotMarketEvolution:
    def test_returns_figure(self, market_result) -> None:
        import matplotlib.figure

        fig = plot_market_evolution(market_result)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_figure_has_two_axes(self, market_result) -> None:
        fig = plot_market_evolution(market_result)
        assert len(fig.axes) == 2

    def test_capture_axes_has_data(self, market_result) -> None:
        fig = plot_market_evolution(market_result)
        ax_cap = fig.axes[0]
        lines = ax_cap.get_lines()
        assert len(lines) >= 1
        assert len(lines[0].get_xdata()) == market_result.n_steps

    def test_customer_bar_axes(self, market_result) -> None:
        """Bar chart should contain n_steps bars."""
        fig = plot_market_evolution(market_result)
        ax_cust = fig.axes[1]
        # Bar containers or patches
        assert len(ax_cust.patches) == market_result.n_steps

    def test_title_contains_alpha_beta(self, market_result) -> None:
        fig = plot_market_evolution(market_result)
        title = fig.axes[0].get_title()
        assert "α=" in title or "a=" in title or str(market_result.gamma_alpha) in title

    def test_n_steps_1(self, rng) -> None:
        """Edge case: single step simulation should still produce a figure."""
        result = simulate_market_evolution(n_steps=1, gamma_alpha=1.0, gamma_beta=1.0, rng=rng)
        import matplotlib.figure

        fig = plot_market_evolution(result)
        assert isinstance(fig, matplotlib.figure.Figure)
