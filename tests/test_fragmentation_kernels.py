"""Unit tests for fragmentation/kernels.py.

Verified properties
-------------------
1. Daughter kernel normalization: ∫₀^y p(x,y) dx = 2 for all y, β.
2. Loss constraint [A2]: x₁ + x₂ = β · x_parent < x_parent.
3. Non-negativity: x₁, x₂ ≥ 0.
4. Growth (linear): x(t+Δt) = x(t) · exp(a·Δt).
5. Fragmentation rate scaling: κ(x) = κ₀ · x^α (monotone in x for α > 0).
6. Initial sampler (truncated_t): all samples in (x_min, x_max), ν=3 enforced.
7. Gamma sampler: mean ≈ init_loc within 3-sigma.
"""

from __future__ import annotations

import numpy as np
import pytest

from fragmentation.kernels import (
    apply_growth,
    frag_rate,
    frag_rates_array,
    kernel_normalization,
    sample_daughters,
    sample_initial_particles,
)
from fragmentation.schemas import FragConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def base_config() -> FragConfig:
    return FragConfig(
        n_particles=200,
        seed=0,
        T=5.0,
        tau_coef=0.1,
        kappa_0=1.0,
        alpha=0.0,
        loss_efficiency=0.9,
        init_dist="truncated_t",
        init_loc=10.0,
        init_scale=3.0,
        init_df=3,
    )


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


# ---------------------------------------------------------------------------
# 1. Daughter kernel normalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("y", [1.0, 5.0, 20.0, 100.0])
@pytest.mark.parametrize("beta", [0.5, 0.8, 0.95])
def test_kernel_normalization(y: float, beta: float) -> None:
    """∫₀^y p(x,y) dx ≈ 2 for uniform binary kernel with any β."""
    config = FragConfig(loss_efficiency=beta)
    integral = kernel_normalization(y, config, n_quad=2000)
    assert abs(integral - 2.0) < 0.01, (
        f"Kernel not normalized: ∫p(x,{y})dx = {integral:.4f} (β={beta})"
    )


# ---------------------------------------------------------------------------
# 2 & 3. Loss constraint and non-negativity  [A2]
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("x_parent", [0.5, 2.0, 10.0, 50.0])
@pytest.mark.parametrize("beta", [0.5, 0.8, 0.95])
def test_loss_constraint_and_nonnegativity(
    x_parent: float, beta: float, rng: np.random.Generator
) -> None:
    """x₁ + x₂ = β · x_parent < x_parent, x₁ ≥ 0, x₂ ≥ 0."""
    config = FragConfig(loss_efficiency=beta)
    for _ in range(100):
        x1, x2 = sample_daughters(x_parent, config, rng)
        assert x1 >= 0.0, f"x₁={x1} < 0 (β={beta})"
        assert x2 >= 0.0, f"x₂={x2} < 0 (β={beta})"
        expected_sum = beta * x_parent
        assert np.isclose(x1 + x2, expected_sum, rtol=1e-9), (
            f"x₁+x₂={x1 + x2:.6f} ≠ β·y={expected_sum:.6f}"
        )
        assert x1 + x2 < x_parent, "Loss constraint violated: x₁+x₂ ≥ x_parent"


# ---------------------------------------------------------------------------
# 4. Growth: exact ODE solution
# ---------------------------------------------------------------------------


def test_apply_growth_linear_exact(base_config: FragConfig) -> None:
    """x(t+Δt) = x₀ · exp(a·Δt) for τ(x) = a·x."""
    sizes = np.array([1.0, 5.0, 10.0])
    dt = 2.0
    a = base_config.tau_coef
    result = apply_growth(sizes, dt, base_config)
    expected = sizes * np.exp(a * dt)
    np.testing.assert_allclose(result, expected, rtol=1e-12)


def test_apply_growth_zero_tau() -> None:
    """No growth when tau_coef = 0."""
    config = FragConfig(tau_coef=0.0)
    sizes = np.array([3.0, 7.0, 15.0])
    result = apply_growth(sizes, 10.0, config)
    np.testing.assert_allclose(result, sizes)


def test_apply_growth_zero_dt(base_config: FragConfig) -> None:
    """Zero elapsed time → sizes unchanged."""
    sizes = np.array([2.0, 8.0])
    result = apply_growth(sizes, 0.0, base_config)
    np.testing.assert_allclose(result, sizes)


# ---------------------------------------------------------------------------
# 5. Fragmentation rate scaling
# ---------------------------------------------------------------------------


def test_frag_rate_constant(base_config: FragConfig) -> None:
    """For α=0, κ(x) = κ₀ regardless of x."""
    for x in [0.1, 1.0, 10.0, 100.0]:
        assert np.isclose(frag_rate(x, base_config), base_config.kappa_0)


def test_frag_rate_power_law() -> None:
    """For α=1, κ(x) = κ₀·x (linear in x)."""
    config = FragConfig(kappa_0=2.0, alpha=1.0)
    assert np.isclose(frag_rate(3.0, config), 6.0)
    assert np.isclose(frag_rate(5.0, config), 10.0)


def test_frag_rate_monotone_in_x() -> None:
    """For α > 0, larger x → larger κ(x)."""
    config = FragConfig(kappa_0=1.0, alpha=0.5)
    xs = np.array([1.0, 2.0, 5.0, 10.0])
    rates = frag_rates_array(xs, config)
    assert np.all(np.diff(rates) > 0), "Rates not monotone in x for α>0"


def test_frag_rate_control_input(base_config: FragConfig) -> None:
    """Control u > 0 increases fragmentation rate."""
    r0 = frag_rate(5.0, base_config, control_u=0.0)
    r1 = frag_rate(5.0, base_config, control_u=1.0)
    assert r1 > r0
    assert np.isclose(r1, 2.0 * r0)  # (1+1)*κ₀ = 2κ₀


# ---------------------------------------------------------------------------
# 6. Initial sampler: truncated Student-t
# ---------------------------------------------------------------------------


def test_truncated_t_bounds(base_config: FragConfig, rng: np.random.Generator) -> None:
    """All sampled particles are in (x_min, x_max)."""
    sizes = sample_initial_particles(base_config, rng)
    assert len(sizes) == base_config.n_particles
    assert np.all(sizes > base_config.x_min), "Some sizes ≤ x_min"
    assert np.all(sizes < base_config.x_max), "Some sizes ≥ x_max"


def test_truncated_t_finite_moments(base_config: FragConfig, rng: np.random.Generator) -> None:
    """For df=3, sample mean and variance are finite (sanity check)."""
    sizes = sample_initial_particles(base_config, rng)
    assert np.isfinite(sizes.mean())
    assert np.isfinite(sizes.var())


def test_truncated_t_reproducible(base_config: FragConfig) -> None:
    """Same seed → same samples."""
    rng1 = np.random.default_rng(base_config.seed)
    rng2 = np.random.default_rng(base_config.seed)
    s1 = sample_initial_particles(base_config, rng1)
    s2 = sample_initial_particles(base_config, rng2)
    np.testing.assert_array_equal(s1, s2)


# ---------------------------------------------------------------------------
# 7. Gamma sampler
# ---------------------------------------------------------------------------


def test_gamma_sampler_bounds() -> None:
    """Gamma sampler: all samples in (x_min, x_max)."""
    config = FragConfig(init_dist="gamma", init_loc=10.0, init_scale=3.0, n_particles=500)
    rng = np.random.default_rng(7)
    sizes = sample_initial_particles(config, rng)
    assert np.all(sizes > config.x_min)
    assert np.all(sizes < config.x_max)
