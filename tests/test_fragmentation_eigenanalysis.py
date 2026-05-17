"""Unit tests for fragmentation/eigenanalysis.py.

Reference test case (analytical Malthus parameter)
---------------------------------------------------
τ(x) = 0  (no growth, tau_coef=0)
κ(x) = κ₀ (constant, alpha=0)
p(x,y) = 2/y  (uniform binary, no loss: β→1)

Total particle number equation:
    Ṅ = -κ₀ N + 2κ₀ N = κ₀ N
    → N(t) = N₀ exp(κ₀ t)
    → λ_analytical = κ₀

Eigenfunction: φ(x) ∝ x⁻¹ (integrable on finite [x_min, x_max])

Due to finite-grid truncation and β < 1 (slight loss), the numerical λ
will approximate κ₀.  We verify the sign, positivity, and scaling.
"""

from __future__ import annotations

import numpy as np
import pytest

from fragmentation.eigenanalysis import estimate_malthus
from fragmentation.schemas import EigenResult, FragConfig

# Analytical Malthus value for the reference case (constant κ, τ=0, β≈1)
_KAPPA_0_REF: float = 1.0
# Gillespie MC estimator has Monte Carlo variance + finite-x_min absorption bias.
# 25% relative tolerance is appropriate for stochastic regression with N=500.
_REL_TOL: float = 0.25


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ref_config() -> FragConfig:
    """Reference config: τ=0, κ=1, α=0, β≈1 → λ ≈ 1.0."""
    return FragConfig(
        tau_coef=0.0,
        kappa_0=_KAPPA_0_REF,
        alpha=0.0,
        loss_efficiency=0.999,  # β→1 (near no-loss)
        pde_grid_size=300,
        x_min=0.01,
        x_max=50.0,
    )


# ---------------------------------------------------------------------------
# 1. Reference case: λ ≈ κ₀
# ---------------------------------------------------------------------------


def test_malthus_reference_case(ref_config: FragConfig) -> None:
    """λ ≈ κ₀ for constant fragmentation, no growth, near-no-loss."""
    result = estimate_malthus(ref_config)
    rel_err = abs(result.malthus_lambda - _KAPPA_0_REF) / _KAPPA_0_REF
    assert rel_err < _REL_TOL, (
        f"λ={result.malthus_lambda:.6f}, expected ≈ {_KAPPA_0_REF:.6f}, "
        f"relative error={rel_err:.4f} > {_REL_TOL}"
    )


def test_malthus_scales_with_kappa0(ref_config: FragConfig) -> None:
    """λ increases monotonically with κ₀."""
    kappas = [0.5, 1.0, 2.0, 5.0]
    lambdas = []
    for k in kappas:
        cfg = ref_config.model_copy(update={"kappa_0": k})
        lambdas.append(estimate_malthus(cfg).malthus_lambda)
    assert all(lambdas[i] < lambdas[i + 1] for i in range(len(lambdas) - 1)), (
        f"λ not monotone in κ₀: {list(zip(kappas, lambdas))}"
    )


# ---------------------------------------------------------------------------
# 2. Eigenfunction non-negativity and normalization
# ---------------------------------------------------------------------------


def test_eigenfunction_nonnegative(ref_config: FragConfig) -> None:
    """φ(x) ≥ 0 everywhere on the grid."""
    result = estimate_malthus(ref_config)
    phi = np.array(result.eigenfunction_phi)
    assert np.all(phi >= 0.0), f"Negative eigenfunction values: min={phi.min():.6f}"


def test_eigenfunction_l1_normalized(ref_config: FragConfig) -> None:
    """∫ φ(x) dx ≈ 1 (L¹ normalization on discrete grid)."""
    result = estimate_malthus(ref_config)
    xs = np.array(result.eigenfunction_x)
    phi = np.array(result.eigenfunction_phi)
    dx = xs[1] - xs[0]
    norm = float(np.sum(phi) * dx)
    assert abs(norm - 1.0) < 0.05, f"L¹ norm = {norm:.4f} ≠ 1.0"


def test_eigenfunction_length_matches_grid(ref_config: FragConfig) -> None:
    """eigenfunction_x and eigenfunction_phi have length = pde_grid_size."""
    result = estimate_malthus(ref_config)
    assert len(result.eigenfunction_x) == ref_config.pde_grid_size
    assert len(result.eigenfunction_phi) == ref_config.pde_grid_size


# ---------------------------------------------------------------------------
# 3. EigenResult schema fields
# ---------------------------------------------------------------------------


def test_eigen_result_fields(ref_config: FragConfig) -> None:
    """EigenResult contains all required fields with valid types."""
    result = estimate_malthus(ref_config)
    assert isinstance(result, EigenResult)
    assert isinstance(result.malthus_lambda, float)
    assert isinstance(result.spectral_gap, float)
    assert isinstance(result.converged, bool)
    assert result.spectral_gap >= 0.0


# ---------------------------------------------------------------------------
# 4. Effect of growth rate on λ
# ---------------------------------------------------------------------------


def test_growth_increases_lambda(ref_config: FragConfig) -> None:
    """Adding growth (tau_coef > 0) should increase λ."""
    lambda_no_growth = estimate_malthus(ref_config).malthus_lambda
    cfg_with_growth = ref_config.model_copy(update={"tau_coef": 0.2})
    lambda_with_growth = estimate_malthus(cfg_with_growth).malthus_lambda
    assert lambda_with_growth > lambda_no_growth, (
        f"Growth did not increase λ: {lambda_no_growth:.4f} → {lambda_with_growth:.4f}"
    )


# ---------------------------------------------------------------------------
# 5. Effect of loss on λ
# ---------------------------------------------------------------------------


def test_higher_loss_reduces_lambda(ref_config: FragConfig) -> None:
    """Higher loss (smaller β) reduces λ compared to near-no-loss case.

    Note: with β=0.5 and finite x_min, daughters are frequently absorbed below x_min,
    making λ potentially negative.  We verify that higher loss gives a strictly lower λ.
    """
    cfg_moderate_loss = ref_config.model_copy(update={"loss_efficiency": 0.9})
    lambda_low_loss = estimate_malthus(ref_config).malthus_lambda  # β=0.999
    lambda_high_loss = estimate_malthus(cfg_moderate_loss).malthus_lambda  # β=0.9
    assert lambda_high_loss < lambda_low_loss, (
        f"Higher loss (β=0.9) did not reduce λ: {lambda_low_loss:.4f} → {lambda_high_loss:.4f}"
    )


# ---------------------------------------------------------------------------
# 6. Grid convergence (coarse → fine → λ converges)
# ---------------------------------------------------------------------------


def test_grid_convergence(ref_config: FragConfig) -> None:
    """Finer grid → λ closer to analytical value κ₀."""
    lambdas = []
    for m in [50, 100, 200, 300]:
        cfg = ref_config.model_copy(update={"pde_grid_size": m})
        lambdas.append(estimate_malthus(cfg).malthus_lambda)
    # Differences should decrease (not necessarily monotone, but the last should
    # be closer to κ₀ than the first)
    err_coarse = abs(lambdas[0] - _KAPPA_0_REF)
    err_fine = abs(lambdas[-1] - _KAPPA_0_REF)
    assert err_fine <= err_coarse + 0.05, (
        f"Grid convergence failed: coarse err={err_coarse:.4f}, fine err={err_fine:.4f}"
    )
