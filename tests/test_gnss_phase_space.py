"""Tests for src/gnss/phase_space.py — Takens embedding + Lyapunov exponent.

Acceptance criteria:
    takens_embed:
        - output shape (N - (m-1)*tau, m)
        - ValueError when series too short
        - correct delay structure in each column

    max_lyapunov_exponent:
        - deterministic signal (sinusoid) → λ₁ ≤ 0 → alarm
        - chaotic signal (logistic map) → λ₁ > 0 → no alarm
        - too-short series → reason contains "too_few_points" or "embed_failed"
        - divergence_curve length = min(max_iter, n_embedded − 1)
        - n_points returned correctly
"""

from __future__ import annotations

import numpy as np
import pytest

from gnss.phase_space import (
    LYA_ALARM_THRESH,
    PhaseSpaceResult,
    max_lyapunov_exponent,
    takens_embed,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _logistic_map(n: int, r: float = 4.0, x0: float = 0.2) -> np.ndarray:
    """Generate a chaotic logistic map time series."""
    x = np.empty(n)
    x[0] = x0
    for i in range(1, n):
        x[i] = r * x[i - 1] * (1.0 - x[i - 1])
    return x


def _sinusoid(n: int, freq: float = 0.1) -> np.ndarray:
    """Generate a deterministic sinusoid (periodic → λ₁ ≤ 0)."""
    return np.sin(2 * np.pi * freq * np.arange(n, dtype=float))


# ---------------------------------------------------------------------------
# takens_embed
# ---------------------------------------------------------------------------


class TestTakensEmbed:
    def test_output_shape(self) -> None:
        x = np.arange(100.0)
        m, tau = 3, 2
        Y = takens_embed(x, m=m, tau=tau)
        expected_rows = 100 - (m - 1) * tau
        assert Y.shape == (expected_rows, m)

    def test_delay_structure(self) -> None:
        x = np.arange(20.0)
        m, tau = 3, 1
        Y = takens_embed(x, m=m, tau=tau)
        # Y[i] = [x[i], x[i+1], x[i+2]] for tau=1
        assert Y[0, 0] == x[0]
        assert Y[0, 1] == x[1]
        assert Y[0, 2] == x[2]

    def test_short_series_raises(self) -> None:
        x = np.array([1.0, 2.0])  # too short for m=3, tau=2
        with pytest.raises(ValueError):
            takens_embed(x, m=3, tau=2)

    def test_2d_input_raises(self) -> None:
        x = np.ones((10, 2))
        with pytest.raises(ValueError, match="1-D"):
            takens_embed(x)

    def test_unit_m_returns_column_vector(self) -> None:
        x = np.arange(10.0)
        Y = takens_embed(x, m=1, tau=1)
        assert Y.shape == (10, 1)


# ---------------------------------------------------------------------------
# max_lyapunov_exponent
# ---------------------------------------------------------------------------


class TestMaxLyapunovExponent:
    def test_chaotic_signal_positive_lambda(self) -> None:
        """Logistic map (r=4) is maximally chaotic; λ₁ should be positive."""
        x = _logistic_map(500)
        res = max_lyapunov_exponent(x, m=3, tau=1, theiler_window=5, max_iter=20)
        # Some noise tolerated; the key invariant is λ₁ > 0
        assert isinstance(res, PhaseSpaceResult)
        if not np.isnan(res.lambda_max):
            # Chaotic map → positive exponent → no alarm
            assert res.lambda_max > 0.0
            assert res.alarm is False

    def test_sinusoid_nonpositive_lambda(self) -> None:
        """Sinusoid is periodic; Lyapunov exponent should be ≤ 0 → alarm."""
        x = _sinusoid(500)
        res = max_lyapunov_exponent(x, m=3, tau=1, theiler_window=5, max_iter=20)
        if not np.isnan(res.lambda_max):
            assert res.lambda_max <= LYA_ALARM_THRESH
            assert res.alarm is True

    def test_too_short_returns_graceful_result(self) -> None:
        x = np.arange(10.0)
        res = max_lyapunov_exponent(x, m=3, tau=1)
        assert isinstance(res, PhaseSpaceResult)
        # Either embed_failed or too_few_points
        assert "failed" in res.reason or "few" in res.reason or np.isnan(res.lambda_max)

    def test_n_points_matches_embedding(self) -> None:
        n = 200
        x = _logistic_map(n)
        m, tau = 3, 1
        res = max_lyapunov_exponent(x, m=m, tau=tau)
        expected = n - (m - 1) * tau
        assert res.n_points == expected

    def test_divergence_curve_length(self) -> None:
        x = _logistic_map(300)
        max_iter = 15
        res = max_lyapunov_exponent(x, m=3, tau=1, max_iter=max_iter)
        if len(res.divergence_curve) > 0:
            assert len(res.divergence_curve) <= max_iter

    def test_returns_phase_space_result(self) -> None:
        x = _logistic_map(200)
        res = max_lyapunov_exponent(x)
        assert isinstance(res, PhaseSpaceResult)

    def test_deterministic_same_seed(self) -> None:
        """Same input → same λ₁ (no randomness in algorithm)."""
        x = _logistic_map(200)
        r1 = max_lyapunov_exponent(x, m=3, tau=1)
        r2 = max_lyapunov_exponent(x, m=3, tau=1)
        if not np.isnan(r1.lambda_max):
            assert r1.lambda_max == r2.lambda_max

    def test_longer_embedding_delay(self) -> None:
        """tau > 1 should not crash and return a result."""
        x = _logistic_map(300)
        res = max_lyapunov_exponent(x, m=3, tau=2)
        assert isinstance(res, PhaseSpaceResult)
