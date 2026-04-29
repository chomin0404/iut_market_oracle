"""Tests for src/graph/rmt.py — Marchenko-Pastur denoising (Tao-Vu 2011)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from graph.rmt import (
    DenoisingResult,
    MPBounds,
    _hhi_offdiag,
    denoise_correlation_matrix,
    marchenko_pastur_bounds,
    rmt_dependency_concentration,
    strength_matrix,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)


def _identity(p: int) -> np.ndarray:
    return np.eye(p, dtype=float)


def _corr_matrix(p: int, rho: float) -> np.ndarray:
    """p×p equicorrelation matrix: C[i,i]=1, C[i,j]=rho (i≠j)."""
    C = np.full((p, p), rho, dtype=float)
    np.fill_diagonal(C, 1.0)
    return C


def _block_spike(p: int, k: int, spike: float, rho_noise: float = 0.0) -> np.ndarray:
    """p×p matrix with k×k block of strength `spike` embedded in identity.

    Used to inject k eigenvalues well above the MP upper bound.
    spike must be large enough that the block eigenvalue exceeds λ_+.
    """
    C = np.eye(p, dtype=float)
    block = np.full((k, k), rho_noise)
    np.fill_diagonal(block, 1.0)
    # Add spike to block eigenvalue: eigenvalue of block ≈ 1 + (k-1)*rho_noise + spike
    block[0, :k] += spike / k
    block[:k, 0] += spike / k
    block[0, 0] -= spike / k  # avoid double-counting diagonal boost
    C[:k, :k] = block
    # Force symmetry
    C = (C + C.T) * 0.5
    np.fill_diagonal(C, 1.0)
    return C


# ---------------------------------------------------------------------------
# marchenko_pastur_bounds
# ---------------------------------------------------------------------------


class TestMarchenkoPasturBounds:
    def test_square_case_q1(self) -> None:
        """q=1 (p=n): λ_+ = 4σ², λ_- = 0."""
        b = marchenko_pastur_bounds(100, 100, sigma_sq=1.0)
        assert isinstance(b, MPBounds)
        assert math.isclose(b.upper, 4.0, rel_tol=1e-12)
        assert math.isclose(b.lower, 0.0, abs_tol=1e-12)
        assert math.isclose(b.q, 1.0, rel_tol=1e-12)
        assert math.isclose(b.sigma_sq, 1.0)

    def test_q_quarter(self) -> None:
        """q=0.25 (p=n/4): λ_+ = (1+0.5)²=2.25, λ_- = (1-0.5)²=0.25."""
        b = marchenko_pastur_bounds(25, 100, sigma_sq=1.0)
        assert math.isclose(b.upper, 2.25, rel_tol=1e-12)
        assert math.isclose(b.lower, 0.25, rel_tol=1e-12)
        assert math.isclose(b.q, 0.25, rel_tol=1e-12)

    def test_sigma_sq_scaling(self) -> None:
        """Bounds scale linearly with σ²."""
        s = 2.5
        b1 = marchenko_pastur_bounds(50, 200, sigma_sq=1.0)
        b2 = marchenko_pastur_bounds(50, 200, sigma_sq=s)
        assert math.isclose(b2.upper, b1.upper * s, rel_tol=1e-12)
        assert math.isclose(b2.lower, b1.lower * s, rel_tol=1e-12)

    def test_q_greater_than_1_bulk_edge(self) -> None:
        """q>1 (rank-deficient case): bulk lower edge λ_- = σ²(√q-1)² > 0.

        The MP density also has a point mass at λ=0 when q>1, but the bulk
        lower edge of the continuous part is still σ²(√q-1)².
        """
        b = marchenko_pastur_bounds(200, 100)  # q=2, √q≈1.414
        expected_lower = (2.0**0.5 - 1.0) ** 2  # ≈ 0.1716
        assert math.isclose(b.lower, expected_lower, rel_tol=1e-12)
        assert b.upper > b.lower

    def test_single_variable(self) -> None:
        b = marchenko_pastur_bounds(1, 1000)
        assert b.upper > 0.0

    def test_invalid_n_vars(self) -> None:
        with pytest.raises(ValueError, match="n_vars"):
            marchenko_pastur_bounds(0, 100)

    def test_invalid_n_obs(self) -> None:
        with pytest.raises(ValueError, match="n_obs"):
            marchenko_pastur_bounds(10, 0)

    def test_invalid_sigma_sq(self) -> None:
        with pytest.raises(ValueError, match="sigma_sq"):
            marchenko_pastur_bounds(10, 100, sigma_sq=-1.0)

    def test_invalid_sigma_sq_zero(self) -> None:
        with pytest.raises(ValueError, match="sigma_sq"):
            marchenko_pastur_bounds(10, 100, sigma_sq=0.0)


# ---------------------------------------------------------------------------
# denoise_correlation_matrix — input validation
# ---------------------------------------------------------------------------


class TestDenoisingValidation:
    def test_1d_raises(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            denoise_correlation_matrix(np.ones(5), 100)

    def test_non_square_raises(self) -> None:
        with pytest.raises(ValueError, match="square"):
            denoise_correlation_matrix(np.ones((3, 4)), 100)

    def test_asymmetric_raises(self) -> None:
        C = np.eye(3)
        C[0, 1] = 0.5  # not symmetric
        with pytest.raises(ValueError, match="symmetric"):
            denoise_correlation_matrix(C, 100)

    def test_n_obs_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="n_obs"):
            denoise_correlation_matrix(np.eye(3), 0)


# ---------------------------------------------------------------------------
# denoise_correlation_matrix — identity matrix
# ---------------------------------------------------------------------------


class TestDenoisingIdentity:
    """Identity matrix: all eigenvalues = 1.

    For p=10, n=10 → q=1, λ_+=4.  Since 1 < 4, n_signal=0, n_noise=p.
    For p=10, n=1000 → q=0.01, λ_+≈(1+0.1)²=1.21.  Still n_signal=0.
    """

    def test_n_signal_zero_q1(self) -> None:
        p = 10
        result = denoise_correlation_matrix(_identity(p), p)
        assert result.n_signal == 0
        assert result.n_noise == p

    def test_n_signal_zero_large_n(self) -> None:
        p = 10
        result = denoise_correlation_matrix(_identity(p), 1000)
        assert result.n_signal == 0

    def test_signal_noise_sum(self) -> None:
        p = 15
        result = denoise_correlation_matrix(_identity(p), 50)
        assert result.n_signal + result.n_noise == p

    def test_psd_after_denoising(self) -> None:
        p = 10
        result = denoise_correlation_matrix(_identity(p), p)
        eigs = np.linalg.eigvalsh(result.matrix_cleaned)
        assert eigs.min() >= -1e-10

    def test_diagonal_preserved_identity(self) -> None:
        """Diagonal of cleaned matrix should equal diagonal of input (=1)."""
        p = 8
        result = denoise_correlation_matrix(_identity(p), 100)
        diag = np.diag(result.matrix_cleaned)
        np.testing.assert_allclose(diag, np.ones(p), atol=1e-10)

    def test_eigenvalues_raw_sorted(self) -> None:
        """eigenvalues_raw must be in ascending order (eigh contract)."""
        p = 10
        result = denoise_correlation_matrix(_identity(p), p)
        raw = result.eigenvalues_raw
        assert raw == sorted(raw)

    def test_result_type(self) -> None:
        result = denoise_correlation_matrix(_identity(5), 20)
        assert isinstance(result, DenoisingResult)
        assert isinstance(result.mp_bounds, MPBounds)
        assert isinstance(result.matrix_cleaned, np.ndarray)


# ---------------------------------------------------------------------------
# denoise_correlation_matrix — equicorrelation matrix
# ---------------------------------------------------------------------------


class TestDenoisingEquicorrelation:
    """Equicorrelation C[i,j]=ρ for i≠j.

    Eigenvalues: one large eigenvalue = 1+(p-1)ρ, rest = 1-ρ (p-1 fold).
    For ρ=0.9, p=20: λ_max = 1+19*0.9 = 18.1 >> λ_+(q=1)=4 → n_signal=1.
    """

    def test_n_signal_one_high_rho(self) -> None:
        p, rho = 20, 0.9
        C = _corr_matrix(p, rho)
        result = denoise_correlation_matrix(C, p)
        assert result.n_signal == 1

    def test_psd_after_denoising_high_rho(self) -> None:
        p, rho = 20, 0.9
        C = _corr_matrix(p, rho)
        result = denoise_correlation_matrix(C, p)
        eigs = np.linalg.eigvalsh(result.matrix_cleaned)
        assert eigs.min() >= -1e-10

    def test_diagonal_preserved_high_rho(self) -> None:
        p, rho = 15, 0.8
        C = _corr_matrix(p, rho)
        result = denoise_correlation_matrix(C, p)
        diag = np.diag(result.matrix_cleaned)
        np.testing.assert_allclose(diag, np.ones(p), atol=1e-10)

    def test_symmetry_preserved(self) -> None:
        p, rho = 12, 0.6
        C = _corr_matrix(p, rho)
        result = denoise_correlation_matrix(C, p * 2)
        M = result.matrix_cleaned
        np.testing.assert_allclose(M, M.T, atol=1e-12)

    def test_hhi_cleaned_leq_raw(self) -> None:
        """Denoising should not increase spurious concentration."""
        p, rho = 20, 0.9
        C = _corr_matrix(p, rho)
        # use n_obs >> p so λ_+ is tight and more noise gets removed
        result = denoise_correlation_matrix(C, p * 10)
        # Denoised HHI ≤ raw HHI is expected for noise-dominated off-diagonal structure
        # (not strictly guaranteed for all matrices, but holds here)
        assert result.hhi_cleaned <= result.hhi_raw + 1e-9


# ---------------------------------------------------------------------------
# denoise_correlation_matrix — eigenvalue count invariant
# ---------------------------------------------------------------------------


class TestDenoisingEigenvalueCount:
    @pytest.mark.parametrize("p,n", [(5, 5), (10, 30), (50, 100), (100, 200)])
    def test_signal_plus_noise_equals_p(self, p: int, n: int) -> None:
        C = _corr_matrix(p, 0.3)
        result = denoise_correlation_matrix(C, n)
        assert result.n_signal + result.n_noise == p

    def test_eigenvalues_raw_length(self) -> None:
        p = 12
        result = denoise_correlation_matrix(_identity(p), 50)
        assert len(result.eigenvalues_raw) == p

    def test_eigenvalues_cleaned_length(self) -> None:
        p = 12
        result = denoise_correlation_matrix(_identity(p), 50)
        assert len(result.eigenvalues_cleaned) == p


# ---------------------------------------------------------------------------
# denoise_correlation_matrix — sigma_sq estimation
# ---------------------------------------------------------------------------


class TestDenoisingSigmaSq:
    def test_sigma_sq_from_trace(self) -> None:
        """For correlation matrix (diag=1), σ² = trace/p = 1."""
        p = 20
        C = _corr_matrix(p, 0.4)
        result = denoise_correlation_matrix(C, p * 2)
        assert math.isclose(result.mp_bounds.sigma_sq, 1.0, rel_tol=1e-12)

    def test_sigma_sq_covariance(self) -> None:
        """For covariance matrix with trace=p*s, σ² = s."""
        p, s = 10, 3.0
        C = np.eye(p) * s
        result = denoise_correlation_matrix(C, p * 2)
        assert math.isclose(result.mp_bounds.sigma_sq, s, rel_tol=1e-12)


# ---------------------------------------------------------------------------
# denoise_correlation_matrix — noise eigenvalue replacement
# ---------------------------------------------------------------------------


class TestDenoisingNoiseReplacement:
    def test_noise_eigenvalues_replaced_by_mean(self) -> None:
        """All cleaned noise eigenvalues should equal the mean of raw noise eigenvalues."""
        p, rho = 20, 0.9
        C = _corr_matrix(p, rho)
        result = denoise_correlation_matrix(C, p)

        raw = np.array(result.eigenvalues_raw)
        cleaned = np.array(result.eigenvalues_cleaned)
        signal_mask = raw > result.mp_bounds.upper
        noise_mask = ~signal_mask

        if noise_mask.any():
            mu_noise = raw[noise_mask].mean()
            np.testing.assert_allclose(cleaned[noise_mask], mu_noise, atol=1e-12)

    def test_signal_eigenvalues_unchanged(self) -> None:
        """Signal eigenvalues should be unchanged after clipping."""
        p, rho = 20, 0.9
        C = _corr_matrix(p, rho)
        result = denoise_correlation_matrix(C, p)

        raw = np.array(result.eigenvalues_raw)
        cleaned = np.array(result.eigenvalues_cleaned)
        signal_mask = raw > result.mp_bounds.upper
        np.testing.assert_allclose(cleaned[signal_mask], raw[signal_mask], atol=1e-12)


# ---------------------------------------------------------------------------
# rmt_dependency_concentration
# ---------------------------------------------------------------------------


class TestRmtDependencyConcentration:
    def test_returns_float(self) -> None:
        C = _corr_matrix(10, 0.5)
        val = rmt_dependency_concentration(C, 50)
        assert isinstance(val, float)

    def test_in_unit_interval(self) -> None:
        for p, rho, n in [(5, 0.0, 20), (10, 0.5, 30), (20, 0.9, 40)]:
            C = _corr_matrix(p, rho)
            # rho=0 gives identity; skip HHI range check (0.0 is valid)
            val = rmt_dependency_concentration(C, n)
            assert 0.0 <= val <= 1.0, f"HHI={val} out of [0,1] for p={p}, rho={rho}"

    def test_identity_low_concentration(self) -> None:
        """Identity: no off-diagonal signal → HHI should be 0 or near-uniform."""
        p = 10
        val = rmt_dependency_concentration(_identity(p), p)
        # For identity all off-diag = 0 → HHI = 0
        assert math.isclose(val, 0.0, abs_tol=1e-10)

    def test_consistent_with_denoising_result(self) -> None:
        C = _corr_matrix(15, 0.7)
        n = 60
        full = denoise_correlation_matrix(C, n)
        shortcut = rmt_dependency_concentration(C, n)
        assert math.isclose(full.hhi_cleaned, shortcut, rel_tol=1e-12)


# ---------------------------------------------------------------------------
# _hhi_offdiag (internal helper)
# ---------------------------------------------------------------------------


class TestHhiOffdiag:
    def test_identity_returns_zero(self) -> None:
        assert _hhi_offdiag(np.eye(5)) == 0.0

    def test_uniform_off_diagonal(self) -> None:
        """All off-diagonal equal → shares are 1/p → HHI = 1/p."""
        p = 4
        C = np.ones((p, p))
        np.fill_diagonal(C, 0.0)
        hhi = _hhi_offdiag(C)
        assert math.isclose(hhi, 1.0 / p, rel_tol=1e-10)

    def test_single_hub_high_hhi(self) -> None:
        """One node connected to all, others isolated → HHI close to 1."""
        p = 5
        C = np.zeros((p, p))
        C[0, 1:] = 1.0
        C[1:, 0] = 1.0
        hhi = _hhi_offdiag(C)
        # Node 0 dominates the column sums
        assert hhi > 1.0 / p

    def test_range(self) -> None:
        """HHI always in [0, 1]."""
        for _ in range(10):
            M = RNG.random((6, 6))
            M = (M + M.T) * 0.5
            h = _hhi_offdiag(M)
            assert 0.0 <= h <= 1.0


# ---------------------------------------------------------------------------
# strength_matrix (GraphInput helper)
# ---------------------------------------------------------------------------


class TestStrengthMatrix:
    def _make_graph(self, edges: list[tuple[str, str, float]]):
        """Minimal GraphInput-like object for testing strength_matrix."""
        from unittest.mock import MagicMock

        graph = MagicMock()
        nodes = []
        node_ids: set[str] = set()
        for src, tgt, _ in edges:
            node_ids.add(src)
            node_ids.add(tgt)
        for nid in sorted(node_ids):
            n = MagicMock()
            n.node_id = nid
            nodes.append(n)
        graph.nodes = nodes

        edge_list = []
        for src, tgt, strength in edges:
            e = MagicMock()
            e.source = src
            e.target = tgt
            e.strength = strength
            edge_list.append(e)
        graph.edges = edge_list
        return graph

    def test_diagonal_is_one(self) -> None:
        graph = self._make_graph([("A", "B", 0.8), ("B", "C", 0.6), ("A", "C", 0.4)])
        S, order = strength_matrix(graph)
        np.testing.assert_allclose(np.diag(S), np.ones(len(order)), atol=1e-12)

    def test_symmetric(self) -> None:
        graph = self._make_graph([("A", "B", 0.8), ("B", "C", 0.6), ("C", "A", 0.3)])
        S, _ = strength_matrix(graph)
        np.testing.assert_allclose(S, S.T, atol=1e-12)

    def test_off_diagonal_in_unit_interval(self) -> None:
        graph = self._make_graph([("X", "Y", 1.0), ("Y", "Z", 0.5)])
        S, _ = strength_matrix(graph)
        p = S.shape[0]
        mask = ~np.eye(p, dtype=bool)
        off = S[mask]
        assert off.min() >= -1e-12
        assert off.max() <= 1.0 + 1e-12

    def test_no_edges_returns_identity(self) -> None:
        # Empty edges → no nodes → trivial; use manual graph with nodes but no edges
        from unittest.mock import MagicMock

        g = MagicMock()
        n1, n2 = MagicMock(), MagicMock()
        n1.node_id, n2.node_id = "A", "B"
        g.nodes = [n1, n2]
        g.edges = []
        S, order = strength_matrix(g)
        np.testing.assert_allclose(S, np.eye(2), atol=1e-12)

    def test_node_order_respected(self) -> None:
        graph = self._make_graph([("A", "B", 1.0)])
        _, order_default = strength_matrix(graph)
        S_explicit, order_explicit = strength_matrix(graph, node_order=["B", "A"])
        assert order_explicit == ["B", "A"]
        # A→B only: S_default[0,1]>0; S_explicit[1,0]>0 (same edge, different indexing)
        assert S_explicit[1, 0] > 0.0


# ---------------------------------------------------------------------------
# End-to-end: random Wishart-like matrix
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_wishart_psd_preserved(self) -> None:
        """Sample from Gaussian data → correlation matrix → denoise → PSD."""
        p, n = 20, 100
        X = RNG.standard_normal((p, n))
        C = np.corrcoef(X)
        result = denoise_correlation_matrix(C, n)
        eigs = np.linalg.eigvalsh(result.matrix_cleaned)
        assert eigs.min() >= -1e-9

    def test_wishart_signal_noise_partition(self) -> None:
        p, n = 20, 100
        X = RNG.standard_normal((p, n))
        C = np.corrcoef(X)
        result = denoise_correlation_matrix(C, n)
        # Pure noise data: most eigenvalues inside MP bulk → few signal eigenvalues
        assert result.n_signal < p
        assert result.n_signal + result.n_noise == p

    def test_wishart_hhi_finite(self) -> None:
        p, n = 15, 60
        X = RNG.standard_normal((p, n))
        C = np.corrcoef(X)
        val = rmt_dependency_concentration(C, n)
        assert math.isfinite(val)
        assert 0.0 <= val <= 1.0
