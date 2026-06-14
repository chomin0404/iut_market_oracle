"""Direct tests for src/gnss/math_utils.py (T1300 / T1500)."""

from __future__ import annotations

import numpy as np

from gnss.math_utils import (
    _ROC_N_THRESHOLDS,
    build_graph,
    compute_roc,
    geometry_matrix,
    init_constellation,
)

# ---------------------------------------------------------------------------
# init_constellation
# ---------------------------------------------------------------------------


class TestInitConstellation:
    def test_shape(self) -> None:
        los = init_constellation(8)
        assert los.shape == (8, 3)

    def test_unit_vectors(self) -> None:
        los = init_constellation(6)
        norms = np.linalg.norm(los, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-10)

    def test_upper_hemisphere(self) -> None:
        # z > 0 for all satellites (elevation > 0)
        los = init_constellation(10)
        assert np.all(los[:, 2] > 0)

    def test_deterministic(self) -> None:
        a = init_constellation(6)
        b = init_constellation(6)
        assert np.array_equal(a, b)

    def test_min_4_satellites(self) -> None:
        los = init_constellation(4)
        assert los.shape == (4, 3)

    def test_large_constellation(self) -> None:
        los = init_constellation(50)
        assert los.shape == (50, 3)
        norms = np.linalg.norm(los, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# build_graph
# ---------------------------------------------------------------------------


class TestBuildGraph:
    def test_shape(self) -> None:
        doppler = np.array([0.1, 0.2, 0.3, 0.4])
        W = build_graph(doppler, sigma=1.5)
        assert W.shape == (4, 4)

    def test_diagonal_zero(self) -> None:
        doppler = np.array([0.1, 0.2, 0.5])
        W = build_graph(doppler, sigma=1.5)
        assert np.all(np.diag(W) == 0.0)

    def test_symmetric(self) -> None:
        doppler = np.array([0.0, 1.0, -0.5, 2.0])
        W = build_graph(doppler, sigma=1.0)
        assert np.allclose(W, W.T)

    def test_weights_in_0_1(self) -> None:
        doppler = np.random.default_rng(0).standard_normal(8)
        W = build_graph(doppler, sigma=1.5)
        assert np.all(W >= 0.0)
        assert np.all(W <= 1.0)

    def test_identical_doppler_gives_weight_one(self) -> None:
        # Off-diagonal: |Δf_i - Δf_j| = 0 → w_ij = exp(0) = 1
        doppler = np.array([1.0, 1.0, 1.0])
        W = build_graph(doppler, sigma=1.0)
        np.fill_diagonal(W, 999)  # ignore diagonal
        off = W[W != 999]
        assert np.allclose(off, 1.0)

    def test_large_difference_gives_small_weight(self) -> None:
        doppler = np.array([0.0, 100.0])
        W = build_graph(doppler, sigma=1.0)
        assert W[0, 1] < 1e-10
        assert W[1, 0] < 1e-10

    def test_sigma_affects_weights(self) -> None:
        doppler = np.array([0.0, 1.0])
        W_narrow = build_graph(doppler, sigma=0.5)
        W_wide = build_graph(doppler, sigma=5.0)
        # Wider sigma → larger weight for the same difference
        assert W_wide[0, 1] > W_narrow[0, 1]


# ---------------------------------------------------------------------------
# geometry_matrix
# ---------------------------------------------------------------------------


class TestGeometryMatrix:
    def test_shape(self) -> None:
        los = init_constellation(6)
        S = [0, 1, 2, 3]
        H = geometry_matrix(los, S)
        assert H.shape == (4, 4)

    def test_subset_size(self) -> None:
        los = init_constellation(8)
        S = [0, 2, 5]
        H = geometry_matrix(los, S)
        assert H.shape == (3, 4)

    def test_scale_consistent(self) -> None:
        from gnss.constants import _L1_FREQ, _SPEED_OF_LIGHT

        los = init_constellation(4)
        S = [0, 1]
        H = geometry_matrix(los, S)
        expected_scale = _L1_FREQ / _SPEED_OF_LIGHT
        # Last column is -scale repeated
        assert np.allclose(H[:, 3], -expected_scale)

    def test_los_rows_match_subset(self) -> None:
        from gnss.constants import _L1_FREQ, _SPEED_OF_LIGHT

        los = init_constellation(6)
        S = [1, 4]
        H = geometry_matrix(los, S)
        scale = _L1_FREQ / _SPEED_OF_LIGHT
        # First 3 columns: -scale * los[S, :]
        expected = -scale * los[S, :]
        assert np.allclose(H[:, :3], expected)

    def test_full_constellation_subset(self) -> None:
        los = init_constellation(4)
        S = list(range(4))
        H = geometry_matrix(los, S)
        assert H.shape == (4, 4)


# ---------------------------------------------------------------------------
# compute_roc
# ---------------------------------------------------------------------------


class TestComputeRoc:
    def test_return_types(self) -> None:
        scores = np.array([0.1, 0.5, 0.9, 0.3])
        labels = np.array([0, 1, 1, 0])
        fpr, tpr, auc = compute_roc(scores, labels)
        assert isinstance(fpr, list)
        assert isinstance(tpr, list)
        assert isinstance(auc, float)

    def test_length(self) -> None:
        scores = np.linspace(0, 1, 50)
        labels = (scores > 0.5).astype(int)
        fpr, tpr, auc = compute_roc(scores, labels)
        assert len(fpr) == _ROC_N_THRESHOLDS
        assert len(tpr) == _ROC_N_THRESHOLDS

    def test_auc_in_range(self) -> None:
        rng = np.random.default_rng(0)
        scores = rng.standard_normal(100)
        labels = (scores > 0).astype(int)
        _, _, auc = compute_roc(scores, labels)
        assert 0.0 <= auc <= 1.0

    def test_perfect_classifier(self) -> None:
        # Discrete threshold grid limits AUC to ~0.875 with 4 data points
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        labels = np.array([0, 0, 1, 1])
        _, _, auc = compute_roc(scores, labels)
        assert auc > 0.8

    def test_uniform_scores_returns_half_auc(self) -> None:
        # All scores identical: s_min >= s_max branch → AUC = 0.5
        scores = np.ones(10)
        labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        fpr, tpr, auc = compute_roc(scores, labels)
        assert auc == 0.5
        assert fpr == [0.0, 1.0]
        assert tpr == [0.0, 1.0]

    def test_single_class_labels(self) -> None:
        # All genuine — no positives, TPR = 0 everywhere
        scores = np.array([0.1, 0.5, 0.9])
        labels = np.array([0, 0, 0])
        fpr, tpr, auc = compute_roc(scores, labels)
        assert all(t == 0.0 for t in tpr)

    def test_auc_clamped(self) -> None:
        # AUC must always be within [0, 1]
        scores = np.array([1.0, 0.0])
        labels = np.array([0, 1])  # inverted — random classifier
        _, _, auc = compute_roc(scores, labels)
        assert 0.0 <= auc <= 1.0
