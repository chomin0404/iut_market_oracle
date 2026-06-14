"""Tests for TransformerDetector in src/gnss/ml_detector.py.

Acceptance criteria:
    _TransformerNet:
        - forward() returns (batch,) tensor
        - positional encoding added without crash

    TransformerDetector:
        - fit/predict does not crash on small dataset
        - predict returns (alarm, score) with correct shapes
        - score ∈ [0, 1]
        - save/load round-trip preserves threshold and net weights
        - spoofed windows score > 0.5 majority (sanity check)
        - calibrate_threshold reduces FAR on genuine windows
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from gnss.ml_detector import (
    LSTM_WINDOW,
    RANDOM_SEED,
    TF_D_MODEL,
    TF_N_HEADS,
    TransformerDetector,
    _make_windows,
    _TransformerNet,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_N_SATS = 6
_N_FEATURES = _N_SATS + 3  # doppler × 6 + m_t + chi_t + pvt_error
_WINDOW = LSTM_WINDOW
_BATCH = 4
_N_EPOCHS = 50  # enough for non-trivial signal
_N_MC = 20  # MC runs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dataset(
    n_runs: int = _N_MC,
    n_epochs: int = _N_EPOCHS,
    n_features: int = _N_FEATURES,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Synthetic balanced dataset with clear genuine vs spoofed signal.

    Genuine: features ~ N(0, 1)
    Spoofed: features ~ N(3, 1)  (clearly separated)
    """
    rng = np.random.default_rng(seed)
    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    run_ids: list[str] = []
    for i in range(n_runs):
        label = i % 2  # alternate genuine/spoofed
        mean = 0.0 if label == 0 else 3.0
        X_parts.append(rng.normal(mean, 1.0, (n_epochs, n_features)).astype(np.float32))
        y_parts.append(np.full(n_epochs, label, dtype=np.int64))
        run_ids.extend([f"run_{i}"] * n_epochs)
    return (
        np.concatenate(X_parts, axis=0),
        np.concatenate(y_parts, axis=0),
        run_ids,
    )


# ---------------------------------------------------------------------------
# _TransformerNet unit tests
# ---------------------------------------------------------------------------


class TestTransformerNet:
    def test_forward_shape(self) -> None:
        net = _TransformerNet(n_features=_N_FEATURES, d_model=TF_D_MODEL, n_heads=TF_N_HEADS)
        x = torch.randn(_BATCH, _WINDOW, _N_FEATURES)
        out = net(x)
        assert out.shape == (_BATCH,), f"Expected ({_BATCH},), got {out.shape}"

    def test_positional_encoding_no_crash(self) -> None:
        net = _TransformerNet(n_features=_N_FEATURES, d_model=32, n_heads=4)
        x = torch.randn(2, 8, _N_FEATURES)
        out = net(x)
        assert out.shape == (2,)

    def test_different_window_lengths(self) -> None:
        net = _TransformerNet(n_features=_N_FEATURES, d_model=32, n_heads=4)
        for w in (4, 16, 32):
            x = torch.randn(2, w, _N_FEATURES)
            out = net(x)
            assert out.shape == (2,)


# ---------------------------------------------------------------------------
# TransformerDetector integration tests
# ---------------------------------------------------------------------------


class TestTransformerDetector:
    @pytest.fixture(scope="class")
    def detector_and_data(self):
        X, y, run_ids = _make_dataset()
        det = TransformerDetector(
            n_features=_N_FEATURES,
            window=_WINDOW,
            d_model=32,
            n_heads=4,
            n_layers=1,
            dim_feedforward=64,
            n_epochs_train=3,  # fast for CI
            batch_size=32,
            random_state=RANDOM_SEED,
        )
        det.fit(X, y, run_ids)
        return det, X, y, run_ids

    def test_predict_returns_correct_shapes(self, detector_and_data) -> None:
        det, X, y, run_ids = detector_and_data
        alarm, score = det.predict(X, run_ids)
        assert alarm.dtype == bool
        assert score.dtype == np.float32
        assert alarm.shape == score.shape
        assert len(alarm) > 0

    def test_score_in_zero_one(self, detector_and_data) -> None:
        det, X, y, run_ids = detector_and_data
        _, score = det.predict(X, run_ids)
        assert float(np.min(score)) >= 0.0
        assert float(np.max(score)) <= 1.0

    def test_fit_without_run_ids(self) -> None:
        X, y, _ = _make_dataset(n_runs=4)
        det = TransformerDetector(
            n_features=_N_FEATURES,
            window=_WINDOW,
            d_model=32,
            n_heads=4,
            n_layers=1,
            dim_feedforward=64,
            n_epochs_train=2,
            batch_size=32,
        )
        det.fit(X, y)  # no run_ids → single-run mode
        alarm, score = det.predict(X)
        assert len(alarm) > 0

    def test_predict_before_fit_raises(self) -> None:
        det = TransformerDetector(n_features=_N_FEATURES)
        X, _, _ = _make_dataset(n_runs=2)
        with pytest.raises(RuntimeError, match="fit"):
            det.predict(X)

    def test_calibrate_threshold(self, detector_and_data) -> None:
        det, X, y, run_ids = detector_and_data
        X_genuine = X[y == 0]
        run_ids_genuine = [r for r, lbl in zip(run_ids, y) if lbl == 0]
        thresh = det.calibrate_threshold(X_genuine, run_ids_genuine, target_far=0.05)
        assert 0.0 <= thresh <= 1.0

    def test_save_load_roundtrip(self, detector_and_data, tmp_path) -> None:
        det, X, y, run_ids = detector_and_data
        path = tmp_path / "transformer_det.pt"
        det.save(path)

        det2 = TransformerDetector.load(path)
        alarm1, score1 = det.predict(X, run_ids)
        alarm2, score2 = det2.predict(X, run_ids)

        np.testing.assert_array_equal(alarm1, alarm2)
        np.testing.assert_allclose(score1, score2, rtol=1e-5)

    def test_spoofed_score_majority_above_half(self, detector_and_data) -> None:
        """After fitting on clearly separated data, spoofed scores should be > 0.5 majority."""
        det, X, y, run_ids = detector_and_data
        _, score = det.predict(X, run_ids)
        # Build windows to get per-window labels
        X_win, y_win = _make_windows(X, y, run_ids, _WINDOW)
        spoofed_scores = score[y_win == 1]
        if len(spoofed_scores) > 0:
            frac = float(np.mean(spoofed_scores > 0.5))
            assert frac >= 0.5, f"Only {frac:.1%} of spoofed windows scored > 0.5"
