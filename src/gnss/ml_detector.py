"""GNSS ML anomaly detectors (Phase 1).

Models
------
IsolationForestDetector
    Unsupervised, per-epoch.  Feature vector: [doppler_dev × n_sats, m_t, chi_t, pvt_error].
    Registry YAML: configs/model_registry/gnss_isolation_forest.yaml

LSTMDetector
    Supervised, sliding-window sequence.  Same feature vector over W epochs.
    Registry YAML: configs/model_registry/gnss_lstm_anomaly.yaml

Both detectors expose a common interface:
    fit(X, y)        — train (y ignored for IsolationForest)
    predict(X)       — return (alarm: np.ndarray[bool], score: np.ndarray[float])
    save(path) / load(path)  — persistence

Constants
---------
RANDOM_SEED          reproducibility seed
IF_N_ESTIMATORS      number of isolation trees
IF_CONTAMINATION     expected anomaly fraction for threshold calibration
LSTM_WINDOW          sequence window length W
LSTM_HIDDEN          hidden dimension
LSTM_N_LAYERS        number of stacked LSTM layers
LSTM_DROPOUT         dropout rate between layers
LSTM_LR              Adam learning rate
LSTM_EPOCHS          training epochs
LSTM_BATCH           mini-batch size
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

RANDOM_SEED: int = 42

# IsolationForest
IF_N_ESTIMATORS: int = 200
IF_CONTAMINATION: float = 0.05
IF_MAX_FEATURES: float = 1.0

# LSTM
LSTM_WINDOW: int = 16
LSTM_HIDDEN: int = 64
LSTM_N_LAYERS: int = 2
LSTM_DROPOUT: float = 0.2
LSTM_LR: float = 1e-3
LSTM_EPOCHS: int = 30
LSTM_BATCH: int = 64

# Decision threshold fallback when calibration data is absent
_DEFAULT_IF_THRESHOLD: float = 0.0  # sklearn decision_function: <0 = anomaly
_DEFAULT_LSTM_THRESHOLD: float = 0.5


# ---------------------------------------------------------------------------
# IsolationForest detector
# ---------------------------------------------------------------------------


class IsolationForestDetector:
    """Unsupervised GNSS spoofing detector using Isolation Forest.

    Invariants:
        - Fit on genuine epochs only (label == 0) for purest null distribution.
        - Decision threshold calibrated to FAR < 1e-4 on a held-out genuine set.
        - If no calibration data is provided, sklearn default (contamination) is used.
    """

    def __init__(
        self,
        n_estimators: int = IF_N_ESTIMATORS,
        contamination: float = IF_CONTAMINATION,
        max_features: float = IF_MAX_FEATURES,
        random_state: int = RANDOM_SEED,
    ) -> None:
        self._model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            max_features=max_features,
            random_state=random_state,
        )
        self._threshold: float = _DEFAULT_IF_THRESHOLD

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> IsolationForestDetector:
        """Fit on genuine samples only (label == 0).

        Args:
            X: (N, n_features) float32 feature matrix.
            y: (N,) int labels {0=genuine, 1=spoofed}.  If None, fits on all X.
        """
        X_genuine = X[y == 0] if y is not None else X
        if len(X_genuine) == 0:
            raise ValueError("No genuine samples (label=0) found for fitting.")
        self._model.fit(X_genuine)
        return self

    def calibrate_threshold(
        self,
        X_genuine: np.ndarray,
        target_far: float = 1e-4,
    ) -> float:
        """Set decision threshold so FAR ≤ target_far on a held-out genuine set.

        Uses exact discrete counting to guarantee FAR ≤ target_far:
            n_allowed = floor(N × target_far)
            threshold = sorted_scores[n_allowed - 1] + ε  (exactly n_allowed alarm)
            or  sorted_scores[0] - ε  if n_allowed == 0  (zero false alarms)

        This avoids the interpolation artefact of np.percentile when
        N × target_far < 1 (e.g. N=1333, target_far=1e-4 → n_allowed=0).

        Returns:
            Calibrated threshold value.
        """
        scores = self._model.decision_function(X_genuine)
        n = len(scores)
        n_allowed = int(np.floor(n * target_far))
        sorted_scores = np.sort(scores)
        if n_allowed == 0:
            # No false alarms allowed: set threshold strictly below minimum genuine score
            self._threshold = float(sorted_scores[0]) - 1e-9
        else:
            # At most n_allowed genuine samples trigger: threshold just above that score
            self._threshold = float(sorted_scores[n_allowed - 1]) + 1e-9
        return self._threshold

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict anomaly for each row in X.

        Returns:
            alarm: (N,) bool array — True = spoofing alarm.
            score: (N,) float array — anomaly score (higher = more anomalous).
        """
        raw = self._model.decision_function(X)  # inlier score (higher = more normal)
        anomaly_score = -raw  # flip: higher = more anomalous
        alarm = raw < self._threshold
        return alarm, anomaly_score

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self._model, "threshold": self._threshold}, path)

    @classmethod
    def load(cls, path: Path) -> IsolationForestDetector:
        state = joblib.load(Path(path))
        obj = cls.__new__(cls)
        obj._model = state["model"]
        obj._threshold = state["threshold"]
        return obj


# ---------------------------------------------------------------------------
# LSTM network definition
# ---------------------------------------------------------------------------


class _LSTMNet(nn.Module):
    """Stacked LSTM binary classifier.

    Input:  (batch, W, n_features)
    Output: (batch,) — logit for spoofing class
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = LSTM_HIDDEN,
        n_layers: int = LSTM_N_LAYERS,
        dropout: float = LSTM_DROPOUT,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, W, n_features)
        out, _ = self.lstm(x)  # (batch, W, hidden_size)
        last = out[:, -1, :]  # (batch, hidden_size) — last time step
        return self.classifier(last).squeeze(-1)  # (batch,)


# ---------------------------------------------------------------------------
# Sliding-window dataset helper
# ---------------------------------------------------------------------------


def _make_windows(
    X: np.ndarray,
    y: np.ndarray,
    run_ids: list[str],
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build sliding windows within each run.

    Each sample window is labelled by the label of the LAST epoch in the window.
    Windows that span different runs are discarded.

    Returns:
        X_win: (M, window, n_features) float32
        y_win: (M,) int64
    """
    unique_runs = dict.fromkeys(run_ids)  # preserves insertion order
    X_wins: list[np.ndarray] = []
    y_wins: list[int] = []

    for run_id in unique_runs:
        mask = np.array([r == run_id for r in run_ids])
        X_run = X[mask]
        y_run = y[mask]
        n = len(X_run)
        for i in range(window, n + 1):
            X_wins.append(X_run[i - window : i])
            y_wins.append(int(y_run[i - 1]))

    if not X_wins:
        raise ValueError(f"No windows of length {window} could be constructed.")

    return np.stack(X_wins, axis=0).astype(np.float32), np.array(y_wins, dtype=np.int64)


# ---------------------------------------------------------------------------
# LSTM detector
# ---------------------------------------------------------------------------


class LSTMDetector:
    """Supervised GNSS spoofing detector using a stacked LSTM.

    Invariants:
        - Trained on balanced windows from spoof_sim output.
        - Decision threshold calibrated to FAR < 1e-4 on held-out genuine windows.
        - Reproducible: torch seed set from RANDOM_SEED in fit().
    """

    def __init__(
        self,
        n_features: int = 9,
        window: int = LSTM_WINDOW,
        hidden_size: int = LSTM_HIDDEN,
        n_layers: int = LSTM_N_LAYERS,
        dropout: float = LSTM_DROPOUT,
        lr: float = LSTM_LR,
        n_epochs_train: int = LSTM_EPOCHS,
        batch_size: int = LSTM_BATCH,
        random_state: int = RANDOM_SEED,
    ) -> None:
        self._n_features = n_features
        self._window = window
        self._hidden_size = hidden_size
        self._n_layers = n_layers
        self._dropout = dropout
        self._lr = lr
        self._n_epochs_train = n_epochs_train
        self._batch_size = batch_size
        self._random_state = random_state
        self._threshold: float = _DEFAULT_LSTM_THRESHOLD
        self._net: _LSTMNet | None = None
        self._device = torch.device("cpu")

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        run_ids: list[str] | None = None,
    ) -> LSTMDetector:
        """Train LSTM on sliding windows of shape (window, n_features).

        Args:
            X:       (N, n_features) float32 feature matrix (all epochs in order).
            y:       (N,) int64 labels.
            run_ids: (N,) list of run identifiers to prevent cross-run windows.
                     If None, all epochs are treated as a single run.
        """
        torch.manual_seed(self._random_state)

        if run_ids is None:
            run_ids = ["run_0"] * len(X)

        X_win, y_win = _make_windows(X, y, run_ids, self._window)

        self._n_features = X.shape[1]
        self._net = _LSTMNet(
            n_features=self._n_features,
            hidden_size=self._hidden_size,
            n_layers=self._n_layers,
            dropout=self._dropout,
        ).to(self._device)

        optimizer = torch.optim.Adam(self._net.parameters(), lr=self._lr)
        criterion = nn.BCEWithLogitsLoss()

        X_t = torch.from_numpy(X_win).to(self._device)
        y_t = torch.from_numpy(y_win.astype(np.float32)).to(self._device)

        n_samples = len(X_t)
        self._net.train()
        rng_idx = np.random.default_rng(self._random_state)

        for _ in range(self._n_epochs_train):
            perm = rng_idx.permutation(n_samples)
            for start in range(0, n_samples, self._batch_size):
                idx = perm[start : start + self._batch_size]
                logits = self._net(X_t[idx])
                loss = criterion(logits, y_t[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return self

    def calibrate_threshold(
        self,
        X: np.ndarray,
        run_ids: list[str] | None = None,
        target_far: float = 1e-4,
    ) -> float:
        """Calibrate decision threshold so FAR ≤ target_far on genuine windows.

        Returns:
            Calibrated threshold (probability).
        """
        if self._net is None:
            raise RuntimeError("Call fit() before calibrate_threshold().")

        if run_ids is None:
            run_ids = ["run_0"] * len(X)

        # Build genuine-only windows
        y_dummy = np.zeros(len(X), dtype=np.int64)
        X_win, _ = _make_windows(X, y_dummy, run_ids, self._window)

        probs = self._predict_proba(X_win)
        self._threshold = float(np.percentile(probs, 100.0 * (1.0 - target_far)))
        return self._threshold

    def _predict_proba(self, X_win: np.ndarray) -> np.ndarray:
        """Return spoofing probabilities for pre-built windows."""
        assert self._net is not None
        self._net.eval()
        with torch.no_grad():
            X_t = torch.from_numpy(X_win.astype(np.float32)).to(self._device)
            logits = self._net(X_t)
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs.astype(np.float32)

    def predict(
        self,
        X: np.ndarray,
        run_ids: list[str] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict spoofing probability for each window ending at each epoch.

        Returns:
            alarm: (M,) bool array.
            score: (M,) float array — spoofing probability ∈ [0, 1].
        """
        if self._net is None:
            raise RuntimeError("Call fit() before predict().")

        if run_ids is None:
            run_ids = ["run_0"] * len(X)

        y_dummy = np.zeros(len(X), dtype=np.int64)
        X_win, _ = _make_windows(X, y_dummy, run_ids, self._window)

        probs = self._predict_proba(X_win)
        alarm = probs >= self._threshold
        return alarm, probs

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state: dict[str, Any] = {
            "n_features": self._n_features,
            "window": self._window,
            "hidden_size": self._hidden_size,
            "n_layers": self._n_layers,
            "dropout": self._dropout,
            "lr": self._lr,
            "n_epochs_train": self._n_epochs_train,
            "batch_size": self._batch_size,
            "random_state": self._random_state,
            "threshold": self._threshold,
            "net_state": self._net.state_dict() if self._net is not None else None,
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path: Path) -> LSTMDetector:
        state = torch.load(Path(path), map_location="cpu", weights_only=True)
        obj = cls(
            n_features=state["n_features"],
            window=state["window"],
            hidden_size=state["hidden_size"],
            n_layers=state["n_layers"],
            dropout=state["dropout"],
            lr=state["lr"],
            n_epochs_train=state["n_epochs_train"],
            batch_size=state["batch_size"],
            random_state=state["random_state"],
        )
        obj._threshold = state["threshold"]
        if state["net_state"] is not None:
            obj._net = _LSTMNet(
                n_features=state["n_features"],
                hidden_size=state["hidden_size"],
                n_layers=state["n_layers"],
                dropout=state["dropout"],
            )
            obj._net.load_state_dict(state["net_state"])
        return obj
