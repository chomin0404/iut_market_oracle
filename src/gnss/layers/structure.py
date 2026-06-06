"""GNSS Resilience Twin — Layers 3, 8, 10: Structure Pillar components.

Layer 3  — Spectral Graph Monitor: Fiedler ratio + spectral entropy + RMT anomaly
Layer 8  — Structural Dependency Monitor: persistent graph-topology tracker
Layer 10 — Duminil-Copin Phase-Transition Monitor: percolation susceptibility
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from gnss.constants import _DOPPLER_NOISE_STD, _GRAPH_SIGMA
from gnss.math_utils import _build_graph

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EPS: float = 1e-300

# Layer 8 — Structural dependency monitor
_STRUCT_STREAK_THRESH: int = 3
_STRUCT_CHANGE_THRESH: float = 0.50
_STRUCT_CLUSTER_WEIGHT_THRESH: float = 0.50

# Layer 10 — Duminil-Copin percolation phase-transition monitor
_DC_N_THRESH_POINTS: int = 41
_DC_SUSCEPTIBILITY_ALERT: float = 10.0
_DC_NULL_THRESHOLD: float = 0.90
_DC_MIN_W_THRESHOLD: float = 0.95

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpectralResult:
    """Output of spectral graph monitor per epoch."""

    fiedler_ratio: float  # ρ_F = λ₂ / λ₂_null
    spectral_entropy: float  # H_spec [nats]
    rmt_anomaly: float  # mean((λₖ − λ₂_null)²) / λ₂_null² for k≥2


@dataclass(frozen=True)
class StructuralMonitorResult:
    """Output of structural dependency monitor (Layer 8).

    Tracks persistent graph-level anomalies across consecutive epochs.
    """

    fiedler_streak: int  # consecutive epochs with Fiedler-ratio anomaly
    graph_change_rate: float  # ‖W_t − W_{t−1}‖_F / ‖W_{t−1}‖_F
    clustering_coeff: float  # mean clustering coefficient of thresholded graph
    alert: bool  # True if streak ≥ threshold or change_rate > threshold


@dataclass(frozen=True)
class PhaseTransitionResult:
    """Output of Duminil-Copin percolation phase-transition monitor (Layer 10).

    Sweeps threshold τ ∈ [0,1] on the satellite similarity graph W:
        A_τ[i,j] = 1  iff  w_ij > τ
        LCC(τ)   = fraction of nodes in the largest connected component
        χ(τ)     = |ΔLCC / Δτ|  — susceptibility (peaks at the phase transition)

    Under nominal conditions (isolated nodes only): χ_peak ≈ 1/(n·Δτ) ≈ 6.7
    Under coordinated spoofing (synchronised collapse): χ_peak >> 10

    Theoretical basis: Duminil-Copin et al. (2020) — sharp phase transitions in
    dependent percolation models; susceptibility peak is a universal indicator.
    """

    percolation_threshold: float  # τ* where χ is maximised
    susceptibility_peak: float  # max χ over the τ sweep
    lcc_at_null: float  # LCC(τ = _DC_NULL_THRESHOLD)
    min_edge_weight: float  # min off-diagonal w_ij — near 1 ↔ tight common-mode attack
    phase_alert: bool  # True if χ_peak > thresh AND min_w > _DC_MIN_W_THRESHOLD


# ---------------------------------------------------------------------------
# Layer 3 — Spectral Graph Monitor
# ---------------------------------------------------------------------------


class SpectralMonitor:
    """Spectral anomaly detection on the satellite similarity graph.

    Under H₀ (all σᵢ = σ_D, complete symmetric graph):
        w_null = 1 / √(1 + 4σ_D²/σ²)    [expected edge weight]
        λ₂_null = n · w_null              [null Fiedler value]

    Metrics:
        ρ_F   = λ₂ / λ₂_null             — Fiedler ratio (>1 → anomaly)
        H_spec = −Σₖ pₖ ln pₖ            — spectral entropy of non-trivial eigenvalues
        rmt    = mean((λₖ − λ₂_null)²) / λ₂_null²   — RMT deviation (k ≥ 2)
    """

    def __init__(
        self,
        n_sats: int,
        noise_std: float = _DOPPLER_NOISE_STD,
        graph_sigma: float = _GRAPH_SIGMA,
    ) -> None:
        self._sigma = graph_sigma
        w_null = 1.0 / math.sqrt(1.0 + 4.0 * noise_std**2 / graph_sigma**2)
        self._lambda2_null = n_sats * w_null

    def analyze(self, doppler_dev: np.ndarray) -> SpectralResult:
        """Compute spectral metrics from current Doppler deviations.

        Args:
            doppler_dev: (n,) Doppler residuals [Hz]
        """
        W = _build_graph(doppler_dev, self._sigma)
        L = np.diag(W.sum(axis=1)) - W
        ev = np.sort(np.linalg.eigvalsh(L))

        lambda2 = float(ev[1]) if len(ev) > 1 else 0.0
        fiedler_ratio = lambda2 / (self._lambda2_null + _EPS)

        ev_pos = np.maximum(ev[1:], 0.0)
        total = ev_pos.sum()
        if total > _EPS:
            p = ev_pos / total
            spectral_entropy = float(-np.sum(p * np.log(np.where(p > _EPS, p, _EPS))))
        else:
            spectral_entropy = 0.0

        rmt_anomaly = float(
            np.mean((ev[1:] - self._lambda2_null) ** 2) / (self._lambda2_null**2 + _EPS)
        )

        return SpectralResult(
            fiedler_ratio=fiedler_ratio,
            spectral_entropy=spectral_entropy,
            rmt_anomaly=rmt_anomaly,
        )


# ---------------------------------------------------------------------------
# Layer 8 — Structural Dependency Monitor
# ---------------------------------------------------------------------------


class StructuralDependencyMonitor:
    """Structural dependency anomaly tracker across consecutive epochs (Layer 8).

    Monitors persistent graph-topology changes that signal coordinated
    multi-satellite manipulation (meaconing):

        fiedler_streak:   consecutive epochs where ρ_F > 1.0
        graph_change_rate: ‖W_t − W_{t−1}‖_F / (‖W_{t−1}‖_F + ε)
        clustering_coeff:  mean clustering coefficient of thresholded graph

    Alert fires when: streak ≥ threshold OR change_rate > threshold.
    """

    def __init__(
        self,
        noise_std: float = _DOPPLER_NOISE_STD,
        graph_sigma: float = _GRAPH_SIGMA,
        streak_thresh: int = _STRUCT_STREAK_THRESH,
        change_thresh: float = _STRUCT_CHANGE_THRESH,
        cluster_weight_thresh: float = _STRUCT_CLUSTER_WEIGHT_THRESH,
    ) -> None:
        self._graph_sigma = graph_sigma
        self._streak_thresh = streak_thresh
        self._change_thresh = change_thresh
        self._cluster_w_thresh = cluster_weight_thresh
        self._streak: int = 0
        self._prev_W: np.ndarray | None = None

    def update(self, doppler_dev: np.ndarray, fiedler_anomaly: bool) -> StructuralMonitorResult:
        """Update structural monitor with current epoch's Doppler observations.

        Args:
            doppler_dev:     (n,) Doppler residuals [Hz]
            fiedler_anomaly: True if ρ_F > 1.0 this epoch
        """
        W = _build_graph(doppler_dev, self._graph_sigma)

        if fiedler_anomaly:
            self._streak += 1
        else:
            self._streak = 0

        if self._prev_W is not None:
            frob_prev = float(np.linalg.norm(self._prev_W, "fro"))
            frob_diff = float(np.linalg.norm(W - self._prev_W, "fro"))
            graph_change_rate = frob_diff / (frob_prev + _EPS)
        else:
            graph_change_rate = 0.0
        self._prev_W = W.copy()

        A = (W > self._cluster_w_thresh).astype(float)
        np.fill_diagonal(A, 0.0)
        degree = A.sum(axis=1)
        mask = degree >= 2
        if mask.any():
            A3_diag = (A @ A * A).sum(axis=1)
            denom = np.where(mask, degree * (degree - 1), 1.0)
            cc_vals = np.where(mask, A3_diag / denom, 0.0)
            clustering_coeff = float(cc_vals[mask].mean())
        else:
            clustering_coeff = 0.0

        alert = self._streak >= self._streak_thresh or graph_change_rate > self._change_thresh
        return StructuralMonitorResult(
            fiedler_streak=self._streak,
            graph_change_rate=graph_change_rate,
            clustering_coeff=clustering_coeff,
            alert=alert,
        )


# ---------------------------------------------------------------------------
# Layer 10 helpers
# ---------------------------------------------------------------------------


def _lcc_curve_batch(W: np.ndarray, tau_grid: np.ndarray, n: int) -> np.ndarray:
    """LCC fraction for every threshold in tau_grid via batched boolean transitive closure.

    Builds the (K, n, n) adjacency stack and finds the reachability matrix for
    each threshold using ceil(log₂ n) repeated squarings of (I + A).
    """
    A = (W[None, :, :] > tau_grid[:, None, None]).astype(np.uint8)
    diag = np.arange(n)
    A[:, diag, diag] = 0
    R = A.copy()
    R[:, diag, diag] = 1
    n_sq = max(int(np.ceil(np.log2(max(n, 2)))), 1)
    for _ in range(n_sq):
        R = (np.matmul(R.astype(np.int16), R.astype(np.int16)) > 0).astype(np.uint8)
    return (R.sum(axis=2).max(axis=1) / n).astype(float)


# ---------------------------------------------------------------------------
# Layer 10 — Duminil-Copin Phase-Transition Monitor
# ---------------------------------------------------------------------------


class DuminilCopinPhaseMonitor:
    """Percolation phase-transition monitor on the satellite similarity graph (Layer 10).

    Sweeps threshold τ ∈ [0,1] on the satellite similarity graph W_ij = exp(-|Δfᵢ−Δfⱼ|²/σ²):
        A_τ[i,j] = 1  iff  w_ij > τ
        LCC(τ)   = |largest connected component| / n_sats
        χ(τ)     = |ΔLCC(τ) / Δτ|  — susceptibility

    A sharp χ_peak marks the percolation threshold (τ*).  Coordinated spoofing
    collapses all edge-weights at once → synchronised χ_peak >> 10.
    An isolated HW fault removes at most 1 node → χ_peak ≈ (1/n)/Δτ ≈ 6.7 < 10.

    Alert threshold: χ_peak > _DC_SUSCEPTIBILITY_ALERT = 10.0.
    """

    def __init__(self, graph_sigma: float = _GRAPH_SIGMA) -> None:
        self._graph_sigma = graph_sigma
        self._tau_grid = np.linspace(0.0, 1.0, _DC_N_THRESH_POINTS)

    def update(self, doppler_dev: np.ndarray) -> PhaseTransitionResult:
        """Compute percolation susceptibility for current epoch.

        Args:
            doppler_dev: (n,) Doppler residuals [Hz]
        """
        W = _build_graph(doppler_dev, self._graph_sigma)
        n = W.shape[0]

        lcc_curve = _lcc_curve_batch(W, self._tau_grid, n)

        delta_tau = float(self._tau_grid[1] - self._tau_grid[0])
        chi = np.abs(np.diff(lcc_curve)) / delta_tau

        chi_peak = float(chi.max()) if len(chi) > 0 else 0.0
        peak_idx = int(np.argmax(chi))
        percolation_threshold = float(
            0.5 * (self._tau_grid[peak_idx] + self._tau_grid[peak_idx + 1])
        )

        null_idx = min(
            int(np.searchsorted(self._tau_grid, _DC_NULL_THRESHOLD)),
            len(self._tau_grid) - 1,
        )
        lcc_at_null = float(lcc_curve[null_idx])

        if n > 1:
            W_off = W.copy()
            np.fill_diagonal(W_off, 1.0)
            min_w = float(W_off.min())
        else:
            min_w = 0.0

        phase_alert = chi_peak > _DC_SUSCEPTIBILITY_ALERT and min_w > _DC_MIN_W_THRESHOLD

        return PhaseTransitionResult(
            percolation_threshold=percolation_threshold,
            susceptibility_peak=chi_peak,
            lcc_at_null=lcc_at_null,
            min_edge_weight=min_w,
            phase_alert=phase_alert,
        )
