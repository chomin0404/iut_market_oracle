"""GNSS Resilience Report Layer (T1500 — diagnostic visualisation).

Generates a 5-panel diagnostic figure from EdgeArrays and optional pipeline
history, covering all key signal layers:

    Panel 1  攻撃証拠       fault_posterior time series + alert event markers
    Panel 2  残差推移       Doppler residual heatmap (epochs × sats)
    Panel 3  エントロピー変化 IMM mode weights + Shannon entropy + entropy alerts
    Panel 4  選択 subset    satellite weight heatmap + n_active / n_excluded
    Panel 5  推定信頼度     confidence + ins_weight + failsafe bands + mc_auc

Typical usage::

    from gnss.edge_collector import EdgeCollector
    from gnss.report import plot_gnss_report

    collector = EdgeCollector()
    collector.collect_all(pipeline.history)
    fig = plot_gnss_report(
        collector.to_arrays(),
        history=pipeline.history,
        title="Spoofing scenario — exp-001",
        save_path="reports/exp-001/gnss_diagnostic.png",
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from gnss.edge_collector import EdgeArrays

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from gnss.mvp import _EpochRecord

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

_FP_COLORS = ["#2ca02c", "#ff7f0e", "#1f77b4", "#d62728"]
_FP_LABELS = ["P(NOMINAL)", "P(MULTIPATH)", "P(HW_FAULT)", "P(SPOOFING)"]

_ALERT_COLORS = {
    "info": "#aaaaaa",
    "caution": "#ff7f0e",
    "warning": "#d62728",
    "critical": "#9467bd",
}

# Background band colours for failsafe level (alpha=0.15 applied at draw time)
_FAILSAFE_BAND_COLORS = {
    "nominal": None,                  # no shading
    "degraded": "#ffcc00",
    "ins_only": "#ff4444",
    "dead_reckoning": "#888888",
}

_IMM_COLORS = ["#2ca02c", "#ff7f0e", "#d62728"]
_IMM_LABELS = ["μ_nom", "μ_mp", "μ_spoof"]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _shannon_entropy(weights: np.ndarray) -> np.ndarray:
    """Per-row Shannon entropy of a (T, K) probability matrix.

    H(row) = -Σ_k w_k · log(w_k),  with 0·log(0) := 0.

    Returns
    -------
    np.ndarray of shape (T,)
    """
    safe = np.where(weights > 0.0, weights, 1.0)
    return -np.sum(weights * np.log(safe), axis=-1)


def _extract_satellite_weights(
    history: list[_EpochRecord], n_sats: int
) -> np.ndarray:
    """Return (n_epochs, n_sats) float64 satellite weight array from history."""
    out = np.zeros((len(history), n_sats), dtype=np.float64)
    for i, rec in enumerate(history):
        ws = rec.action.satellite_weights
        out[i, : len(ws)] = ws
    return out


def _extract_failsafe_levels(history: list[_EpochRecord]) -> list[str]:
    """Return per-epoch failsafe level strings from action history."""
    return [rec.action.failsafe.level.value for rec in history]


def _extract_alert_levels(history: list[_EpochRecord]) -> list[str]:
    """Return per-epoch alert level strings from action history."""
    return [rec.action.alert.level.value for rec in history]


def _add_failsafe_bands(ax: Axes, epochs: np.ndarray, levels: list[str]) -> None:
    """Shade background of *ax* by failsafe severity level."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    ylim = ax.get_ylim()
    prev_lvl = levels[0]
    start_e = int(epochs[0])

    def _shade(e_start: int, e_end: int, lvl: str) -> None:
        color = _FAILSAFE_BAND_COLORS.get(lvl)
        if color is not None:
            ax.axvspan(e_start - 0.5, e_end - 0.5, color=color, alpha=0.12, zorder=0)

    for i, lvl in enumerate(levels[1:], start=1):
        if lvl != prev_lvl:
            _shade(start_e, int(epochs[i - 1]) + 1, prev_lvl)
            start_e = int(epochs[i])
            prev_lvl = lvl
    _shade(start_e, int(epochs[-1]) + 1, prev_lvl)
    ax.set_ylim(ylim)
    del plt  # imported only for type hint; axvspan via axes object


def _add_event_vlines(
    ax: Axes,
    epochs: np.ndarray,
    alert_levels: list[str],
) -> None:
    """Draw vertical event lines coloured by alert severity."""
    for e, lvl in zip(epochs, alert_levels):
        if lvl in ("caution", "warning", "critical"):
            ax.axvline(
                x=int(e),
                color=_ALERT_COLORS[lvl],
                linewidth=0.8,
                alpha=0.6,
                zorder=5,
            )


# ---------------------------------------------------------------------------
# Panel builders
# ---------------------------------------------------------------------------


def _panel_attack_evidence(
    ax: Axes,
    arrays: EdgeArrays,
    alert_levels: list[str] | None,
    failsafe_levels: list[str] | None,
) -> None:
    """Panel 1: fault_posterior time series + alert event markers."""
    import matplotlib.patches as mpatches  # noqa: PLC0415

    ep = arrays.epochs
    fp = arrays.fault_posterior

    for j, (col, lbl) in enumerate(zip(_FP_COLORS, _FP_LABELS)):
        ax.plot(ep, fp[:, j], color=col, linewidth=1.5, label=lbl)

    # P(SPOOFING) fill
    ax.fill_between(ep, fp[:, 3], alpha=0.20, color=_FP_COLORS[3])

    if failsafe_levels is not None:
        _add_failsafe_bands(ax, ep, failsafe_levels)
    if alert_levels is not None:
        _add_event_vlines(ax, ep, alert_levels)

    # Legend for alert markers
    patches = [
        mpatches.Patch(color=_ALERT_COLORS[k], label=k.upper())
        for k in ("caution", "warning", "critical")
    ]
    ax.legend(
        handles=list(ax.get_lines()) + patches,
        loc="upper right",
        fontsize=7,
        ncol=4,
    )
    ax.set_ylabel("Probability", fontsize=9)
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("攻撃証拠 — Fault Posterior", fontsize=10, fontweight="bold")
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)


def _panel_doppler_residuals(
    ax: Axes,
    arrays: EdgeArrays,
) -> None:
    """Panel 2: Doppler residual heatmap (epochs × sats)."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    ep = arrays.epochs
    n_sats = arrays.n_sats
    doppler = arrays.doppler_residuals  # (T, S)

    vmax = max(float(np.abs(doppler).max()), 0.01)
    im = ax.imshow(
        doppler.T,
        aspect="auto",
        origin="lower",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        extent=[float(ep[0]) - 0.5, float(ep[-1]) + 0.5, -0.5, n_sats - 0.5],
    )
    cb = plt.colorbar(im, ax=ax, pad=0.01)
    cb.set_label("Doppler residual [Hz]", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    ax.set_yticks(range(n_sats))
    ax.set_yticklabels([f"SV{i}" for i in range(n_sats)], fontsize=7)
    ax.set_ylabel("Satellite", fontsize=9)
    ax.set_title("残差推移 — Doppler Residuals (epochs × sats)", fontsize=10, fontweight="bold")


def _panel_entropy(
    ax: Axes,
    arrays: EdgeArrays,
) -> None:
    """Panel 3: IMM mode weights + Shannon entropy + entropy alert markers."""
    ep = arrays.epochs
    mw = arrays.imm_mode_weights  # (T, 3)
    entropy = _shannon_entropy(mw)  # (T,)

    # Stacked area
    ax.stackplot(
        ep,
        mw[:, 0],
        mw[:, 1],
        mw[:, 2],
        labels=_IMM_LABELS,
        colors=_IMM_COLORS,
        alpha=0.65,
    )

    # Shannon entropy on secondary y-axis
    ax2 = ax.twinx()
    ax2.plot(ep, entropy, color="#7f7f7f", linewidth=1.2, linestyle="--", label="H(μ)")
    ax2.set_ylabel("H(μ) [nats]", fontsize=8, color="#7f7f7f")
    ax2.tick_params(axis="y", labelcolor="#7f7f7f", labelsize=7)
    ax2.set_ylim(0.0, float(np.log(3)) * 1.15)

    # Entropy alert markers
    ea_epochs = ep[arrays.entropy_alert]
    if len(ea_epochs) > 0:
        ax2.scatter(
            ea_epochs,
            entropy[arrays.entropy_alert],
            marker="^",
            color="#d62728",
            zorder=6,
            s=30,
            label="entropy_alert",
        )

    ax.set_ylabel("IMM mode weight", fontsize=9)
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="upper left", fontsize=7)
    ax2.legend(loc="upper right", fontsize=7)
    ax.set_title("エントロピー変化 — IMM Mode Weights & H(μ)", fontsize=10, fontweight="bold")
    ax.grid(axis="y", linewidth=0.4, alpha=0.4)


def _panel_subset(
    ax_heat: Axes,
    ax_count: Axes,
    arrays: EdgeArrays,
    sat_weights: np.ndarray | None,
) -> None:
    """Panel 4: satellite weight heatmap (left) + n_active step (right)."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    ep = arrays.epochs
    n_sats = arrays.n_sats

    # Left: heatmap
    if sat_weights is not None:
        im = ax_heat.imshow(
            sat_weights.T,
            aspect="auto",
            origin="lower",
            cmap="YlGn",
            vmin=0.0,
            vmax=1.0,
            extent=[float(ep[0]) - 0.5, float(ep[-1]) + 0.5, -0.5, n_sats - 0.5],
        )
        cb = plt.colorbar(im, ax=ax_heat, pad=0.01)
        cb.set_label("weight", fontsize=7)
        cb.ax.tick_params(labelsize=6)
    else:
        # Fallback: show gmm_gamma
        im = ax_heat.imshow(
            arrays.gmm_gamma.T,
            aspect="auto",
            origin="lower",
            cmap="Reds",
            vmin=0.0,
            vmax=1.0,
            extent=[float(ep[0]) - 0.5, float(ep[-1]) + 0.5, -0.5, n_sats - 0.5],
        )
        cb = plt.colorbar(im, ax=ax_heat, pad=0.01)
        cb.set_label("GMM γ", fontsize=7)
        cb.ax.tick_params(labelsize=6)

    ax_heat.set_yticks(range(n_sats))
    ax_heat.set_yticklabels([f"SV{i}" for i in range(n_sats)], fontsize=7)
    ax_heat.set_ylabel("Satellite", fontsize=9)
    title = "選択 Subset — Satellite Weights" if sat_weights is not None else "GMM γ per SV"
    ax_heat.set_title(title, fontsize=10, fontweight="bold")

    # Right: n_active / n_excluded bar
    ax_count.step(
        ep, arrays.n_active, where="post", color="#2ca02c", linewidth=1.5, label="n_active"
    )
    ax_count.step(
        ep, arrays.n_excluded, where="post",
        color="#d62728", linewidth=1.5, linestyle="--", label="n_excluded",
    )
    ax_count.set_ylabel("Satellite count", fontsize=9)
    ax_count.set_ylim(0, n_sats + 1)
    ax_count.legend(fontsize=7)
    ax_count.set_title("Active / Excluded Count", fontsize=10, fontweight="bold")
    ax_count.grid(axis="y", linewidth=0.4, alpha=0.5)


def _panel_confidence(
    ax: Axes,
    arrays: EdgeArrays,
    failsafe_levels: list[str] | None,
) -> None:
    """Panel 5: confidence + ins_weight + failsafe bands + mc_auc scatter."""
    ep = arrays.epochs

    if failsafe_levels is not None:
        _add_failsafe_bands(ax, ep, failsafe_levels)

    ax.plot(ep, arrays.confidence, color="#1f77b4", linewidth=1.5, label="confidence")
    ax.plot(ep, arrays.ins_weight, color="#ff7f0e", linewidth=1.5, label="ins_weight")

    # mc_auc scatter (non-NaN epochs only)
    valid = ~np.isnan(arrays.mc_auc)
    if valid.any():
        ax.scatter(
            ep[valid],
            arrays.mc_auc[valid],
            marker="D",
            color="#9467bd",
            zorder=6,
            s=25,
            label="mc_auc",
        )

    # INS_ONLY and DEAD_RECKONING reference lines
    ax.axhline(y=0.90, color="#d62728", linewidth=0.7, linestyle=":", alpha=0.6)
    ax.axhline(y=0.45, color="#ff7f0e", linewidth=0.7, linestyle=":", alpha=0.6)

    ax.set_ylabel("Weight / Confidence", fontsize=9)
    ax.set_ylim(-0.02, 1.08)
    ax.legend(loc="upper right", fontsize=7)
    ax.set_title(
        "推定信頼度 — Confidence, INS Weight & Failsafe", fontsize=10, fontweight="bold"
    )
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plot_gnss_report(
    arrays: EdgeArrays,
    history: list[_EpochRecord] | None = None,
    title: str = "GNSS Resilience Diagnostic",
    save_path: str | Path | None = None,
) -> Figure:
    """Generate a 5-panel GNSS diagnostic figure from EdgeArrays.

    Parameters
    ----------
    arrays:
        EdgeArrays returned by EdgeCollector.to_arrays().
    history:
        Optional MVPPipeline.history list.  Enables satellite_weights heatmap
        and per-epoch failsafe / alert level annotations.  When None, those
        layers are omitted.
    title:
        Figure super-title (suptitle).
    save_path:
        If provided, the figure is saved to this path (PNG at 150 dpi).
        Parent directories are created automatically.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.gridspec as gridspec  # noqa: PLC0415
    import matplotlib.pyplot as plt  # noqa: PLC0415

    # --- Extract optional history-derived arrays ---
    sat_weights: np.ndarray | None = None
    failsafe_levels: list[str] | None = None
    alert_levels: list[str] | None = None

    if history is not None:
        sat_weights = _extract_satellite_weights(history, arrays.n_sats)
        failsafe_levels = _extract_failsafe_levels(history)
        alert_levels = _extract_alert_levels(history)

    # --- Build figure layout ---
    # 5 rows: panels 1-3 full-width, panel 4 split 3:2, panel 5 full-width
    fig = plt.figure(figsize=(14, 20))
    gs = gridspec.GridSpec(
        5, 5,
        figure=fig,
        hspace=0.55,
        wspace=0.40,
        left=0.07, right=0.97, top=0.94, bottom=0.04,
    )

    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, :])
    ax3 = fig.add_subplot(gs[2, :])
    ax4_heat = fig.add_subplot(gs[3, :3])
    ax4_cnt = fig.add_subplot(gs[3, 3:])
    ax5 = fig.add_subplot(gs[4, :])

    ep = arrays.epochs
    for ax in (ax1, ax2, ax3, ax4_heat, ax4_cnt, ax5):
        ax.set_xlim(float(ep[0]) - 0.5, float(ep[-1]) + 0.5)  # type: ignore[union-attr]
        ax.set_xlabel("Epoch", fontsize=8)  # type: ignore[union-attr]
        ax.tick_params(labelsize=8)  # type: ignore[union-attr]

    _panel_attack_evidence(ax1, arrays, alert_levels, failsafe_levels)
    _panel_doppler_residuals(ax2, arrays)
    _panel_entropy(ax3, arrays)
    _panel_subset(ax4_heat, ax4_cnt, arrays, sat_weights)
    _panel_confidence(ax5, arrays, failsafe_levels)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.97)

    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")

    return fig
