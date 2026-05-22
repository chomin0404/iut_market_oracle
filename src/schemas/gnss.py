"""T1300/T1350/T1500 GNSS spoofing detection and Resilience Twin schemas."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# T1300  Monte Carlo GNSS spoofing detection schemas
# ---------------------------------------------------------------------------


class RunTrace(BaseModel):
    """Per-epoch time series for a single MC run (T1300).

    All lists have length n_epochs.
    delay[t] is t − attack_start when the first alarm fires at epoch t;
    None for all other epochs.
    """

    score: list[float] = Field(..., description="Detection score T(t) per epoch [Hz²]")
    alarm: list[bool] = Field(..., description="T(t) > τ at each epoch")
    delay: list[float | None] = Field(
        ..., description="t − attack_start at first alarm epoch; None elsewhere"
    )
    pvt_error: list[float] = Field(..., description="‖r_S(t)‖₂ WLS residual norm [Hz]")


class RunResult(BaseModel):
    """Per-run summary and trace for one MC realisation (T1300).

    score_max:  max T(t) over the run.
    alarm_any:  True iff at least one epoch triggered an alarm.
    delay:      First alarm delay [epochs after attack start]; None if undetected.
    pvt_rmse:   sqrt(mean(‖r_S(t)‖²)) over all epochs [Hz].
    pvt_max:    max(‖r_S(t)‖) over all epochs [Hz].
    trace:      Per-epoch time series.
    """

    score_max: float = Field(..., ge=0.0)
    alarm_any: bool
    delay: float | None = Field(..., description="Epochs from attack start to first alarm")
    pvt_rmse: float = Field(..., ge=0.0)
    pvt_max: float = Field(..., ge=0.0)
    trace: RunTrace


class MCSimReport(BaseModel):
    """Results of the Monte Carlo GNSS spoofing detection simulation (T1300).

    roc_fpr / roc_tpr:
        FPR/TPR pairs at 200 thresholds for ROC curve plotting.
    auc:
        Area under ROC curve (trapezoidal integration).
    mean_detection_delay / std_detection_delay:
        First-alarm epoch relative to attack start [epochs].
        mean is NaN when no run achieved detection.
    mean_pvt_degradation / std_pvt_degradation:
        Ratio ||r_S|| / ||r_all|| during attack epochs.  Values < 1
        indicate subset selection improves PVT accuracy under attack.
    p_detection:
        Empirical detection probability at the Neyman-Pearson threshold.
    p_false_alarm:
        Empirical false-alarm rate at the NP threshold.
    n_mc:
        Number of Monte Carlo runs used.
    """

    roc_fpr: list[float]
    roc_tpr: list[float]
    auc: float = Field(..., ge=0.0, le=1.0)
    mean_detection_delay: float = Field(
        ...,
        description="Mean epochs from attack start to first alarm (NaN if no detection)",
    )
    std_detection_delay: float = Field(..., ge=0.0)
    mean_pvt_degradation: float = Field(
        ...,
        description="Mean ||r_S|| / ||r_all|| during attack epochs",
    )
    std_pvt_degradation: float = Field(..., ge=0.0)
    p_detection: float = Field(..., ge=0.0, le=1.0)
    p_false_alarm: float = Field(..., ge=0.0, le=1.0)
    n_mc: int = Field(..., ge=1)
    produced_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    runs: list[RunResult] = Field(
        default_factory=list,
        description="Per-run summary and trace for each MC realisation",
    )


# ---------------------------------------------------------------------------
# T1350  Multi-sensor GNSS spoofing detection schemas
# ---------------------------------------------------------------------------


class MSRunTrace(BaseModel):
    """Per-epoch time series for a single multi-sensor MC run (T1350).

    All lists have length T (cfg.T epochs).
    score[t]: weighted detection score s(t) = w₁·m + w₂·clip(chi/χ₀) + w₃·clip(lor_dev).
    alarm[t]: s(t) > detect_threshold.
    mix[t]:   spoofing mix fraction α ∈ [0, 1]; 0 for genuine trials.
    m[t]:     largest-connected-component fraction from percolation graph.
    chi[t]:   chi statistic (degree mean + variance / (mean + ε)) / n_sat.
    lor_dev[t]: Lorentz AoA diversity deviation ∈ [0, 1].
    pos_err[t]: position error proxy [m].
    """

    score: list[float] = Field(..., description="Detection score s(t) per epoch")
    alarm: list[bool] = Field(..., description="s(t) > threshold at each epoch")
    mix: list[float] = Field(..., description="Spoofing mix fraction α(t)")
    m: list[float] = Field(..., description="Percolation largest-component fraction m(t)")
    chi: list[float] = Field(..., description="Degree heterogeneity statistic chi(t)")
    lor_dev: list[float] = Field(..., description="Lorentz AoA diversity deviation")
    pos_err: list[float] = Field(..., description="Position error proxy [m]")


class MSRunResult(BaseModel):
    """Per-run summary for one multi-sensor MC realisation (T1350).

    score_max:       max s(t) over the trial.
    alarm_any:       True iff any epoch triggered an alarm.
    delay:           First alarm epoch − attack_start; None if undetected or genuine.
    pvt_rmse:        sqrt(mean(pos_err²)) over all epochs [m].
    pvt_max:         max(pos_err) over all epochs [m].
    hazard_no_alarm: 1 if max pos-error exceeded hazard_pos during attack with no alarm.
    trace:           Per-epoch time series.
    """

    score_max: float = Field(..., ge=0.0)
    alarm_any: bool
    delay: int | None = Field(..., description="Epochs from attack start to first alarm")
    pvt_rmse: float = Field(..., ge=0.0)
    pvt_max: float = Field(..., ge=0.0)
    hazard_no_alarm: int = Field(..., ge=0, le=1)
    trace: MSRunTrace


class MSSimReport(BaseModel):
    """Results of the multi-sensor Monte Carlo GNSS spoofing simulation (T1350).

    p_fa:                  Empirical false-alarm rate at the fixed detect_threshold.
    p_d:                   Empirical detection probability at the fixed detect_threshold.
    p_md:                  Miss-detection probability = 1 − p_d.
    median_delay:          Median first-alarm delay over detected attack runs [epochs];
                           None if no run detected.
    mean_delay:            Mean first-alarm delay [epochs]; None if no run detected.
    auc:                   Area under ROC curve (trapezoidal integration of score_max).
    nominal_rmse_mean:     Mean pvt_rmse over genuine runs [m].
    attack_rmse_mean:      Mean pvt_rmse over attack runs [m].
    attack_pvt_max_mean:   Mean pvt_max over attack runs [m].
    hazard_no_alarm_rate:  Fraction of attack runs with hazard and no alarm.
    roc_fpr / roc_tpr:     FPR/TPR pairs for ROC curve plotting.
    n_nominal / n_attack:  Run counts.
    runs:                  Per-run results (nominal runs first, attack runs second).
    """

    p_fa: float = Field(..., ge=0.0, le=1.0)
    p_d: float = Field(..., ge=0.0, le=1.0)
    p_md: float = Field(..., ge=0.0, le=1.0)
    median_delay: float | None = Field(
        ..., description="Median epochs from attack start to first alarm"
    )
    mean_delay: float | None = Field(
        ..., description="Mean epochs from attack start to first alarm"
    )
    auc: float = Field(..., ge=0.0, le=1.0)
    nominal_rmse_mean: float = Field(..., ge=0.0)
    attack_rmse_mean: float = Field(..., ge=0.0)
    attack_pvt_max_mean: float = Field(..., ge=0.0)
    hazard_no_alarm_rate: float = Field(..., ge=0.0, le=1.0)
    roc_fpr: list[float]
    roc_tpr: list[float]
    n_nominal: int = Field(..., ge=1)
    n_attack: int = Field(..., ge=1)
    produced_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    runs: list[MSRunResult] = Field(
        default_factory=list,
        description="Per-run results: nominal runs first, attack runs second",
    )


# ---------------------------------------------------------------------------
# T1500  GNSS Resilience Twin schemas
# ---------------------------------------------------------------------------


class FaultClass(str, Enum):
    """4-class GNSS fault taxonomy for the Resilience Twin (T1500).

    Index order [0..3] matches the fault_posterior array layout:
        0: NOMINAL        — receiver operating correctly
        1: MULTIPATH      — elevation-correlated range errors, no inter-sat coherence
        2: HARDWARE_FAULT — isolated single-satellite outlier
        3: SPOOFING       — coordinated cross-satellite bias
    """

    NOMINAL = "nominal"
    MULTIPATH = "multipath"
    HARDWARE_FAULT = "hardware_fault"
    SPOOFING = "spoofing"


class ResilienceTwinReport(BaseModel):
    """Results of the GNSS Resilience Twin Monte Carlo simulation (T1500).

    Binary detection (any fault vs nominal):
        p_detection:   P(alarm | any fault class) at the median nominal score threshold.
        p_false_alarm: P(alarm | nominal) at the same threshold.
        auc:           Area under the binary ROC curve (max P_fault score vs label).

    4-class classification:
        per_class_accuracy: accuracy per fault class (FaultClass.value → float).
        confusion_matrix:   4×4 count table [gt_class_index][pred_class_index].

    Summary:
        mean_confidence: mean max(fault_posterior) over all trial epochs.
        n_mc:            Total Monte Carlo trials.
        n_mc_per_class:  Trial counts per fault class (FaultClass.value → int).
    """

    p_detection: float = Field(..., ge=0.0, le=1.0)
    p_false_alarm: float = Field(..., ge=0.0, le=1.0)
    auc: float = Field(..., ge=0.0, le=1.0)
    per_class_accuracy: dict[str, float] = Field(
        ..., description="FaultClass.value → accuracy ∈ [0, 1]"
    )
    confusion_matrix: list[list[int]] = Field(..., description="4×4 [gt][pred] count table")
    mean_confidence: float = Field(..., ge=0.0, le=1.0)
    n_mc: int = Field(..., ge=1)
    n_mc_per_class: dict[str, int] = Field(..., description="FaultClass.value → trial count")
    produced_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def confusion_matrix_4x4(self) -> ResilienceTwinReport:
        if len(self.confusion_matrix) != 4 or any(len(r) != 4 for r in self.confusion_matrix):
            raise ValueError("confusion_matrix must be 4×4")
        return self


# ---------------------------------------------------------------------------
# T1500  Probabilistic Digital Twin — observation-driven inference schemas
# ---------------------------------------------------------------------------


class RecommendedAction(str, Enum):
    """Ordered severity scale for GNSS Resilience Twin operator recommendations.

    Decision boundary (applied in priority order):
        P(spoofing) > 0.50  → GROUND_IMMEDIATELY
        P(hw_fault) > 0.50  → SWITCH_SOURCE
        P(multipath) > 0.50 → REDUCE_TRUST
        P(nominal) < 0.70 or entropy_alert → MONITOR
        otherwise           → NOMINAL
    """

    NOMINAL = "nominal"
    MONITOR = "monitor"
    REDUCE_TRUST = "reduce_trust"
    SWITCH_SOURCE = "switch_source"
    GROUND_IMMEDIATELY = "ground_immediately"


class ObservationEpoch(BaseModel):
    """Single-epoch observation from a GNSS receiver.

    Attributes:
        epoch:             Epoch index (must be monotonically increasing within a run).
        doppler_residuals: Per-satellite Doppler residuals Δf_i = f_meas − f_pred [Hz].
                           Length must equal the n_sats declared in TwinRunRequest.
        elevations_deg:    Per-satellite elevation angles [degrees].
                           If provided, used by GM-RAIM for elevation-adjusted noise σ_i.
                           If omitted, elevations are derived from the LOS geometry.
    """

    epoch: int = Field(..., ge=0, description="Epoch index (monotonically increasing)")
    doppler_residuals: list[float] = Field(
        ..., min_length=5, description="Doppler residuals Δf_i [Hz] — one entry per satellite"
    )
    elevations_deg: list[float] | None = Field(
        default=None,
        description="Satellite elevation angles [degrees]. Derived from LOS if omitted.",
    )
    ins_velocity_ms: list[float] | None = Field(
        default=None,
        description=(
            "INS velocity deviation [m/s] — 3-component vector [Δvx, Δvy, Δvz]. "
            "Used by Layer 5 INS coupling chi² cross-check. Omit for GPS-only receivers."
        ),
    )
    osnma_auth_per_sat: list[bool] | None = Field(
        default=None,
        description=(
            "Per-satellite Galileo OSNMA authentication flags (one bool per satellite). "
            "Used by Layer 7. Omit for non-Galileo constellations (defaults to fully auth)."
        ),
    )


class EpochReport(BaseModel):
    """Full probabilistic diagnosis for one observation epoch.

    Probabilistic assessments:
        authenticity:   {"genuine": P(not spoofed), "spoofed": P(spoofing)}
        integrity:      {"nominal": P(nominal), "degraded": P(any fault)}
        fault_posterior: {FaultClass.value: P} for all 4 fault classes

    Layer signals (for transparency / debugging):
        entropy_alert:          True if H(π) > threshold OR KL > threshold OR |ΔH| > threshold
        gmm_n_fault:            Number of satellites with per-satellite fault posterior > 0.5
        imm_spoof_weight:       μ_spoof — IMM mode weight for the spoofing regime
        spectral_fiedler_ratio: ρ_F = λ₂ / λ₂_null (>1 indicates graph anomaly)
    """

    epoch: int
    authenticity: dict[str, float] = Field(..., description='{"genuine": float, "spoofed": float}')
    integrity: dict[str, float] = Field(..., description='{"nominal": float, "degraded": float}')
    fault_posterior: dict[str, float] = Field(
        ..., description="FaultClass.value → posterior probability ∈ [0, 1]"
    )
    diagnosis: str = Field(..., description="MAP fault class (FaultClass.value)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="max(fault_posterior)")
    recommended_action: RecommendedAction
    action_reason: str = Field(..., description="Human-readable justification for the action")
    entropy_alert: bool
    gmm_n_fault: int = Field(..., ge=0, description="Satellites flagged by GM-RAIM")
    imm_spoof_weight: float = Field(..., ge=0.0, le=1.0, description="μ_spoof from IMM-KF")
    spectral_fiedler_ratio: float = Field(..., description="ρ_F = λ₂ / λ₂_null")
    # Layer 5 — INS coupling
    ins_chi2_vel: float = Field(..., ge=0.0, description="INS coupling chi²(3) statistic")
    ins_alert: bool = Field(..., description="True if INS chi² test triggered")
    # Layer 6 — Cooperative RAIM
    coop_parity_chi2: float = Field(..., ge=0.0, description="Cooperative RAIM parity chi²")
    coop_parity_alert: bool = Field(..., description="True if cooperative RAIM parity test fired")
    # Layer 7 — OSNMA
    osnma_auth_fraction: float = Field(
        ..., ge=0.0, le=1.0, description="Fraction of OSNMA-authenticated satellites"
    )
    # Layer 8 — Structural dependency
    structural_fiedler_streak: int = Field(
        ..., ge=0, description="Consecutive epochs with Fiedler-ratio anomaly"
    )
    structural_alert: bool = Field(..., description="True if structural monitor triggered")
    # Pillar-level summary signals
    auth_p_spoofed: float = Field(
        ..., ge=0.0, le=1.0, description="Authentication pillar P(spoofed) = 1 − auth_fraction"
    )
    integrity_base_fault: float = Field(
        ..., ge=0.0, le=1.0, description="Integrity pillar P(any fault) before structural fusion"
    )
    structure_intensity: float = Field(
        ..., ge=0.0, description="Structure pillar anomaly intensity: max(ρ_F−1,0) + rmt_anomaly"
    )
    # Layer 9 — Huh D-optimal subset selection
    huh_det_ratio: float = Field(
        ..., ge=0.0, description="det(H_sel ᵀ H_sel) / det(H_all ᵀ H_all) — D-optimal improvement"
    )
    huh_n_excluded: int = Field(
        ..., ge=0, description="Satellites excluded by Huh selector (GM-RAIM fault flags)"
    )
    huh_log_concavity_ratio: float = Field(
        ..., description="min σₖ² / (σₖ₋₁ σₖ₊₁) on H_sel singular values (log-concavity proxy)"
    )
    # Layer 10 — Duminil-Copin percolation phase-transition monitor
    phase_percolation_threshold: float = Field(
        ..., ge=0.0, le=1.0, description="τ* where percolation susceptibility χ is maximised"
    )
    phase_susceptibility_peak: float = Field(
        ..., ge=0.0, description="max |ΔLCC/Δτ| over the τ sweep (χ_peak)"
    )
    phase_alert: bool = Field(
        ..., description="True if χ_peak > 10.0 (coordinated synchronised graph collapse)"
    )


class TwinRunReport(BaseModel):
    """Full probabilistic digital twin report for an observation window.

    epoch_reports:             Per-epoch EpochReport list (same length as input observations).
    dominant_diagnosis:        Most frequent MAP diagnosis across all epochs.
    mean_authenticity_genuine: Mean P(genuine) across all epochs.
    mean_integrity_nominal:    Mean P(nominal) across all epochs.
    alert_epochs:              Epoch indices where entropy_alert was raised.
    spoofing_window:           [first, last] epoch indices with P(spoofing) > 0.50,
                               or null if no spoofing detected.
    worst_action:              Highest-severity recommended_action observed in the window.
    """

    epoch_reports: list[EpochReport]
    n_epochs: int = Field(..., ge=1)
    n_sats: int = Field(..., ge=5)
    dominant_diagnosis: str = Field(..., description="Mode of per-epoch MAP diagnoses")
    mean_authenticity_genuine: float = Field(..., ge=0.0, le=1.0)
    mean_integrity_nominal: float = Field(..., ge=0.0, le=1.0)
    alert_epochs: list[int] = Field(..., description="Epochs where entropy_alert was raised")
    spoofing_window: list[int] | None = Field(
        default=None,
        description="[first_epoch, last_epoch] with P(spoofing) > 0.50, or null",
    )
    worst_action: RecommendedAction = Field(
        ..., description="Highest-severity action across all epochs"
    )
    produced_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    run_id: str | None = Field(
        default=None,
        description="Unique run identifier (8-char hex). Set when save=True.",
    )
    result_path: str | None = Field(
        default=None,
        description="Relative path to the saved JSON artifact, e.g. output/<run_id>/twin_run.json",
    )
