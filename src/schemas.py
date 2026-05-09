"""Typed schemas shared across all research modules (T200–T1400)."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ClaimTag(str, Enum):
    """Epistemic status of a research claim."""

    PROVEN = "proven"
    HEURISTIC = "heuristic"
    EMPIRICAL = "empirical"
    TODO = "todo"


class EvidenceKind(str, Enum):
    """Kind of evidence submitted to the Bayesian engine."""

    OBSERVATION = "observation"
    EXPERT_PRIOR = "expert_prior"
    MARKET_DATA = "market_data"
    BACKTEST = "backtest"


# ---------------------------------------------------------------------------
# T200  Bayesian Engine schemas
# ---------------------------------------------------------------------------


class Evidence(BaseModel):
    """Single piece of evidence for Bayesian updating."""

    source: str = Field(..., min_length=1, description="Origin of the evidence")
    kind: EvidenceKind
    value: float = Field(..., description="Point estimate or likelihood ratio")
    weight: float = Field(default=1.0, gt=0.0, description="Relative credibility weight")
    tag: ClaimTag = ClaimTag.EMPIRICAL
    notes: str = ""

    @field_validator("value")
    @classmethod
    def finite_value(cls, v: float) -> float:
        import math

        if not math.isfinite(v):
            raise ValueError("value must be finite")
        return v


class PriorSpec(BaseModel):
    """Specification of a prior distribution."""

    distribution: str = Field(..., description="e.g. 'beta', 'normal', 'uniform'")
    params: dict[str, float] = Field(..., description="Distribution parameters")
    description: str = ""

    @field_validator("params")
    @classmethod
    def non_empty_params(cls, v: dict[str, float]) -> dict[str, float]:
        if not v:
            raise ValueError("params must not be empty")
        return v


class PosteriorSummary(BaseModel):
    """Output of one Bayesian update cycle."""

    mean: float
    variance: float = Field(..., ge=0.0)
    credible_interval_95: tuple[float, float]
    n_evidence: int = Field(..., ge=0)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def interval_ordered(self) -> PosteriorSummary:
        lo, hi = self.credible_interval_95
        if lo > hi:
            raise ValueError("credible_interval_95 lower bound must be <= upper bound")
        return self


# ---------------------------------------------------------------------------
# T300  Dependency / Skill Graph schemas
# ---------------------------------------------------------------------------


class NodeMeta(BaseModel):
    """Metadata for a node in the skill/dependency graph."""

    node_id: str = Field(..., min_length=1)
    label: str = ""
    category: str = ""
    weight: float = Field(default=1.0, gt=0.0)
    attributes: dict[str, Any] = Field(default_factory=dict)


class EdgeMeta(BaseModel):
    """Directed edge in the dependency graph."""

    source: str = Field(..., min_length=1)
    target: str = Field(..., min_length=1)
    strength: float = Field(default=1.0, ge=0.0)
    label: str = ""

    @model_validator(mode="after")
    def no_self_loop(self) -> EdgeMeta:
        if self.source == self.target:
            raise ValueError("Self-loops are not allowed (source == target)")
        return self


class GraphInput(BaseModel):
    """Full graph payload for graph analysis module."""

    nodes: list[NodeMeta] = Field(..., min_length=1)
    edges: list[EdgeMeta] = Field(default_factory=list)

    @model_validator(mode="after")
    def edges_reference_existing_nodes(self) -> GraphInput:
        ids = {n.node_id for n in self.nodes}
        for e in self.edges:
            if e.source not in ids:
                raise ValueError(f"Edge source '{e.source}' not in node list")
            if e.target not in ids:
                raise ValueError(f"Edge target '{e.target}' not in node list")
        return self


class PortfolioMetrics(BaseModel):
    """Computed metrics from graph analysis."""

    basis_diversity: float = Field(..., ge=0.0, le=1.0)
    dependency_concentration: float = Field(..., ge=0.0)
    portfolio_score: float = Field(..., ge=0.0, le=1.0)
    node_count: int = Field(..., ge=1)
    edge_count: int = Field(..., ge=0)
    notes: str = ""


# ---------------------------------------------------------------------------
# T400  Valuation / Scenario schemas
# ---------------------------------------------------------------------------


class AssumptionSet(BaseModel):
    """Named, versioned set of modelling assumptions."""

    name: str = Field(..., min_length=1)
    version: str = "1.0"
    params: dict[str, float | str | bool] = Field(..., description="Scenario parameters")
    random_seed: int | None = None
    description: str = ""

    @field_validator("params")
    @classmethod
    def non_empty_params(cls, v: dict) -> dict:
        if not v:
            raise ValueError("params must not be empty")
        return v


class ScenarioResult(BaseModel):
    """Output produced by one valuation scenario run."""

    scenario_name: str = Field(..., min_length=1)
    assumption_version: str
    value: float
    unit: str = ""
    sensitivity: dict[str, float] = Field(
        default_factory=dict,
        description="Partial derivatives w.r.t. each assumption param",
    )
    produced_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    output_path: str | None = None


# ---------------------------------------------------------------------------
# T500  Experiment Registry schemas
# ---------------------------------------------------------------------------


class ExperimentMeta(BaseModel):
    """Metadata envelope for a single reproducible experiment run."""

    experiment_id: str = Field(..., pattern=r"^exp-\d{3}$", description="e.g. 'exp-001'")
    title: str = Field(..., min_length=1)
    config_path: str = Field(..., description="Relative path to config file used")
    result_path: str | None = None
    note_path: str | None = None
    random_seed: int | None = None
    tags: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    summary: str = ""


# ---------------------------------------------------------------------------
# T800  Digital Twin Engine schemas
# ---------------------------------------------------------------------------


class DigitalTwinState(BaseModel):
    """State snapshot of the Digital Twin at a single time step.

    state_vector:
        Latent state x_t ∈ ℝ^d.  Default 3-D layout:
            x[0] log-revenue      (log JPY millions)
            x[1] growth_rate      (annual decimal)
            x[2] log-volatility   (log annual decimal)
    state_labels:
        Human-readable name for each dimension; length must equal state_vector.
    param_snapshot:
        Calibrated model parameters at this step (μ, σ, etc.).
    step:
        Non-negative integer time step index.
    """

    experiment_id: str = Field(..., pattern=r"^exp-\d{3}$", description="e.g. 'exp-001'")
    state_vector: list[float] = Field(..., min_length=1)
    state_labels: list[str] = Field(..., min_length=1)
    param_snapshot: dict[str, float] = Field(default_factory=dict)
    step: int = Field(default=0, ge=0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def labels_match_vector(self) -> DigitalTwinState:
        if len(self.state_labels) != len(self.state_vector):
            raise ValueError(
                f"state_labels length ({len(self.state_labels)}) must match "
                f"state_vector length ({len(self.state_vector)})"
            )
        return self


class SimulationResult(BaseModel):
    """Output of one Monte Carlo forward simulation (T800).

    trajectories:
        Nested list of shape (n_samples, horizon+1, state_dim).
        trajectories[i][t] is the state vector at step t for sample i.
    """

    experiment_id: str = Field(..., pattern=r"^exp-\d{3}$")
    trajectories: list[list[list[float]]] = Field(..., min_length=1)
    n_samples: int = Field(..., ge=1)
    horizon: int = Field(..., ge=1)
    state_labels: list[str] = Field(..., min_length=1)
    config_path: str | None = None
    produced_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def dimensions_consistent(self) -> SimulationResult:
        if len(self.trajectories) != self.n_samples:
            raise ValueError(
                f"trajectories count ({len(self.trajectories)}) must equal "
                f"n_samples ({self.n_samples})"
            )
        expected_steps = self.horizon + 1  # initial state + horizon forward steps
        for i, traj in enumerate(self.trajectories):
            if len(traj) != expected_steps:
                raise ValueError(
                    f"trajectory[{i}] has {len(traj)} steps; "
                    f"expected {expected_steps} (horizon + 1)"
                )
        return self


# ---------------------------------------------------------------------------
# T900  Exit Strategy Engine schemas
# ---------------------------------------------------------------------------


class ExitType(str, Enum):
    """Exit route classification."""

    IPO = "ipo"
    MA = "ma"
    SECONDARY = "secondary"
    WIND_DOWN = "wind_down"


class ExitOption(BaseModel):
    """Specification of one exit route with timing and value estimates.

    Timing model: triangular distribution over [timing_earliest, timing_latest]
    with mode at timing_expected (all in years from now).

    value_by_scenario:
        Mapping of scenario name → enterprise value in the same unit as
        floor_value (e.g. JPY millions).  At least one scenario required.

    floor_value:
        Minimum net deal value (analogous to a put strike).
        Payoff per scenario = max(V_s - floor_value, 0).

    discount_rate:
        Annual WACC used to discount future payoffs to present value.
    """

    name: str = Field(..., min_length=1)
    exit_type: ExitType
    timing_earliest: float = Field(..., ge=0.0, description="Years from now")
    timing_expected: float = Field(..., ge=0.0, description="Years from now (mode)")
    timing_latest: float = Field(..., ge=0.0, description="Years from now")
    value_by_scenario: dict[str, float] = Field(..., description="scenario name → enterprise value")
    floor_value: float = Field(default=0.0, ge=0.0, description="Minimum deal value")
    discount_rate: float = Field(..., gt=0.0, description="Annual WACC")

    @model_validator(mode="after")
    def timing_ordered(self) -> ExitOption:
        if not (self.timing_earliest <= self.timing_expected <= self.timing_latest):
            raise ValueError(
                "timing must satisfy earliest <= expected <= latest, got "
                f"({self.timing_earliest}, {self.timing_expected}, {self.timing_latest})"
            )
        return self

    @field_validator("value_by_scenario")
    @classmethod
    def non_empty_scenarios(cls, v: dict[str, float]) -> dict[str, float]:
        if not v:
            raise ValueError("value_by_scenario must contain at least one scenario")
        return v


class ExitValueSummary(BaseModel):
    """Pricing output for one exit option (T900).

    scenario_payoffs:
        max(V_s - floor, 0) per scenario (before discounting).
    scenario_pvs:
        Present value of each scenario payoff, discounted at timing_expected.
    expected_value:
        Probability-weighted mean of scenario_pvs.
    sensitivity:
        Central-difference ∂EV/∂p for discount_rate, timing_expected, floor_value.
    """

    option_name: str = Field(..., min_length=1)
    exit_type: ExitType
    scenario_payoffs: dict[str, float]
    scenario_pvs: dict[str, float]
    expected_value: float
    sensitivity: dict[str, float] = Field(default_factory=dict)
    produced_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TimingDistribution(BaseModel):
    """Discretised exit-timing probability distribution (T900).

    Derived from a triangular distribution over [earliest, latest] with
    mode at expected.  Probabilities sum to 1.0 up to floating-point error.

    time_steps:
        Discrete time points in years from now.
    probabilities:
        P(exit at t_k) for each step; normalised to sum to 1.0.
    expected_timing:
        Probability-weighted mean timing E[T] = Σ_k t_k · P(t_k).
    """

    option_name: str = Field(..., min_length=1)
    time_steps: list[float] = Field(..., min_length=1)
    probabilities: list[float] = Field(..., min_length=1)
    expected_timing: float = Field(..., ge=0.0)

    @model_validator(mode="after")
    def steps_and_probs_aligned(self) -> TimingDistribution:
        if len(self.time_steps) != len(self.probabilities):
            raise ValueError(
                f"time_steps length ({len(self.time_steps)}) must equal "
                f"probabilities length ({len(self.probabilities)})"
            )
        total = sum(self.probabilities)
        if not (0.999 <= total <= 1.001):
            raise ValueError(f"probabilities must sum to ~1.0, got {total:.6f}")
        return self


# ---------------------------------------------------------------------------
# T1100  Regime Switching / Market Evolution schemas
# ---------------------------------------------------------------------------


class RegimeSwitchResult(BaseModel):
    """Output of a 2-state Markov regime-switching price simulation (T1100).

    prices:
        Simulated asset price series of length n_steps.
        prices[0] is the initial price; prices[t] = prices[t-1] * (1 + ret_t).
    regimes:
        Regime label at each step: 0 = normal (Laplace returns),
        1 = volatile (Cauchy returns, clipped).
    """

    n_steps: int = Field(..., ge=1)
    prices: list[float] = Field(..., min_length=1)
    regimes: list[int] = Field(..., min_length=1)
    p_stay_normal: float = Field(..., gt=0.0, lt=1.0, description="P(regime=0 | prev=0)")
    p_stay_volatile: float = Field(..., gt=0.0, lt=1.0, description="P(regime=1 | prev=1)")
    produced_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def lengths_consistent(self) -> RegimeSwitchResult:
        if len(self.prices) != self.n_steps:
            raise ValueError(
                f"prices length ({len(self.prices)}) must equal n_steps ({self.n_steps})"
            )
        if len(self.regimes) != self.n_steps:
            raise ValueError(
                f"regimes length ({len(self.regimes)}) must equal n_steps ({self.n_steps})"
            )
        return self


class MarketEvolutionResult(BaseModel):
    """Output of a Gamma-Poisson market evolution simulation (T1100).

    Customer arrivals per step follow a Negative Binomial (Gamma-Poisson) mixture:
        lambda_t ~ Gamma(alpha, scale=1/beta)
        k_t | lambda_t ~ Poisson(lambda_t)

    Market capture is modulated by a logistic sigmoid adoption curve.

    new_customers:
        Customer arrivals k_t at each step.
    cumulative_base:
        Cumulative sum of new_customers up to step t.
    sigmoid_factor:
        Logistic sigmoid values sigma(t) mapped over [-5, 5].
    market_capture:
        cumulative_base[t] * sigmoid_factor[t] — market capture index.
    """

    n_steps: int = Field(..., ge=1)
    new_customers: list[int] = Field(..., min_length=1)
    cumulative_base: list[float] = Field(..., min_length=1)
    sigmoid_factor: list[float] = Field(..., min_length=1)
    market_capture: list[float] = Field(..., min_length=1)
    gamma_alpha: float = Field(..., gt=0.0, description="Gamma shape parameter alpha")
    gamma_beta: float = Field(..., gt=0.0, description="Gamma rate parameter beta (scale=1/beta)")
    produced_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def lengths_consistent(self) -> MarketEvolutionResult:
        for name, lst in [
            ("new_customers", self.new_customers),
            ("cumulative_base", self.cumulative_base),
            ("sigmoid_factor", self.sigmoid_factor),
            ("market_capture", self.market_capture),
        ]:
            if len(lst) != self.n_steps:
                raise ValueError(f"{name} length ({len(lst)}) must equal n_steps ({self.n_steps})")
        return self


# ---------------------------------------------------------------------------
# T1200  Matroid Log-Concavity schemas
# ---------------------------------------------------------------------------


class MatroidLogConcavityResult(BaseModel):
    """Output of a matroid characteristic-polynomial log-concavity computation (T1200).

    Models the coefficients of a rank-generating polynomial:
        b_k = C(n, k) * rank_weight^k * corank_weight^(n-k),  k = 0…n
    After normalisation these form a probability mass function over subset sizes k.

    June Huh (2022 Fields Medal) proved that the characteristic polynomial of any
    matroid has log-concave coefficients.  This schema captures that property.

    log_concavity_checks:
        Boolean per interior index k=1…n-1: True iff b_k² >= b_{k-1} * b_{k+1}.
        Length = n_assets - 1.
    is_log_concave:
        True iff all entries in log_concavity_checks are True.
    """

    n_assets: int = Field(..., ge=1)
    rank_weight: float = Field(..., gt=0.0, description="Weight per rank unit (alpha)")
    corank_weight: float = Field(..., gt=0.0, description="Weight per corank unit (beta)")
    subset_sizes: list[int] = Field(..., min_length=1, description="k = 0, 1, …, n_assets")
    probability_mass: list[float] = Field(..., min_length=1, description="Normalised b_k")
    log_probability: list[float] = Field(..., min_length=1, description="log(b_k + eps)")
    log_concavity_checks: list[bool] = Field(..., description="b_k² >= b_{k-1}*b_{k+1} for k=1…n-1")
    is_log_concave: bool
    produced_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def lengths_consistent(self) -> MatroidLogConcavityResult:
        expected = self.n_assets + 1
        for name, lst in [
            ("subset_sizes", self.subset_sizes),
            ("probability_mass", self.probability_mass),
            ("log_probability", self.log_probability),
        ]:
            if len(lst) != expected:
                raise ValueError(f"{name} length ({len(lst)}) must equal n_assets + 1 ({expected})")
        if len(self.log_concavity_checks) != self.n_assets - 1:
            raise ValueError(
                f"log_concavity_checks length ({len(self.log_concavity_checks)}) "
                f"must equal n_assets - 1 ({self.n_assets - 1})"
            )
        return self


# ---------------------------------------------------------------------------
# T1000  Entropy Layer schemas
# ---------------------------------------------------------------------------


class AlertType(str, Enum):
    """Kind of entropy alert generated by the detector."""

    KL_THRESHOLD = "kl_threshold"
    ENTROPY_GRADIENT = "entropy_gradient"


class EntropyAlert(BaseModel):
    """Single alert event emitted by the entropy detector.

    triggered_at:
        Step index at which the alert condition was detected.
    alert_type:
        KL_THRESHOLD if KL divergence crossed the configured threshold;
        ENTROPY_GRADIENT if the rolling entropy rate gradient triggered.
    metric_value:
        The KL divergence or gradient value that caused the alert.
    threshold:
        The configured threshold that was exceeded.
    """

    experiment_id: str = Field(..., pattern=r"^exp-\d{3}$")
    triggered_at: int = Field(..., ge=0)
    alert_type: AlertType
    metric_value: float
    threshold: float
    message: str = ""
    produced_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class EntropyReport(BaseModel):
    """Aggregated entropy monitoring output for one experiment run (T1000).

    entropy_series:
        Shannon entropy H_t at each observation step.
    kl_series:
        KL divergence KL(posterior_t || prior) at each step.
    entropy_rate_series:
        Rolling mean of ΔH_t = H_t − H_{t−1}.  Length is
        len(entropy_series) − 1 (first differences).
    alerts:
        All alerts fired during monitoring.
    """

    experiment_id: str = Field(..., pattern=r"^exp-\d{3}$")
    entropy_series: list[float] = Field(..., min_length=1)
    kl_series: list[float] = Field(..., min_length=1)
    entropy_rate_series: list[float] = Field(default_factory=list)
    alerts: list[EntropyAlert] = Field(default_factory=list)
    produced_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def series_lengths_consistent(self) -> EntropyReport:
        if len(self.kl_series) != len(self.entropy_series):
            raise ValueError(
                f"kl_series length ({len(self.kl_series)}) must equal "
                f"entropy_series length ({len(self.entropy_series)})"
            )
        return self


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
# T1400  Mathematical Model schemas
# ---------------------------------------------------------------------------


class ModelSpec(BaseModel):
    """Formal specification of a mathematical model."""

    problem_type: str = Field(..., min_length=1, description="Class of mathematical problem")
    objective: str = Field(..., min_length=1, description="Optimisation or inference objective")
    state_variables: list[str] = Field(..., description="Latent/state variables and their domains")
    observables: list[str] = Field(..., description="Observed/measured quantities")
    parameters: list[str] = Field(..., description="Model parameters to be calibrated")
    constraints: list[str] = Field(default_factory=list, description="Mathematical constraints")
    uncertainty: dict[str, str] = Field(
        default_factory=dict, description="Noise/uncertainty specifications"
    )
    equations: list[str] = Field(..., description="Key equations in mathematical notation")
    priors: dict[str, str] = Field(
        default_factory=dict, description="Prior distributions over parameters"
    )
    loss_function: str | None = Field(None, description="Loss or energy function if applicable")
    solver: str = Field(..., min_length=1, description="Recommended solver or inference algorithm")
    outputs: list[str] = Field(..., description="Model outputs and their interpretations")
    assumptions: list[str] = Field(default_factory=list, description="Modelling assumptions")
    evidence_needed: list[str] = Field(
        default_factory=list, description="Data or domain knowledge required"
    )


class ModelRegistryEntry(ModelSpec):
    """ModelSpec augmented with registry metadata."""

    id: str = Field(..., min_length=1, description="snake_case unique identifier")
    name: str = Field(..., min_length=1, description="Human-readable model name")
    category: str = Field(default="", description="Top-level domain category")
    tags: list[str] = Field(default_factory=list, description="Searchable keyword tags")
    references: list[str] = Field(default_factory=list, description="Key citations")


class ModelRecommendation(BaseModel):
    """LLM-generated model recommendation for a given problem description."""

    problem_type: str = Field(..., min_length=1, description="Inferred class of the problem")
    recommended_models: list[str] = Field(
        ..., min_length=1, description="Ordered list of recommended model identifiers"
    )
    rationale: list[str] = Field(
        ..., min_length=1, description="Reasons each model class is appropriate"
    )


# ---------------------------------------------------------------------------
# Formalize-Idea schemas
# ---------------------------------------------------------------------------


class IdeaInput(BaseModel):
    title: str = Field(..., min_length=3, max_length=200, description="Short problem title.")
    description: str = Field(..., min_length=10, description="Natural-language problem statement.")
    domain: str | None = Field(default=None, description="Optional domain label.")
    goal_type: Literal[
        "prediction",
        "optimization",
        "control",
        "anomaly_detection",
        "simulation",
        "causal_inference",
        "decision_support",
    ] = Field(..., description="Primary modeling goal.")
    time_horizon: Literal["static", "sequential", "continuous"] = Field(
        default="static",
        description="Temporal structure of the problem.",
    )
    data_regime: Literal["small", "medium", "large", "unknown"] = Field(
        default="unknown",
        description="Approximate amount of available data.",
    )
    uncertainty_level: Literal["low", "medium", "high", "unknown"] = Field(
        default="unknown",
        description="Expected uncertainty level.",
    )
    physical_constraints: bool = Field(
        default=False,
        description="Whether the system must respect physical laws or domain constraints.",
    )
    decision_variables_present: bool = Field(
        default=False,
        description="Whether optimization or control variables are present.",
    )
    latent_state_present: bool = Field(
        default=False,
        description="Whether hidden system states are believed to exist.",
    )


class ProblemStructure(BaseModel):
    """Typed boolean flags characterising the mathematical structure of a problem."""

    is_sequential: bool = Field(..., description="Observations arrive over time / system evolves.")
    has_latent_state: bool = Field(..., description="Hidden/unobserved state variables exist.")
    has_decision_variables: bool = Field(
        ..., description="Problem involves optimisation or control actions."
    )
    has_physical_constraints: bool = Field(
        ..., description="Physical laws or hard domain constraints apply."
    )
    is_high_uncertainty: bool = Field(..., description="uncertainty_level == 'high'.")
    is_data_scarce: bool = Field(..., description="data_regime == 'small'.")


class ParsedIdeaResponse(BaseModel):
    problem_structure: ProblemStructure
    candidate_families: list[str]
    missing_information: list[str]


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
    authenticity: dict[str, float] = Field(
        ..., description='{"genuine": float, "spoofed": float}'
    )
    integrity: dict[str, float] = Field(
        ..., description='{"nominal": float, "degraded": float}'
    )
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


# ---------------------------------------------------------------------------
# T1600  Process Yield Twin schemas
# ---------------------------------------------------------------------------


class FactorSpec(BaseModel):
    """Specification of one process factor (input variable).

    Values are assumed to lie in [low, high] in physical units.
    The twin normalises to [0, 1] internally for all GP computations.
    """

    name: str = Field(..., min_length=1)
    low: float = Field(..., description="Lower bound of factor range (physical units)")
    high: float = Field(..., description="Upper bound of factor range (physical units)")

    @model_validator(mode="after")
    def low_lt_high(self) -> FactorSpec:
        if not self.low < self.high:
            raise ValueError(f"low ({self.low}) must be strictly less than high ({self.high})")
        return self


class ExperimentPoint(BaseModel):
    """Single design point with optional observed yield.

    factors:   Mapping of factor name → value in physical units.
    yield_obs: Observed yield ∈ [0, 1] (e.g. fraction of conforming parts),
               or None if the experiment has not yet been run.
    """

    factors: dict[str, float]
    yield_obs: float | None = Field(None, ge=0.0, le=1.0)


class DOERecommendation(BaseModel):
    """Next experiment recommended by the Process Yield Twin.

    factors:            Recommended factor settings in physical units.
    expected_improvement: EI(x*) from the GP surrogate [yield units].
    d_leverage:         φ(x*)ᵀ M⁻¹ φ(x*) — D-optimal leverage score.
    fusion_score:       Weighted combination of normalised EI and d_leverage.
    predicted_yield:    GP posterior mean at x* (in [0, 1]).
    predicted_std:      GP posterior standard deviation at x*.
    acquisition_mode:   "doe_explore" | "fused" | "ei_exploit"
    n_observations:     Number of observations available when recommendation was made.
    """

    factors: dict[str, float]
    expected_improvement: float = Field(..., ge=0.0)
    d_leverage: float = Field(..., ge=0.0)
    fusion_score: float = Field(..., ge=0.0)
    predicted_yield: float = Field(..., ge=0.0, le=1.0)
    predicted_std: float = Field(..., ge=0.0)
    acquisition_mode: Literal["doe_explore", "fused", "ei_exploit"]
    n_observations: int = Field(..., ge=0)


class YieldTwinReport(BaseModel):
    """Full optimisation report from the Process Yield Twin (T1600).

    n_observations:      Number of completed experiments.
    best_yield_observed: Maximum observed yield so far (None if no data).
    best_factors:        Factor settings that achieved best_yield_observed.
    surrogate_loocv_r2:  GP leave-one-out cross-validated R² (None if < 3 obs).
    recommendation:      Next experiment to run.
    gp_hyperparams:      Fitted GP hyperparameters:
                           signal_var, noise_var, length_scale_<name> per factor.
    factor_specs:        Factor definitions used for this run.
    """

    n_observations: int = Field(..., ge=0)
    best_yield_observed: float | None = Field(None, ge=0.0, le=1.0)
    best_factors: dict[str, float] | None = None
    surrogate_loocv_r2: float | None = Field(
        None, description="GP LOO cross-validated R² (None when < 3 observations)"
    )
    recommendation: DOERecommendation
    gp_hyperparams: dict[str, float] = Field(
        default_factory=dict,
        description="signal_var, noise_var, length_scale_<factor_name>",
    )
    factor_specs: list[FactorSpec]
    produced_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# T1700  Strategy Twin schemas
# ---------------------------------------------------------------------------


class MoatDimension(str, Enum):
    """Five canonical sources of competitive moat (Morningstar framework)."""

    COST_ADVANTAGE = "cost_advantage"
    SWITCHING_COSTS = "switching_costs"
    NETWORK_EFFECTS = "network_effects"
    INTANGIBLE_ASSETS = "intangible_assets"
    EFFICIENT_SCALE = "efficient_scale"


class MoatScore(BaseModel):
    """Strength assessment for one moat dimension.

    score: 0 = no moat, 0.5 = narrow moat, 1.0 = wide moat.
    weight: relative importance of this dimension (default 1.0).
    """

    dimension: MoatDimension
    score: float = Field(..., ge=0.0, le=1.0, description="Moat strength [0=none, 1=wide]")
    weight: float = Field(default=1.0, gt=0.0, description="Relative importance weight")


class BusinessUnit(BaseModel):
    """Single business segment for SOTP valuation.

    initial_fcf:           FCF₀ in consistent units (e.g. JPY millions).
    growth_rate:           CAGR estimate (decimal; may be negative).
    discount_rate:         Base WACC before moat adjustment (decimal).
    terminal_growth_rate:  Gordon long-run growth (decimal; must be < discount_rate).
    forecast_years:        Explicit DCF horizon.
    moat_scores:           Moat dimension assessments (empty = no adjustment).
    """

    name: str = Field(..., min_length=1)
    initial_fcf: float = Field(..., gt=0.0, description="Initial free cash flow (FCF₀)")
    growth_rate: float = Field(..., description="Revenue/FCF CAGR estimate (decimal)")
    discount_rate: float = Field(..., gt=0.0, description="Base WACC (decimal)")
    terminal_growth_rate: float = Field(
        default=0.03, description="Long-run Gordon growth rate (decimal)"
    )
    forecast_years: int = Field(default=5, ge=1)
    moat_scores: list[MoatScore] = Field(default_factory=list)


class MacroEnvironment(BaseModel):
    """Macro-economic context for Strategy Twin.

    tam: Total Addressable Market in the same unit as BusinessUnit.initial_fcf.
    """

    gdp_growth: float = Field(..., description="Real GDP growth rate (decimal, e.g. 0.02)")
    risk_free_rate: float = Field(..., ge=0.0, description="Risk-free rate (decimal)")
    market_risk_premium: float = Field(..., gt=0.0, description="Equity risk premium (decimal)")
    inflation: float = Field(..., ge=0.0, description="CPI inflation rate (decimal)")
    tam: float = Field(..., gt=0.0, description="Total Addressable Market (same unit as FCF)")


class BLView(BaseModel):
    """Single investor view for Black-Litterman (He-Litterman 1999).

    assets:          asset_name → pick weight
                     Absolute view: {A: 1.0}
                     Relative view: {A: 1.0, B: -1.0}  (long A, short B)
    expected_return: q_k — view's expected excess return (decimal).
    uncertainty:     Ω_kk — view variance (larger = less confident).
    """

    assets: dict[str, float] = Field(..., min_length=1)
    expected_return: float = Field(..., description="View's expected excess return (decimal)")
    uncertainty: float = Field(..., gt=0.0, description="View variance Ω_kk")


class CausalEdge(BaseModel):
    """Directed edge in a linear Structural Causal Model.

    coefficient: b_{effect ← cause}  — linear regression coefficient.
    """

    cause: str = Field(..., min_length=1)
    effect: str = Field(..., min_length=1)
    coefficient: float = Field(..., description="b_{effect ← cause}")

    @model_validator(mode="after")
    def no_self_loop(self) -> CausalEdge:
        if self.cause == self.effect:
            raise ValueError("Self-loops are not allowed (cause == effect)")
        return self


class ViabilityCondition(BaseModel):
    """One measurable condition the business must satisfy to reach its target EV.

    gap = current_estimate − minimum_required.
    is_met = True iff gap >= 0.
    """

    metric: str = Field(..., min_length=1, description="e.g. 'revenue_growth_rate'")
    minimum_required: float = Field(..., description="Minimum value for this condition")
    current_estimate: float = Field(..., description="Current estimated value")
    gap: float = Field(..., description="current_estimate − minimum_required (≥0 means met)")
    is_met: bool
    explanation: str = ""


class SOTPSegment(BaseModel):
    """SOTP valuation result for one business unit."""

    unit_name: str = Field(..., min_length=1)
    enterprise_value: float
    moat_adjusted_wacc: float = Field(..., description="WACC after moat reduction")
    moat_adjusted_terminal_growth: float = Field(..., description="g_T after moat uplift")
    moat_composite_score: float = Field(..., ge=0.0, le=1.0)


class BLResult(BaseModel):
    """Black-Litterman posterior return estimates per asset."""

    equilibrium_returns: dict[str, float] = Field(
        ..., description="Π = δ·Σ·w_mkt (CAPM equilibrium)"
    )
    posterior_returns: dict[str, float] = Field(
        ..., description="μ_BL: posterior mean after blending views"
    )
    posterior_std: dict[str, float] = Field(
        ..., description="sqrt(diag(M⁻¹)): posterior standard deviation"
    )
    market_weights: dict[str, float] = Field(..., description="Normalised FCF market weights")


class CausalEffect(BaseModel):
    """Total linear causal effect via all directed paths (ATE).

    total_effect = Σ_{paths p: cause→effect} Π_{(a→b)∈p} b_{b←a}
    """

    cause: str
    effect: str
    total_effect: float = Field(..., description="ATE = sum of path products")
    n_paths: int = Field(..., ge=0, description="Number of directed paths found")


class StrategyTwinReport(BaseModel):
    """Full output of one Strategy Twin run (T1700).

    sotp_segments:        Per-unit SOTP breakdown with moat adjustments.
    sotp_total_ev:        Sum(segment EVs) − net_debt.
    bl_result:            Black-Litterman posterior returns per segment.
    viability_conditions: Minimum conditions to reach the target EV.
    implied_growth_rate:  g* from reverse DCF (NaN serialised as null if infeasible).
    causal_effects:       All pairwise ATE in the causal graph (sorted by |ATE| desc).
    macro:                Macro snapshot used for this run.
    verdict:              One-line executive decision summary (Japanese).
    """

    sotp_segments: list[SOTPSegment]
    sotp_total_ev: float
    bl_result: BLResult
    viability_conditions: list[ViabilityCondition]
    implied_growth_rate: float | None = Field(
        None, description="g* from reverse DCF; None if infeasible"
    )
    causal_effects: list[CausalEffect]
    macro: MacroEnvironment
    verdict: str
    produced_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# ModelForge  Traceability + Verification schemas
# ---------------------------------------------------------------------------


class VerificationStatus(str, Enum):
    """Result level for one verification check or an overall report."""

    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


class VerificationCheck(BaseModel):
    """Single atomic check in a ModelForge verification run."""

    name: str = Field(..., min_length=1, description="Machine-readable check identifier")
    status: VerificationStatus
    message: str = Field(default="", description="Human-readable detail or empty on PASS")


class VerificationReport(BaseModel):
    """Static verification result for one model registry entry.

    overall: worst status across all checks (FAIL > WARN > PASS).
    registry_hash: SHA-256 of the YAML file content at verification time.
    """

    model_id: str = Field(..., min_length=1)
    registry_hash: str = Field(..., description="SHA-256 hex digest of the YAML content")
    checks: list[VerificationCheck]
    overall: VerificationStatus
    produced_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TraceNodeType(str, Enum):
    """Artifact type in the ModelForge traceability graph."""

    REGISTRY = "registry"          # configs/model_registry/<id>.yaml
    VERIFICATION = "verification"  # artifacts/modelforge/<id>/verification.json
    GENERATED_CODE = "generated_code"  # artifacts/modelforge/<id>/impl_skeleton.py
    AUDIT_ENTRY = "audit_entry"    # .claude/audit/modelforge.jsonl line


class TraceNode(BaseModel):
    """Node in the ModelForge directed acyclic traceability graph.

    node_id: unique stable identifier (SHA-256 prefix of content_hash + type).
    parent_ids: edges — this node was produced from these parent nodes.
    content_hash: SHA-256 of the artifact file at creation time.
    """

    node_id: str = Field(..., min_length=1)
    node_type: TraceNodeType
    model_id: str = Field(..., min_length=1)
    artifact_path: str = Field(..., description="Relative path from project root")
    content_hash: str = Field(..., description="SHA-256 hex digest of artifact content")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    parent_ids: list[str] = Field(default_factory=list)


class ForgeReport(BaseModel):
    """Full output of one ModelForge pipeline run for a single model.

    Connects YAML spec → verification → skeleton code → trace nodes → audit.
    """

    model_id: str = Field(..., min_length=1)
    registry_yaml_path: str
    verification: VerificationReport
    skeleton_code_path: str = Field(..., description="Path to generated impl_skeleton.py")
    trace_node_ids: list[str] = Field(..., description="Ordered node IDs created in this run")
    produced_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
