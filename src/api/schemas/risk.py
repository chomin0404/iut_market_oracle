"""Request/response schemas for the Monte Carlo risk router."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

MAX_N_SAMPLES = 100_000
DEFAULT_ALPHA = 0.95


class DistributionSpec(BaseModel):
    name: str = Field(
        ...,
        description=(
            "Distribution name. Supported: normal, lognormal, t, uniform, gev, expon, beta, gamma."
        ),
    )
    params: dict[str, float] = Field(
        default_factory=dict,
        description="scipy.stats keyword arguments (e.g. loc, scale, s, df, c).",
    )


class CopulaSpec(BaseModel):
    type: str = Field(
        "gaussian",
        description="Copula type: gaussian, student_t, clayton, independent.",
    )
    corr_matrix: list[list[float]] | None = Field(
        None,
        description="Correlation matrix (n_vars × n_vars). Required for gaussian and student_t.",
    )
    df: float | None = Field(
        None,
        ge=2.0,
        description="Degrees of freedom. Required for student_t copula.",
    )
    theta: float | None = Field(
        None,
        gt=0.0,
        description="Dependence parameter theta > 0. Required for clayton copula.",
    )

    @model_validator(mode="after")
    def _check_copula_fields(self) -> CopulaSpec:
        if self.type in ("gaussian", "student_t"):
            if self.corr_matrix is None:
                raise ValueError(f"corr_matrix is required for copula type {self.type!r}.")
        if self.type == "student_t" and self.df is None:
            raise ValueError("df is required for student_t copula.")
        if self.type == "clayton" and self.theta is None:
            raise ValueError("theta is required for clayton copula.")
        return self


class SimulateRequest(BaseModel):
    n_vars: int = Field(..., ge=1, le=20, description="Number of variables.")
    n_samples: int = Field(..., ge=100, le=MAX_N_SAMPLES, description="Number of MC samples.")
    distributions: list[DistributionSpec] = Field(
        ..., description="Marginal distribution for each variable (length must equal n_vars)."
    )
    copula: CopulaSpec
    seed: int | None = Field(None, description="Random seed for reproducibility.")

    @model_validator(mode="after")
    def _check_dimensions(self) -> SimulateRequest:
        if len(self.distributions) != self.n_vars:
            raise ValueError(
                f"distributions has {len(self.distributions)} entries but n_vars={self.n_vars}"
            )
        if self.copula.corr_matrix is not None:
            rows = len(self.copula.corr_matrix)
            if rows != self.n_vars:
                raise ValueError(f"corr_matrix has {rows} rows but n_vars={self.n_vars}")
        return self


class VariableSummary(BaseModel):
    mean: float
    std: float
    var_95: float = Field(..., description="95th-percentile VaR of variable index 0.")
    es_95: float = Field(..., description="95% Expected Shortfall of variable index 0.")


class SimulateResponse(BaseModel):
    simulation_id: str = Field(..., description="UUID for downstream /risk/boundary calls.")
    n_samples: int
    summary: VariableSummary = Field(..., description="Summary statistics for variable index 0.")


class BoundaryRequest(BaseModel):
    simulation_id: str | None = Field(None, description="ID returned by POST /api/v1/simulate.")
    samples: list[float] | None = Field(
        None, description="Raw samples (alternative to simulation_id)."
    )
    target_variable_index: int = Field(
        0, ge=0, description="Column index in the stored simulation array."
    )
    thresholds: list[float] = Field(..., min_length=1, description="Threshold values.")
    confidence_level: float = Field(
        DEFAULT_ALPHA, ge=0.5, lt=1.0, description="Confidence level for VaR/ES."
    )
    bootstrap_n: int = Field(500, ge=10, le=5000, description="Bootstrap resamples.")
    bootstrap_seed: int | None = Field(None, description="Seed for bootstrap reproducibility.")

    @model_validator(mode="after")
    def _check_source(self) -> BoundaryRequest:
        if self.simulation_id is None and self.samples is None:
            raise ValueError("Provide either simulation_id or samples.")
        return self


class ConfidenceBand(BaseModel):
    lower: list[float]
    upper: list[float]


class BoundaryResponse(BaseModel):
    thresholds: list[float]
    exceedance_probs: list[float]
    confidence_band: ConfidenceBand
    var_95: float
    es_95: float


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------

RiskMetric = Literal["var_95", "es_95", "var_99"]


class SweepParameter(BaseModel):
    target: str = Field(
        ...,
        description=('Target to sweep: "copula" or "distribution_{i}" (e.g. "distribution_0").'),
    )
    param_name: str = Field(
        ...,
        description=(
            'Copula params: "corr_matrix_off_diagonal", "df", "theta". '
            'Distribution params: any scipy.stats kwarg (e.g. "scale", "s").'
        ),
    )
    values: list[float] = Field(..., min_length=1, description="Sweep values.")


class SensitivityRequest(BaseModel):
    base_config: SimulateRequest = Field(..., description="Base simulation configuration.")
    sweep_parameter: SweepParameter
    risk_metric: RiskMetric = Field(
        "var_95", description='Risk metric to compute: "var_95", "es_95", "var_99".'
    )
    target_variable_index: int = Field(
        0, ge=0, description="Variable index for risk metric computation."
    )


class SensitivityResponse(BaseModel):
    parameter_values: list[float]
    risk_values: list[float]
    sensitivity_index: float = Field(
        ...,
        description="Relative range: (max - min) / |max|. Zero when |max| is zero.",
    )
    most_sensitive_at: float = Field(
        ..., description="Parameter value at which the risk metric is highest."
    )


# ---------------------------------------------------------------------------
# Tail analysis
# ---------------------------------------------------------------------------


class TailRequest(BaseModel):
    simulation_id: str | None = Field(None, description="ID returned by POST /risk/simulate.")
    samples: list[float] | None = Field(
        None, description="Raw samples (alternative to simulation_id)."
    )
    target_variable_index: int = Field(
        0, ge=0, description="Column index in the stored simulation array."
    )
    alphas: list[float] = Field(
        ...,
        min_length=1,
        description="Confidence levels, e.g. [0.90, 0.95, 0.99]. Each must be in [0.5, 1).",
    )

    @model_validator(mode="after")
    def _check_source(self) -> TailRequest:
        if self.simulation_id is None and self.samples is None:
            raise ValueError("Provide either simulation_id or samples.")
        for a in self.alphas:
            if not (0.5 <= a < 1.0):
                raise ValueError(f"alpha={a} out of range; each alpha must be in [0.5, 1).")
        return self


class TailStatEntry(BaseModel):
    alpha: float
    var: float = Field(..., description="Value at Risk at the given confidence level.")
    es: float = Field(..., description="Expected Shortfall (CVaR) at the given confidence level.")


class TailResponse(BaseModel):
    tail_stats: list[TailStatEntry]
    n_samples: int


# ---------------------------------------------------------------------------
# GNSS scenario analysis
# ---------------------------------------------------------------------------


class GnssVariableSpec(BaseModel):
    name: str = Field(..., description="Variable name, e.g. 'position_error_m'.")
    distribution: DistributionSpec


class GnssScenarioRequest(BaseModel):
    scenario_name: str = Field(..., description="Human-readable scenario label.")
    variables: list[GnssVariableSpec] = Field(..., min_length=1)
    copula: CopulaSpec
    n_samples: int = Field(10_000, ge=100, le=MAX_N_SAMPLES, description="Number of MC samples.")
    seed: int | None = Field(None, description="Random seed for reproducibility.")
    alphas: list[float] = Field(
        default=[0.90, 0.95, 0.99],
        min_length=1,
        description="Confidence levels for VaR/ES. Each must be in [0.5, 1).",
    )

    @model_validator(mode="after")
    def _check_dimensions(self) -> GnssScenarioRequest:
        n_vars = len(self.variables)
        if self.copula.corr_matrix is not None:
            rows = len(self.copula.corr_matrix)
            if rows != n_vars:
                raise ValueError(f"corr_matrix has {rows} rows but variables has {n_vars} entries.")
        for a in self.alphas:
            if not (0.5 <= a < 1.0):
                raise ValueError(f"alpha={a} out of range; each alpha must be in [0.5, 1).")
        return self


class GnssVariableResult(BaseModel):
    name: str
    mean: float
    std: float
    tail_stats: list[TailStatEntry]


class GnssScenarioResponse(BaseModel):
    scenario_name: str
    n_samples: int
    per_variable: list[GnssVariableResult]
