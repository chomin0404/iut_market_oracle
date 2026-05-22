"""T800 Digital Twin Engine schemas."""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field, model_validator


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

    experiment_id: str = Field(..., pattern=r"^exp-\d+$", description="e.g. 'exp-001'")
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

    experiment_id: str = Field(..., pattern=r"^exp-\d+$")
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
