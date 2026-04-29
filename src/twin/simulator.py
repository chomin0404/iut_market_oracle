"""Monte Carlo forward simulation for the Digital Twin Engine (T800).

State Space Model
-----------------
State transition (linear Gaussian):

    x_{t+1} = F x_t + w_t,    w_t ~ N(0, Q)

where:
    x_t  : state vector in ℝ^d at step t
    F    : (d × d) state transition matrix
    Q    : (d × d) process noise covariance; Q = σ²·I_d  (isotropic)

Default state interpretation (d = 3):
    x[0] : log-revenue       [log JPY millions]
    x[1] : annual growth rate [decimal]
    x[2] : log-volatility    [log decimal]

Default transition matrix (local linear trend for log-revenue):

         ┌ 1  dt  0 ┐
    F =  │ 0   1  0 │
         └ 0   0  1 ┘

    x[0] accumulates growth_rate over dt years;
    x[1] and x[2] follow independent random walks.

Observation model (reserved for future Kalman filter extension):

    y_t = H x_t + v_t,    v_t ~ N(0, R)

Note: floating-point accumulation in matrix products over long horizons
is O(horizon · ε_mach).  For horizon > 100, precision may degrade visibly.
"""

from __future__ import annotations

import numpy as np

from schemas import DigitalTwinState, SimulationResult

# Default time step: one quarter (years)
DEFAULT_DT: float = 0.25


def _default_transition_matrix(state_dim: int, dt: float) -> np.ndarray:
    """Build the local linear trend transition matrix F.

    For state_dim == 3:
        F[0, 1] = dt  →  log_revenue += growth_rate × dt
        All other diagonal entries = 1 (random walks).

    For state_dim != 3: returns the identity matrix I_d.
    """
    F = np.eye(state_dim, dtype=float)
    if state_dim >= 2:
        F[0, 1] = dt
    return F


def simulate(
    initial_state: DigitalTwinState,
    horizon: int,
    n_samples: int,
    process_noise_std: float,
    rng: np.random.Generator,
    dt: float = DEFAULT_DT,
    transition_matrix: np.ndarray | None = None,
) -> SimulationResult:
    """Draw N independent Monte Carlo trajectories from the state space model.

    Parameters
    ----------
    initial_state:
        Starting state snapshot.  All samples begin from the same x_0.
    horizon:
        Number of forward time steps to simulate (>= 1).
    n_samples:
        Number of independent Monte Carlo trajectories (>= 1).
    process_noise_std:
        Standard deviation σ of the isotropic process noise; Q = σ²·I_d.
        Set to 0.0 for a deterministic (noise-free) forward simulation.
    rng:
        NumPy random generator.  Caller must supply a seeded instance to
        ensure reproducibility, e.g. ``np.random.default_rng(42)``.
    dt:
        Time step in years (default 0.25 = one quarter).
    transition_matrix:
        Optional (d × d) ndarray F.  If None, uses _default_transition_matrix.

    Returns
    -------
    SimulationResult
        trajectories has shape (n_samples, horizon+1, state_dim):
        index 0 is the initial state; indices 1…horizon are forward steps.

    Raises
    ------
    ValueError
        If horizon < 1, n_samples < 1, process_noise_std < 0, or
        transition_matrix has incompatible shape.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    if process_noise_std < 0.0:
        raise ValueError(f"process_noise_std must be >= 0, got {process_noise_std}")

    d = len(initial_state.state_vector)
    x0 = np.array(initial_state.state_vector, dtype=float)

    F = transition_matrix if transition_matrix is not None else _default_transition_matrix(d, dt)
    if F.shape != (d, d):
        raise ValueError(f"transition_matrix must be ({d}, {d}), got {F.shape}")

    # Allocate trajectories: (n_samples, horizon+1, d)
    traj = np.empty((n_samples, horizon + 1, d), dtype=float)
    traj[:, 0, :] = x0  # all samples share the same initial state

    for t in range(horizon):
        noise = rng.normal(loc=0.0, scale=process_noise_std, size=(n_samples, d))
        # Vectorised: x_{t+1} = F x_t + w_t  for all samples simultaneously
        traj[:, t + 1, :] = traj[:, t, :] @ F.T + noise

    return SimulationResult(
        experiment_id=initial_state.experiment_id,
        trajectories=traj.tolist(),
        n_samples=n_samples,
        horizon=horizon,
        state_labels=initial_state.state_labels,
        config_path=None,
    )
