"""Tests for src/twin/simulator.py and src/twin/calibrator.py (T800)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from schemas import DigitalTwinState, PriorSpec, SimulationResult
from twin.calibrator import calibrate
from twin.simulator import _default_transition_matrix, simulate

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

EXP_ID = "exp-001"

NORMAL_PRIOR = PriorSpec(distribution="normal", params={"mu": 0.10, "sigma": 0.05})


def _make_state(
    vector: list[float] | None = None,
    labels: list[str] | None = None,
) -> DigitalTwinState:
    vector = vector or [5.0, 0.10, -2.3]
    labels = labels or ["log_revenue", "growth_rate", "log_volatility"]
    return DigitalTwinState(
        experiment_id=EXP_ID,
        state_vector=vector,
        state_labels=labels,
    )


def _rng(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# simulator — determinism
# ---------------------------------------------------------------------------


class TestSimulateDeterminism:
    def test_same_seed_produces_identical_trajectories(self):
        state = _make_state()
        result_a = simulate(state, horizon=10, n_samples=50, process_noise_std=0.02, rng=_rng(42))
        result_b = simulate(state, horizon=10, n_samples=50, process_noise_std=0.02, rng=_rng(42))
        assert result_a.trajectories == result_b.trajectories

    def test_different_seeds_produce_different_trajectories(self):
        state = _make_state()
        result_a = simulate(state, horizon=5, n_samples=20, process_noise_std=0.05, rng=_rng(1))
        result_b = simulate(state, horizon=5, n_samples=20, process_noise_std=0.05, rng=_rng(2))
        # At least one sample must differ (probability of collision is negligible)
        assert result_a.trajectories != result_b.trajectories


# ---------------------------------------------------------------------------
# simulator — dimension consistency
# ---------------------------------------------------------------------------


class TestSimulateDimensions:
    def test_n_samples_correct(self):
        result = simulate(
            _make_state(), horizon=4, n_samples=30, process_noise_std=0.01, rng=_rng()
        )
        assert result.n_samples == 30
        assert len(result.trajectories) == 30

    def test_horizon_gives_horizon_plus_one_steps(self):
        horizon = 8
        result = simulate(
            _make_state(), horizon=horizon, n_samples=10, process_noise_std=0.01, rng=_rng()
        )
        assert result.horizon == horizon
        for traj in result.trajectories:
            assert len(traj) == horizon + 1

    def test_state_dim_preserved_across_steps(self):
        state = _make_state()
        d = len(state.state_vector)
        result = simulate(state, horizon=5, n_samples=5, process_noise_std=0.01, rng=_rng())
        for traj in result.trajectories:
            for step_vec in traj:
                assert len(step_vec) == d

    def test_state_labels_propagated(self):
        state = _make_state()
        result = simulate(state, horizon=3, n_samples=5, process_noise_std=0.0, rng=_rng())
        assert result.state_labels == state.state_labels

    def test_experiment_id_propagated(self):
        result = simulate(_make_state(), horizon=3, n_samples=5, process_noise_std=0.0, rng=_rng())
        assert result.experiment_id == EXP_ID


# ---------------------------------------------------------------------------
# simulator — initial state preservation
# ---------------------------------------------------------------------------


class TestSimulateInitialState:
    def test_step_zero_equals_initial_state(self):
        x0 = [5.0, 0.10, -2.3]
        state = _make_state(vector=x0)
        result = simulate(state, horizon=5, n_samples=10, process_noise_std=0.0, rng=_rng())
        for traj in result.trajectories:
            assert traj[0] == pytest.approx(x0, abs=1e-12)

    def test_zero_noise_produces_deterministic_trajectory(self):
        """With σ=0, all samples must follow the exact transition x_{t+1} = F x_t."""
        state = _make_state(vector=[1.0, 0.05, 0.0])
        result = simulate(state, horizon=4, n_samples=5, process_noise_std=0.0, rng=_rng())
        first = np.array(result.trajectories[0])
        for traj in result.trajectories[1:]:
            np.testing.assert_allclose(np.array(traj), first, atol=1e-12)


# ---------------------------------------------------------------------------
# simulator — transition matrix
# ---------------------------------------------------------------------------


class TestTransitionMatrix:
    def test_default_matrix_shape(self):
        F = _default_transition_matrix(state_dim=3, dt=0.25)
        assert F.shape == (3, 3)

    def test_default_matrix_growth_coupling(self):
        """F[0, 1] == dt: log_revenue accumulates growth_rate over one step."""
        F = _default_transition_matrix(state_dim=3, dt=0.25)
        assert F[0, 1] == pytest.approx(0.25)

    def test_custom_transition_matrix_used(self):
        """Identity F with zero noise → all steps equal the initial state."""
        d = 3
        state = _make_state(vector=[2.0, 0.08, -1.0])
        F = np.eye(d)
        result = simulate(
            state, horizon=5, n_samples=3, process_noise_std=0.0, rng=_rng(), transition_matrix=F
        )
        for traj in result.trajectories:
            for step_vec in traj:
                assert step_vec == pytest.approx(state.state_vector, abs=1e-12)

    def test_invalid_transition_matrix_shape_raises(self):
        state = _make_state()
        bad_F = np.eye(2)  # wrong shape for 3-D state
        with pytest.raises(ValueError, match="transition_matrix must be"):
            simulate(
                state,
                horizon=3,
                n_samples=5,
                process_noise_std=0.01,
                rng=_rng(),
                transition_matrix=bad_F,
            )


# ---------------------------------------------------------------------------
# simulator — invalid inputs
# ---------------------------------------------------------------------------


class TestSimulateInvalidInputs:
    def test_horizon_zero_raises(self):
        with pytest.raises(ValueError, match="horizon must be >= 1"):
            simulate(_make_state(), horizon=0, n_samples=5, process_noise_std=0.01, rng=_rng())

    def test_n_samples_zero_raises(self):
        with pytest.raises(ValueError, match="n_samples must be >= 1"):
            simulate(_make_state(), horizon=3, n_samples=0, process_noise_std=0.01, rng=_rng())

    def test_negative_noise_raises(self):
        with pytest.raises(ValueError, match="process_noise_std must be >= 0"):
            simulate(_make_state(), horizon=3, n_samples=5, process_noise_std=-0.01, rng=_rng())


# ---------------------------------------------------------------------------
# calibrator — basic correctness
# ---------------------------------------------------------------------------


class TestCalibrateBasic:
    def test_no_observations_returns_prior_mean(self):
        posterior, state = calibrate([], NORMAL_PRIOR, EXP_ID, obs_precision=25.0)
        assert posterior.mean == pytest.approx(0.10, abs=1e-9)
        assert state.state_vector[0] == pytest.approx(0.10, abs=1e-9)

    def test_state_vector_equals_posterior_mean(self):
        obs = [0.15, 0.18, 0.12]
        posterior, state = calibrate(obs, NORMAL_PRIOR, EXP_ID, obs_precision=25.0)
        assert state.state_vector[0] == pytest.approx(posterior.mean, abs=1e-12)

    def test_step_equals_number_of_observations(self):
        obs = [0.10, 0.12, 0.09, 0.11]
        _, state = calibrate(obs, NORMAL_PRIOR, EXP_ID)
        assert state.step == len(obs)

    def test_state_label_is_mu(self):
        _, state = calibrate([0.10], NORMAL_PRIOR, EXP_ID)
        assert state.state_labels == ["mu"]

    def test_experiment_id_propagated(self):
        _, state = calibrate([], NORMAL_PRIOR, EXP_ID)
        assert state.experiment_id == EXP_ID

    def test_param_snapshot_contains_mu_and_sigma(self):
        _, state = calibrate([0.12], NORMAL_PRIOR, EXP_ID)
        assert "mu" in state.param_snapshot
        assert "sigma" in state.param_snapshot

    def test_param_snapshot_sigma_consistent_with_variance(self):
        posterior, state = calibrate([0.12, 0.14], NORMAL_PRIOR, EXP_ID, obs_precision=10.0)
        expected_sigma = math.sqrt(posterior.variance)
        assert state.param_snapshot["sigma"] == pytest.approx(expected_sigma, rel=1e-9)


# ---------------------------------------------------------------------------
# calibrator — variance reduction
# ---------------------------------------------------------------------------


class TestCalibrateVarianceReduction:
    def test_more_observations_reduce_posterior_variance(self):
        obs_value = 0.12
        post1, _ = calibrate([obs_value], NORMAL_PRIOR, EXP_ID, obs_precision=25.0)
        post10, _ = calibrate([obs_value] * 10, NORMAL_PRIOR, EXP_ID, obs_precision=25.0)
        assert post10.variance < post1.variance

    def test_high_precision_observation_dominates_prior(self):
        """Very precise observation should pull posterior mean close to observation."""
        obs = [0.30]
        posterior, _ = calibrate(obs, NORMAL_PRIOR, EXP_ID, obs_precision=1e6)
        assert posterior.mean == pytest.approx(0.30, abs=1e-2)

    def test_observations_above_prior_raise_mean(self):
        prior_post, _ = calibrate([], NORMAL_PRIOR, EXP_ID)
        updated_post, _ = calibrate([0.25, 0.28], NORMAL_PRIOR, EXP_ID, obs_precision=25.0)
        assert updated_post.mean > prior_post.mean


# ---------------------------------------------------------------------------
# calibrator — invalid inputs
# ---------------------------------------------------------------------------


class TestCalibrateInvalidInputs:
    def test_non_normal_prior_raises(self):
        beta_prior = PriorSpec(distribution="beta", params={"alpha": 2.0, "beta": 18.0})
        with pytest.raises(ValueError, match="distribution='normal'"):
            calibrate([0.1], beta_prior, EXP_ID)

    def test_zero_obs_precision_raises(self):
        with pytest.raises(ValueError, match="obs_precision must be > 0"):
            calibrate([0.1], NORMAL_PRIOR, EXP_ID, obs_precision=0.0)

    def test_negative_obs_precision_raises(self):
        with pytest.raises(ValueError, match="obs_precision must be > 0"):
            calibrate([0.1], NORMAL_PRIOR, EXP_ID, obs_precision=-1.0)


# ---------------------------------------------------------------------------
# schemas — DigitalTwinState validation
# ---------------------------------------------------------------------------


class TestDigitalTwinStateSchema:
    def test_labels_length_mismatch_raises(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="state_labels length"):
            DigitalTwinState(
                experiment_id=EXP_ID,
                state_vector=[1.0, 2.0],
                state_labels=["only_one"],
            )

    def test_negative_step_raises(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            DigitalTwinState(
                experiment_id=EXP_ID,
                state_vector=[1.0],
                state_labels=["x"],
                step=-1,
            )


# ---------------------------------------------------------------------------
# schemas — SimulationResult validation
# ---------------------------------------------------------------------------


class TestSimulationResultSchema:
    def _valid_traj(self, n_samples: int, horizon: int, state_dim: int) -> list[list[list[float]]]:
        return [
            [[float(i + t + k) for k in range(state_dim)] for t in range(horizon + 1)]
            for i in range(n_samples)
        ]

    def test_mismatched_n_samples_raises(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="trajectories count"):
            SimulationResult(
                experiment_id=EXP_ID,
                trajectories=self._valid_traj(3, 4, 2),
                n_samples=5,  # mismatch: 3 trajectories but n_samples=5
                horizon=4,
                state_labels=["a", "b"],
            )

    def test_mismatched_horizon_raises(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="steps"):
            SimulationResult(
                experiment_id=EXP_ID,
                trajectories=self._valid_traj(2, 4, 2),
                n_samples=2,
                horizon=6,  # mismatch: trajectories have 5 steps but horizon+1=7
                state_labels=["a", "b"],
            )
