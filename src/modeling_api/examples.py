"""Shared OpenAPI response/request examples for modeling endpoints.

These constants are referenced by ``json_schema_extra`` in ``src/schemas.py``
so that ``/v1/models/recommend`` and ``/v1/models/formalize`` (and related
skills) all display identical, consistent examples in the docs UI.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Canonical GPS-spoofing scenario used across all modeling schemas
# ---------------------------------------------------------------------------

EXAMPLE_IDEA_INPUT: dict = {
    "title": "GPS spoofing detection and defense",
    "description": (
        "Detect spoofing from sequential inconsistencies across GNSS, IMU, "
        "clock bias, and signal-quality measurements, then optimize defense "
        "actions under uncertainty."
    ),
    "domain": "navigation_security",
    "goal_type": "anomaly_detection",
    "time_horizon": "sequential",
    "data_regime": "small",
    "uncertainty_level": "high",
    "physical_constraints": True,
    "decision_variables_present": True,
    "latent_state_present": True,
}

EXAMPLE_PARSED_IDEA_RESPONSE: dict = {
    "problem_structure": {
        "is_sequential": True,
        "has_latent_state": True,
        "has_decision_variables": True,
        "has_physical_constraints": True,
        "is_high_uncertainty": True,
        "is_data_scarce": True,
    },
    "candidate_families": [
        "kalman_filter",
        "bayesian_sequential_detector",
        "particle_filter",
        "hmm",
    ],
    "missing_information": [
        "Number of GNSS satellites tracked",
        "IMU sampling rate [Hz]",
        "Expected spoofing attack duration [epochs]",
    ],
}

EXAMPLE_MODEL_RECOMMENDATION: dict = {
    "problem_type": "sequential_anomaly_detection",
    "recommended_models": [
        "extended_kalman_filter",
        "particle_filter",
        "hmm_viterbi",
    ],
    "rationale": [
        "EKF handles nonlinear GNSS measurement equations with low latency",
        "Particle filter captures non-Gaussian spoofing signal distributions",
        "HMM-Viterbi provides hard regime assignment for alarm logic",
    ],
}

EXAMPLE_MODEL_SPEC: dict = {
    "problem_type": "sequential_anomaly_detection",
    "objective": "Maximise detection probability subject to P_FA ≤ 0.01",
    "state_variables": [
        "position_error [m]",
        "clock_bias [s]",
        "spoofing_flag ∈ {0,1}",
    ],
    "observables": [
        "pseudorange_residuals [m]",
        "C/N0 [dB-Hz]",
        "IMU_accel [m/s²]",
    ],
    "parameters": ["threshold τ", "noise_variance σ²", "transition_prob p"],
    "constraints": ["P_FA ≤ 0.01", "τ > 0"],
    "uncertainty": {
        "observation_noise": "N(0, σ²_obs)",
        "process_noise": "N(0, Q)",
    },
    "equations": [
        "x_{t+1} = F x_t + w_t",
        "z_t = H x_t + v_t",
        "T(t) = z_t^T S_t^{-1} z_t",
    ],
    "priors": {"τ": "Gamma(2, 1)", "σ²": "InvGamma(3, 1)"},
    "loss_function": "Neyman-Pearson: max P_D s.t. P_FA ≤ α",
    "solver": "Extended Kalman Filter + CUSUM threshold test",
    "outputs": [
        "alarm_flag",
        "detection_delay [epochs]",
        "pvt_error_norm [Hz]",
    ],
    "assumptions": [
        "Gaussian process noise",
        "Attack onset epoch known in simulation",
    ],
    "evidence_needed": [
        "Historical pseudorange data",
        "IMU calibration records",
    ],
}
