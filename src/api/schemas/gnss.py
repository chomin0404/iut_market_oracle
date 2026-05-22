"""Request/response schemas for the GNSS router (T1300 / T1350 / T1500)."""

from __future__ import annotations

import math

from pydantic import BaseModel, Field, model_validator

from schemas import ObservationEpoch

# ---------------------------------------------------------------------------
# T1300  OSNMA / TESLA authentication endpoints
# ---------------------------------------------------------------------------


class SimulateRequest(BaseModel):
    num_epochs: int = Field(
        default=40, ge=10, le=500, description="Number of subframe epochs to simulate"
    )
    attack_prob: float = Field(
        default=0.25, ge=0.0, le=1.0, description="Per-epoch attack injection probability"
    )
    seed: int = Field(default=42, description="RNG seed for reproducibility")


class AttackTypeStat(BaseModel):
    total: int
    detected: int
    p_detect: float


class SimulateResponse(BaseModel):
    total: int
    spoofed: int
    normal: int
    tp: int
    fp: int
    fn: int
    tn: int
    p_fa: float
    p_md: float
    precision: float
    recall: float
    f1: float
    by_attack_type: dict[str, AttackTypeStat]
    quantum_detections: int = Field(
        default=0,
        description="key_compromise attacks caught exclusively by quantum fidelity layer",
    )


class VerifyKeyRequest(BaseModel):
    candidate_key_hex: str = Field(..., description="Hex-encoded TESLA key to verify")
    candidate_index: int = Field(..., ge=0, description="Chain index of the candidate key")
    anchor_key_hex: str = Field(..., description="Hex-encoded verified anchor key")
    anchor_index: int = Field(..., ge=1, description="Chain index of the anchor (> candidate)")


class VerifyKeyResponse(BaseModel):
    valid: bool
    steps: int = Field(description="Number of hash steps from anchor to candidate")


class NavObservation(BaseModel):
    """Single NAV message observation from a GNSS receiver."""

    svid: int = Field(..., ge=1, le=36, description="Satellite vehicle ID")
    epoch: int = Field(..., ge=0, description="Subframe epoch number")
    gst: int = Field(..., ge=0, description="Galileo System Time [s]")
    eph_data_hex: str = Field(..., description="Hex-encoded ephemeris data (32 bytes = 64 chars)")
    mac_tag_hex: str = Field(..., description="Hex-encoded MAC tag (5 bytes = 10 chars)")
    tesla_key_hex: str | None = Field(
        default=None, description="Disclosed TESLA key hex, if present"
    )
    receive_time_epoch: float = Field(..., description="Actual receive time [epoch units]")


class DetectionResult(BaseModel):
    svid: int
    epoch: int
    disclosure_epoch: int
    key_valid: bool
    mac_valid: bool
    receipt_safe: bool
    spoofing_detected: bool


class DetectRequest(BaseModel):
    observations: list[NavObservation] = Field(..., min_length=1)
    num_chain_epochs: int = Field(
        default=60, ge=10, le=1000, description="Total chain length (must cover all epochs)"
    )
    seed: int = Field(default=42, description="Chain generation seed")


class DetectResponse(BaseModel):
    results: list[DetectionResult]
    total_verified: int
    detected_count: int


# ---------------------------------------------------------------------------
# T1300  Signal-level MC spoofing detection
# ---------------------------------------------------------------------------

_N_MC_MAX: int = 2000  # upper bound to keep response time reasonable


class SpooferSimRequest(BaseModel):
    n_mc: int = Field(default=200, ge=1, le=_N_MC_MAX, description="Monte Carlo runs")
    n_epochs: int = Field(default=80, ge=20, le=500, description="Time steps per run")
    n_sats: int = Field(default=6, ge=5, le=20, description="Number of visible satellites")
    dirichlet_alpha: float = Field(
        default=2.0, gt=0.0, description="Symmetric Dirichlet concentration for attack window"
    )
    doppler_noise_std: float = Field(
        default=0.30, gt=0.0, description="Genuine Doppler noise 1-σ [Hz]"
    )
    spoof_bias_std: float = Field(
        default=2.50, gt=0.0, description="Common meaconing bias 1-σ [Hz]"
    )
    spoof_diff_std: float = Field(
        default=0.80, ge=0.0, description="Per-satellite differential spoofing noise 1-σ [Hz]"
    )
    graph_sigma: float = Field(default=1.50, gt=0.0, description="Gaussian kernel bandwidth σ [Hz]")
    false_alarm_rate: float = Field(
        default=0.05, gt=0.0, lt=1.0, description="Neyman-Pearson target false-alarm rate α"
    )
    subset_size: int = Field(
        default=4, ge=2, description="Satellite subset size k (must be < n_sats)"
    )
    random_seed: int = Field(default=42, description="RNG seed for reproducibility")


# ---------------------------------------------------------------------------
# T1350  Multi-sensor MC spoofing detection
# ---------------------------------------------------------------------------

_N_MS_MC_MAX: int = 500  # upper bound to keep response time reasonable


class MultiSensorSimRequest(BaseModel):
    T: int = Field(default=200, ge=20, le=1000, description="Total epochs per trial")
    dt: float = Field(default=1.0, gt=0.0, description="Epoch duration [s]")
    n_sat: int = Field(default=8, ge=4, le=32, description="Number of visible satellites")
    attack_start: int = Field(default=80, ge=0, description="First attacked epoch")
    attack_end: int = Field(default=140, ge=1, description="Last attacked epoch (inclusive)")
    capture_len: int = Field(default=20, ge=1, description="Gradual capture ramp length [epochs]")
    n_nominal: int = Field(default=100, ge=1, le=_N_MS_MC_MAX, description="Genuine MC trials")
    n_attack: int = Field(default=100, ge=1, le=_N_MS_MC_MAX, description="Attack MC trials")
    noise_pr: float = Field(default=2.0, gt=0.0, description="PR noise 1-σ [m]")
    noise_dopp: float = Field(default=0.08, gt=0.0, description="Doppler noise 1-σ [Hz]")
    noise_aoa: float = Field(default=3.0, gt=0.0, description="AoA noise 1-σ [deg]")
    noise_ins: float = Field(default=1.5, gt=0.0, description="INS residual noise 1-σ [m/s]")
    carryoff_rate: float = Field(default=4.0, gt=0.0, description="PR drift rate [m/epoch]")
    spoof_aoa_center: float = Field(default=30.0, description="Mean spoofed AoA [deg]")
    score_weights: tuple[float, float, float] = Field(
        default=(0.55, 0.25, 0.20),
        description="Detection score weights (w_m, w_chi, w_lor_dev)",
    )
    detect_threshold: float = Field(default=0.62, gt=0.0, description="Alarm threshold")
    hazard_pos: float = Field(default=150.0, gt=0.0, description="Hazard pos-error threshold [m]")
    random_seed: int = Field(default=42, description="RNG seed for reproducibility")


# ---------------------------------------------------------------------------
# T1500  GNSS Resilience Twin — 4-layer fault discrimination
# ---------------------------------------------------------------------------

_N_RT_MC_MAX: int = 2000  # upper bound to keep response time reasonable


class ResilienceSimRequest(BaseModel):
    n_mc: int = Field(
        default=400,
        ge=4,
        le=_N_RT_MC_MAX,
        description="Total MC trials (cycles NOMINAL/MULTIPATH/HW_FAULT/SPOOFING in round-robin)",
    )
    n_epochs: int = Field(default=80, ge=10, le=500, description="Time steps per trial")
    n_sats: int = Field(default=6, ge=5, le=20, description="Number of visible satellites")
    doppler_noise_std: float = Field(
        default=0.30, gt=0.0, description="Genuine Doppler noise 1-σ [Hz]"
    )
    spoof_bias_std: float = Field(
        default=2.50, gt=0.0, description="Common meaconing bias 1-σ [Hz]"
    )
    spoof_diff_std: float = Field(
        default=0.80, ge=0.0, description="Per-satellite differential spoofing noise 1-σ [Hz]"
    )
    graph_sigma: float = Field(default=1.50, gt=0.0, description="Gaussian kernel bandwidth σ [Hz]")
    dirichlet_alpha: float = Field(
        default=2.0, gt=0.0, description="Symmetric Dirichlet concentration for attack window"
    )
    random_seed: int = Field(default=42, description="RNG seed for reproducibility")


# ---------------------------------------------------------------------------
# T1500  Probabilistic Digital Twin — observation-driven inference
# ---------------------------------------------------------------------------

_TWIN_EPOCHS_MAX: int = 5000


class TwinRunRequest(BaseModel):
    """Request body for POST /gnss/twin/run.

    observations:     Ordered list of ObservationEpoch.  Length = number of epochs T.
    n_sats:           Number of visible satellites. Must equal len(doppler_residuals)
                      in every ObservationEpoch.
    los_vectors:      Optional (n_sats × 3) unit LOS vectors for IMM-KF geometry.
                      If omitted, a Fibonacci-lattice constellation is used.
    doppler_noise_std: Nominal Doppler noise 1-σ [Hz]. Used for GM-RAIM and IMM-KF.
    graph_sigma:      Gaussian kernel bandwidth σ [Hz] for spectral layer.
    """

    observations: list[ObservationEpoch] = Field(
        ...,
        min_length=2,
        max_length=_TWIN_EPOCHS_MAX,
        description="Ordered observation sequence (2 – 5000 epochs)",
    )
    n_sats: int = Field(default=6, ge=5, le=20, description="Number of visible satellites")
    los_vectors: list[list[float]] | None = Field(
        default=None,
        description=(
            "(n_sats × 3) unit LOS vectors. Each row must have exactly 3 components. "
            "If omitted, a Fibonacci-lattice constellation is generated automatically."
        ),
    )
    doppler_noise_std: float = Field(
        default=0.30, gt=0.0, description="Nominal Doppler noise 1-σ [Hz]"
    )
    graph_sigma: float = Field(default=1.50, gt=0.0, description="Gaussian kernel bandwidth σ [Hz]")
    ins_noise_std: float = Field(
        default=0.05,
        gt=0.0,
        description=(
            "INS velocity noise 1-σ [m/s]. Used by Layer 5 INS coupling chi² test. "
            "Only relevant when ObservationEpoch.ins_velocity_ms is supplied."
        ),
    )
    save: bool = Field(
        default=True,
        description=(
            "Persist the run request and report to output/<run_id>/twin_run.json. "
            "Set False to skip file I/O (e.g. in latency-sensitive contexts)."
        ),
    )

    @model_validator(mode="after")
    def _validate_dimensions(self) -> TwinRunRequest:
        # Every epoch must supply exactly n_sats residuals
        for i, obs in enumerate(self.observations):
            if len(obs.doppler_residuals) != self.n_sats:
                raise ValueError(
                    f"observations[{i}].doppler_residuals has {len(obs.doppler_residuals)} "
                    f"entries; expected n_sats={self.n_sats}"
                )
            if obs.elevations_deg is not None and len(obs.elevations_deg) != self.n_sats:
                raise ValueError(
                    f"observations[{i}].elevations_deg has {len(obs.elevations_deg)} "
                    f"entries; expected n_sats={self.n_sats}"
                )
        # Validate LOS matrix shape and unit-vector norm
        if self.los_vectors is not None:
            if len(self.los_vectors) != self.n_sats:
                raise ValueError(
                    f"los_vectors has {len(self.los_vectors)} rows; expected n_sats={self.n_sats}"
                )
            for j, row in enumerate(self.los_vectors):
                if len(row) != 3:
                    raise ValueError(f"los_vectors[{j}] must have 3 components, got {len(row)}")
                norm = math.sqrt(sum(x * x for x in row))
                if not (0.5 < norm < 2.0):
                    raise ValueError(
                        f"los_vectors[{j}] norm={norm:.3f} is far from 1.0 — supply unit vectors"
                    )
        return self
