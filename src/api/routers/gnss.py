"""GNSS spoofing detection endpoints.

POST /gnss/simulate       — Run OSNMA/TESLA simulation, return detection metrics
POST /gnss/verify-key     — Verify a single TESLA key against a chain anchor
POST /gnss/detect         — Stream NAV observations through the TESLA verifier
POST /gnss/spoof-sim      — Monte Carlo signal-level spoofing detection (T1300)
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from gnss.core import (
    DISCLOSURE_DELAY,
    KEY_SIZE_BITS,
    MAC_SIZE_BITS,
    NavMessage,
    OSNMAAuthority,
    OSNMAReceiver,
    TESLAKeyChain,
    run_simulation,
    verify_tesla_key,
)
from gnss.spoof_sim import SimConfig, run_mc_simulation
from schemas import MCSimReport

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class SimulateRequest(BaseModel):
    num_epochs: int = Field(default=40, ge=10, le=500,
                            description="Number of subframe epochs to simulate")
    attack_prob: float = Field(default=0.25, ge=0.0, le=1.0,
                               description="Per-epoch attack injection probability")
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
    tesla_key_hex: str | None = Field(default=None,
                                      description="Disclosed TESLA key hex, if present")
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
    num_chain_epochs: int = Field(default=60, ge=10, le=1000,
                                  description="Total chain length (must cover all epochs)")
    seed: int = Field(default=42, description="Chain generation seed")


class DetectResponse(BaseModel):
    results: list[DetectionResult]
    total_verified: int
    detected_count: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/simulate", response_model=SimulateResponse)
def simulate(req: SimulateRequest) -> SimulateResponse:
    """Run OSNMA/TESLA spoofing simulation and return detection metrics.

    Simulates 4 attack types:
    - **naive_replay**: replays old message → key chain mismatch
    - **modified_replay**: forged ephemeris + random MAC → MAC mismatch
    - **key_disclosure**: valid MAC with disclosed key → receipt safety fail
    - **late_injection**: back-dated message injection → receipt safety fail
    """
    try:
        report = run_simulation(
            num_epochs=req.num_epochs,
            attack_prob=req.attack_prob,
            seed=req.seed,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return SimulateResponse(
        total=report.total,
        spoofed=report.spoofed,
        normal=report.normal,
        tp=report.tp,
        fp=report.fp,
        fn=report.fn,
        tn=report.tn,
        p_fa=report.p_fa,
        p_md=report.p_md,
        precision=report.precision,
        recall=report.recall,
        f1=report.f1,
        by_attack_type={
            k: AttackTypeStat(**v) for k, v in report.by_attack_type.items()
        },
        quantum_detections=report.quantum_detections,
    )


@router.post("/verify-key", response_model=VerifyKeyResponse)
def verify_key(req: VerifyKeyRequest) -> VerifyKeyResponse:
    """Verify a TESLA key against a chain anchor.

    Recomputes the hash chain from anchor_key down to candidate_index
    and checks equality:

        K_i = SHA-256( K_{i+1} || LE32(i) ) [:key_bytes]
        valid ⟺ hash^(anchor_index - candidate_index)(anchor_key) == candidate_key
    """
    try:
        valid = verify_tesla_key(
            candidate_key_hex=req.candidate_key_hex,
            candidate_index=req.candidate_index,
            anchor_key_hex=req.anchor_key_hex,
            anchor_index=req.anchor_index,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid hex or key format: {e}")

    return VerifyKeyResponse(
        valid=valid,
        steps=req.anchor_index - req.candidate_index,
    )


@router.post("/detect", response_model=DetectResponse)
def detect(req: DetectRequest) -> DetectResponse:
    """Run TESLA + receipt-safety + MAC verification on a stream of NAV observations.

    Observations are processed in epoch order.  A VerificationResult is produced
    only when a message includes a disclosed TESLA key (epoch >= disclosure_delay).

    The chain is freshly generated for each request using the provided seed;
    supply the same seed used to generate the observations in a test/simulation
    context, or integrate with real chain parameters for production use.
    """
    try:
        chain = TESLAKeyChain(n=req.num_chain_epochs, seed=req.seed)
        authority = OSNMAAuthority()
        chain_params: dict[str, int] = dict(
            key_size_bits=KEY_SIZE_BITS,
            mac_size_bits=MAC_SIZE_BITS,
            delay=DISCLOSURE_DELAY,
        )
        root_epoch = req.num_chain_epochs - 1
        root_sig = authority.sign_root(chain.root, root_epoch, chain_params)
        rx = OSNMAReceiver(
            authority.public_key, chain_params, root_sig, chain.root, root_epoch, authority
        )

        results: list[DetectionResult] = []
        for obs in sorted(req.observations, key=lambda o: (o.epoch, o.svid)):
            try:
                eph_data = bytes.fromhex(obs.eph_data_hex)
                mac_tag = bytes.fromhex(obs.mac_tag_hex)
                tesla_key = bytes.fromhex(obs.tesla_key_hex) if obs.tesla_key_hex else None
            except ValueError as e:
                raise HTTPException(status_code=422, detail=f"Hex decode error: {e}")

            msg = NavMessage(
                svid=obs.svid,
                epoch=obs.epoch,
                gst=obs.gst,
                eph_data=eph_data,
                tesla_key=tesla_key,
                mac_tag=mac_tag,
            )
            vr = rx.receive(msg, obs.receive_time_epoch)
            if vr is not None:
                results.append(DetectionResult(
                    svid=vr.svid,
                    epoch=vr.epoch,
                    disclosure_epoch=vr.disclosure_epoch,
                    key_valid=vr.key_valid,
                    mac_valid=vr.mac_valid,
                    receipt_safe=vr.receipt_safe,
                    spoofing_detected=vr.detected,
                ))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    detected_count = sum(1 for r in results if r.spoofing_detected)
    return DetectResponse(
        results=results,
        total_verified=len(results),
        detected_count=detected_count,
    )


# ---------------------------------------------------------------------------
# T1300  Monte Carlo signal-level spoofing detection
# ---------------------------------------------------------------------------

_N_MC_MAX: int = 2000   # upper bound to keep response time reasonable


class SpooferSimRequest(BaseModel):
    n_mc: int = Field(default=200, ge=1, le=_N_MC_MAX, description="Monte Carlo runs")
    n_epochs: int = Field(default=80, ge=20, le=500, description="Time steps per run")
    n_sats: int = Field(default=6, ge=5, le=20, description="Number of visible satellites")
    attack_start_frac: float = Field(
        default=0.40, gt=0.0, lt=1.0, description="Attack start as fraction of n_epochs"
    )
    attack_duration_frac: float = Field(
        default=0.35, gt=0.0, lt=1.0, description="Attack duration as fraction of n_epochs"
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
    graph_sigma: float = Field(
        default=1.50, gt=0.0, description="Gaussian kernel bandwidth σ [Hz]"
    )
    false_alarm_rate: float = Field(
        default=0.05, gt=0.0, lt=1.0, description="Neyman-Pearson target false-alarm rate α"
    )
    subset_size: int = Field(
        default=4, ge=2, description="Satellite subset size k (must be < n_sats)"
    )
    random_seed: int = Field(default=42, description="RNG seed for reproducibility")


@router.post("/spoof-sim", response_model=MCSimReport)
def spoof_sim(req: SpooferSimRequest) -> MCSimReport:
    """Monte Carlo GNSS signal-level spoofing detection simulation (T1300).

    Simulates M independent runs of T epochs each.  In each run:

    - Genuine satellites: Doppler deviations Δf_i ∼ N(0, σ_D²)
    - Attack window: meaconing bias  b_i = b_common + δ_i,
      b_common ∼ N(0, σ_bias²),  δ_i ∼ N(0, σ_diff²)
    - Similarity graph: w_{ij} = exp(−|Δf_i − Δf_j|² / σ²)
    - m(t) = det(I + L_w)  — all-forests count (cycle matroid)
    - chi(t) = Σ(Δf_i − mean)² / σ_D²  — Doppler chi-squared
    - Subset S_t selected by greedy Fiedler-value maximisation
    - Detection score T = rᵀ diag(w_S) r  tested against χ²_{1−α}(k−4)

    Returns ROC curve, AUC, detection delay, and PVT degradation statistics.
    """
    try:
        config = SimConfig(
            n_mc=req.n_mc,
            n_epochs=req.n_epochs,
            n_sats=req.n_sats,
            attack_start_frac=req.attack_start_frac,
            attack_duration_frac=req.attack_duration_frac,
            doppler_noise_std=req.doppler_noise_std,
            spoof_bias_std=req.spoof_bias_std,
            spoof_diff_std=req.spoof_diff_std,
            graph_sigma=req.graph_sigma,
            false_alarm_rate=req.false_alarm_rate,
            subset_size=req.subset_size,
            random_seed=req.random_seed,
        )
        return run_mc_simulation(config)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
