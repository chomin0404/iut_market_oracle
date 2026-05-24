"""GNSS Resilience Twin — spoofing detection and fault discrimination API.

OSNMA / TESLA authentication layer:
  POST /gnss/simulate        — OSNMA/TESLA end-to-end simulation (4 attack types)
  POST /gnss/verify-key      — Verify a single TESLA key against a chain anchor
  POST /gnss/detect          — Stream NAV observations through the TESLA verifier

Signal-level detection (Monte Carlo):
  POST /gnss/spoof-sim       — Fisher-combined Doppler + subset-selection MC (T1300)
  POST /gnss/multi-sensor-sim — Multi-sensor percolation graph MC (T1350)

Resilience Twin — 4-layer fault discrimination (flagship, T1500):
  POST /gnss/resilience-sim  — MC benchmark: GM-RAIM + IMM-KF + Spectral + Entropy
  POST /gnss/twin/run        — Probabilistic digital twin: ingest real observations,
                               return per-epoch authenticity / integrity / action
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from fastapi import APIRouter, HTTPException

from api.schemas.gnss import (
    AttackTypeStat,
    DetectionResult,
    DetectRequest,
    DetectResponse,
    MultiSensorSimRequest,
    ResilienceSimRequest,
    SimulateRequest,
    SimulateResponse,
    SpooferSimRequest,
    TwinRunRequest,
    VerifyKeyRequest,
    VerifyKeyResponse,
)
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
from gnss.math_utils import _init_constellation
from gnss.multi_sensor_sim import MultiSensorConfig, run_ms_simulation
from gnss.persistence import new_run_id
from gnss.persistence import save_twin_run as _save_twin_run
from gnss.resilience_twin import (
    EpochDiagnosis,
    ResilienceTwinConfig,
    run_resilience_simulation,
    run_twin_on_observations,
)
from gnss.spoof_sim import SimConfig, run_mc_simulation
from schemas import (
    EpochReport,
    MCSimReport,
    MSSimReport,
    RecommendedAction,
    ResilienceTwinReport,
    TwinRunReport,
)

router = APIRouter()


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
    except (ValueError, RuntimeError, OSError):
        raise HTTPException(status_code=500, detail="Simulation failed due to an internal error.")

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
            k: AttackTypeStat(
                total=int(v["total"]),
                detected=int(v["detected"]),
                p_detect=float(v["p_detect"]),
            )
            for k, v in report.by_attack_type.items()
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
    except (TypeError, AttributeError) as e:
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
                results.append(
                    DetectionResult(
                        svid=vr.svid,
                        epoch=vr.epoch,
                        disclosure_epoch=vr.disclosure_epoch,
                        key_valid=vr.key_valid,
                        mac_valid=vr.mac_valid,
                        receipt_safe=vr.receipt_safe,
                        spoofing_detected=vr.detected,
                    )
                )
    except HTTPException:
        raise
    except (RuntimeError, ValueError, KeyError, AttributeError):
        raise HTTPException(status_code=500, detail="Detection failed due to an internal error.")

    detected_count = sum(1 for r in results if r.spoofing_detected)
    return DetectResponse(
        results=results,
        total_verified=len(results),
        detected_count=detected_count,
    )


# ---------------------------------------------------------------------------
# T1300  Monte Carlo signal-level spoofing detection
# ---------------------------------------------------------------------------


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
            dirichlet_alpha=req.dirichlet_alpha,
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


# ---------------------------------------------------------------------------
# T1350  Multi-sensor Monte Carlo spoofing detection
# ---------------------------------------------------------------------------


@router.post("/multi-sensor-sim", response_model=MSSimReport)
def multi_sensor_sim(req: MultiSensorSimRequest) -> MSSimReport:
    """Monte Carlo multi-sensor GNSS spoofing detection simulation (T1350).

    Sensors fused per epoch:
    - **Pseudorange (PR)**: meaconing drift proxy
    - **Doppler**: Doppler shift consistency
    - **Angle-of-Arrival (AoA)**: geometry diversity
    - **INS residuals**: inertial inconsistency

    Attack model (gradual meaconing):
        x(t) = (1−α)·x_genuine + α·x_spoof,
        α = min(1, max(0, (t−t₀+1)/capture_len))

    Detection: weighted score s = w₁·m + w₂·clip(chi/χ₀) + w₃·clip(lor_dev)
    compared against detect_threshold.

    Returns ROC curve, AUC, detection delay statistics, and per-run traces.
    """
    try:
        config = MultiSensorConfig(
            T=req.T,
            dt=req.dt,
            n_sat=req.n_sat,
            attack_start=req.attack_start,
            attack_end=req.attack_end,
            capture_len=req.capture_len,
            n_nominal=req.n_nominal,
            n_attack=req.n_attack,
            noise_pr=req.noise_pr,
            noise_dopp=req.noise_dopp,
            noise_aoa=req.noise_aoa,
            noise_ins=req.noise_ins,
            carryoff_rate=req.carryoff_rate,
            spoof_aoa_center=req.spoof_aoa_center,
            score_weights=req.score_weights,
            detect_threshold=req.detect_threshold,
            hazard_pos=req.hazard_pos,
            random_seed=req.random_seed,
        )
        return run_ms_simulation(config)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------------------------------------------------------------
# T1500  GNSS Resilience Twin — 4-layer fault discrimination (flagship)
# ---------------------------------------------------------------------------


@router.post("/resilience-sim", response_model=ResilienceTwinReport)
def resilience_sim(req: ResilienceSimRequest) -> ResilienceTwinReport:
    """GNSS Resilience Twin Monte Carlo simulation — 4-layer fault discrimination (T1500).

    Runs `n_mc` trials cycling through 4 fault classes:
    - **NOMINAL**: genuine Doppler deviations only
    - **MULTIPATH**: elevation-correlated noise on low-elevation satellites
    - **HARDWARE_FAULT**: persistent large bias on a single satellite
    - **SPOOFING**: common meaconing bias injected during a Dirichlet-randomised window

    Each trial processes `n_epochs` epochs through 4 fused detection layers:

    | Layer | Algorithm | Output |
    |---|---|---|
    - Layer 1 GM-RAIM: 2-component GMM per satellite → γᵢ posteriors, sign/elev correlation
    - Layer 2 IMM-KF: 3-mode IMM Kalman filter → μ = [μ_nom, μ_mp, μ_spoof]
    - Layer 3 Spectral: Laplacian Fiedler ratio + RMT anomaly → ρ_F, rmt
    - Layer 4 Entropy: Shannon H + KL divergence on 4-class posterior → alert flag

    Fusion: softmax-normalised heuristic Bayesian scorer → MAP classification.

    Returns a 4-class confusion matrix, per-class accuracy, binary ROC/AUC,
    detection/false-alarm rates at threshold 0.5, and mean epoch confidence.
    """
    try:
        config = ResilienceTwinConfig(
            n_mc=req.n_mc,
            n_epochs=req.n_epochs,
            n_sats=req.n_sats,
            doppler_noise_std=req.doppler_noise_std,
            spoof_bias_std=req.spoof_bias_std,
            spoof_diff_std=req.spoof_diff_std,
            graph_sigma=req.graph_sigma,
            dirichlet_alpha=req.dirichlet_alpha,
            random_seed=req.random_seed,
        )
        return run_resilience_simulation(config)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------------------------------------------------------------
# T1500  Probabilistic Digital Twin — observation-driven inference
# ---------------------------------------------------------------------------

# Recommended-action decision thresholds (Priority: spoof > hw > mp > monitor)
_ACT_SPOOF_THRESH: float = 0.50
_ACT_HW_THRESH: float = 0.50
_ACT_MP_THRESH: float = 0.50
_ACT_NOMINAL_MIN: float = 0.70

# Ordered severity for worst-action aggregation
_ACTION_SEVERITY: dict[RecommendedAction, int] = {
    RecommendedAction.NOMINAL: 0,
    RecommendedAction.MONITOR: 1,
    RecommendedAction.REDUCE_TRUST: 2,
    RecommendedAction.SWITCH_SOURCE: 3,
    RecommendedAction.GROUND_IMMEDIATELY: 4,
}


def _decide_action(
    fault_posterior: tuple[float, float, float, float],
    entropy_alert: bool,
) -> tuple[RecommendedAction, str]:
    """Map 4-class fault posterior + entropy alert to a recommended action.

    Evaluated in priority order (highest severity first):
        P(spoofing) > 0.50  → GROUND_IMMEDIATELY
        P(hw_fault) > 0.50  → SWITCH_SOURCE
        P(multipath) > 0.50 → REDUCE_TRUST
        P(nominal) < 0.70 or entropy_alert → MONITOR
        otherwise           → NOMINAL
    """
    p_nom, p_mp, p_hw, p_spoof = fault_posterior

    if p_spoof > _ACT_SPOOF_THRESH:
        return (
            RecommendedAction.GROUND_IMMEDIATELY,
            f"Spoofing posterior P={p_spoof:.2f} exceeds threshold "
            f"{_ACT_SPOOF_THRESH} — coordinated cross-satellite bias detected across all layers",
        )
    if p_hw > _ACT_HW_THRESH:
        return (
            RecommendedAction.SWITCH_SOURCE,
            f"Hardware-fault posterior P={p_hw:.2f} exceeds threshold "
            f"{_ACT_HW_THRESH} — isolated satellite outlier; exclude and recompute PVT",
        )
    if p_mp > _ACT_MP_THRESH:
        return (
            RecommendedAction.REDUCE_TRUST,
            f"Multipath posterior P={p_mp:.2f} exceeds threshold "
            f"{_ACT_MP_THRESH} — deweight low-elevation satellites or widen measurement noise",
        )
    if p_nom < _ACT_NOMINAL_MIN or entropy_alert:
        reason_parts: list[str] = []
        if p_nom < _ACT_NOMINAL_MIN:
            reason_parts.append(f"nominal posterior P={p_nom:.2f} < {_ACT_NOMINAL_MIN}")
        if entropy_alert:
            reason_parts.append("entropy monitor raised alert")
        return (
            RecommendedAction.MONITOR,
            "; ".join(reason_parts) + " — increase monitoring rate",
        )
    return RecommendedAction.NOMINAL, "All 4 layers consistent with nominal operation"


def _diag_to_epoch_report(diag: EpochDiagnosis) -> EpochReport:
    """Convert a single EpochDiagnosis to its EpochReport API representation."""
    fp = diag.fault_posterior  # (P_nom, P_mp, P_hw, P_spoof)
    p_nom, p_mp, p_hw, p_spoof = fp
    action, reason = _decide_action(fp, diag.entropy.alert)
    return EpochReport(
        epoch=diag.t,
        authenticity={"genuine": float(p_nom + p_mp + p_hw), "spoofed": float(p_spoof)},
        integrity={"nominal": float(p_nom), "degraded": float(p_mp + p_hw + p_spoof)},
        fault_posterior={
            "nominal": float(p_nom),
            "multipath": float(p_mp),
            "hardware_fault": float(p_hw),
            "spoofing": float(p_spoof),
        },
        diagnosis=diag.diagnosis.value,
        confidence=diag.confidence,
        recommended_action=action,
        action_reason=reason,
        entropy_alert=diag.entropy.alert,
        gmm_n_fault=diag.integrity.gmm.n_fault,
        imm_spoof_weight=diag.integrity.imm.mode_weights[2],
        spectral_fiedler_ratio=diag.structure.spectral.fiedler_ratio,
        ins_chi2_vel=diag.integrity.ins.chi2_vel,
        ins_alert=diag.integrity.ins.alert,
        coop_parity_chi2=diag.integrity.coop_raim.parity_chi2,
        coop_parity_alert=diag.integrity.coop_raim.parity_alert,
        osnma_auth_fraction=diag.auth.osnma.auth_fraction,
        structural_fiedler_streak=diag.structure.structural.fiedler_streak,
        structural_alert=diag.structure.structural.alert,
        auth_p_spoofed=diag.auth.p_spoofed,
        integrity_base_fault=1.0 - diag.integrity.base_posterior[0],
        structure_intensity=diag.structure.structure_intensity,
        huh_det_ratio=diag.integrity.huh.det_ratio,
        huh_n_excluded=diag.integrity.huh.n_excluded,
        huh_log_concavity_ratio=diag.integrity.huh.log_concavity_ratio,
        phase_percolation_threshold=diag.structure.phase.percolation_threshold,
        phase_susceptibility_peak=diag.structure.phase.susceptibility_peak,
        phase_alert=diag.structure.phase.phase_alert,
    )


@router.post("/twin/run", response_model=TwinRunReport)
def twin_run(req: TwinRunRequest) -> TwinRunReport:
    """GNSS Resilience Twin — probabilistic digital twin inference on real observations.

    Synchronises a stream of receiver observations to the virtual twin and returns
    per-epoch probabilistic assessments, including:

    - **authenticity** — P(genuine signal) vs P(spoofed signal)
    - **integrity** — P(nominal PVT) vs P(any fault degrading PVT)
    - **fault_posterior** — full 4-class posterior: nominal / multipath / hw_fault / spoofing
    - **recommended_action** — operator directive derived from posteriors and entropy alerts

    ### Input geometry
    Provide `los_vectors` when satellite ephemerides are available (higher PVT fidelity).
    If omitted, a Fibonacci-lattice constellation is used as the geometry model.

    ### Observation format
    Each `ObservationEpoch` carries `doppler_residuals` — the per-satellite difference
    between measured and predicted Doppler shift: Δf_i = f_measured_i − f_predicted_i [Hz].
    Supply `elevations_deg` if available; they improve GM-RAIM elevation-adjusted noise.

    ### Sliding-window operation
    For continuous real-time use, call this endpoint with a rolling window of recent
    epochs (e.g., 30–120 s). The twin reinitialises per request to ensure reproducibility.
    """
    run_id = new_run_id()

    try:
        # ── Build satellite geometry ────────────────────────────────────────
        if req.los_vectors is not None:
            los = np.array(req.los_vectors, dtype=float)
            # Normalise rows to unit vectors
            norms = np.linalg.norm(los, axis=1, keepdims=True)
            los = los / np.where(norms > 0, norms, 1.0)
        else:
            los = _init_constellation(req.n_sats)

        # ── Build per-epoch arrays ──────────────────────────────────────────
        doppler_seq: list[np.ndarray] = []
        ins_seq: list[np.ndarray | None] = []
        osnma_seq: list[list[bool] | None] = []
        elevations_rad: np.ndarray | None = None  # per-epoch override (last epoch wins)

        for obs in req.observations:
            doppler_seq.append(np.array(obs.doppler_residuals, dtype=float))
            if obs.elevations_deg is not None:
                # Convert degrees → radians; use the last epoch's elevations
                # (typically stable over short windows)
                elevations_rad = np.deg2rad(np.array(obs.elevations_deg, dtype=float))
            ins_seq.append(
                np.array(obs.ins_velocity_ms, dtype=float)
                if obs.ins_velocity_ms is not None
                else None
            )
            osnma_seq.append(obs.osnma_auth_per_sat)

        # ── Run the digital twin ────────────────────────────────────────────
        epoch_diags = run_twin_on_observations(
            doppler_sequence=doppler_seq,
            los=los,
            elevations=elevations_rad,
            noise_std=req.doppler_noise_std,
            graph_sigma=req.graph_sigma,
            ins_sequence=ins_seq,
            osnma_sequence=osnma_seq,
            ins_noise_std=req.ins_noise_std,
        )

        # ── Convert EpochDiagnosis → EpochReport ────────────────────────────
        epoch_reports: list[EpochReport] = [_diag_to_epoch_report(d) for d in epoch_diags]

        # ── Aggregate summary ───────────────────────────────────────────────
        diag_counts = Counter(r.diagnosis for r in epoch_reports)
        dominant_diagnosis = diag_counts.most_common(1)[0][0]

        mean_auth = float(
            sum(r.authenticity["genuine"] for r in epoch_reports) / len(epoch_reports)
        )
        mean_integ = float(sum(r.integrity["nominal"] for r in epoch_reports) / len(epoch_reports))
        alert_epochs = [r.epoch for r in epoch_reports if r.entropy_alert]

        spoof_epochs = [r.epoch for r in epoch_reports if r.fault_posterior["spoofing"] > 0.50]
        spoofing_window = [spoof_epochs[0], spoof_epochs[-1]] if spoof_epochs else None

        worst_action = max(
            (r.recommended_action for r in epoch_reports),
            key=lambda a: _ACTION_SEVERITY[a],
        )

        report = TwinRunReport(
            epoch_reports=epoch_reports,
            n_epochs=len(epoch_reports),
            n_sats=req.n_sats,
            dominant_diagnosis=dominant_diagnosis,
            mean_authenticity_genuine=mean_auth,
            mean_integrity_nominal=mean_integ,
            alert_epochs=alert_epochs,
            spoofing_window=spoofing_window,
            worst_action=worst_action,
            run_id=run_id,
        )

        if req.save:
            result_path = _save_twin_run(
                req.model_dump(mode="json"),
                report.model_dump(mode="json"),
                run_id,
            )
            report = report.model_copy(update={"result_path": result_path})

        return report

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
