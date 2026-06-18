"""GNSS Resilience Twin — per-epoch result dataclasses (T1500).

Provides the structured output types for the 4-pillar architecture:

    AuthenticationScore  — Pillar 1 (OSNMA)
    IntegrityScore       — Pillar 2 (GM-RAIM + IMM + INS + CoopRAIM + Huh)
    StructuralScore      — Pillar 3 (Spectral + Structural + Phase)
    EpochDiagnosis       — Orchestrator composite output
"""

from __future__ import annotations

from dataclasses import dataclass

from gnss.cn0_detector import CN0AnomalyResult
from gnss.layers import (
    CoopRAIMResult,
    FaultEntropyResult,
    GMMResult,
    HuhSelectionResult,
    IMMResult,
    INSCouplingResult,
    OSNMALayerResult,
    PhaseTransitionResult,
    SpectralResult,
    StructuralMonitorResult,
)
from schemas import FaultClass


@dataclass(frozen=True)
class AuthenticationScore:
    """Pillar 1 — OSNMA Galileo authentication coverage score."""

    auth_fraction: float  # fraction of authenticated satellites ∈ [0, 1]
    p_spoofed: float  # 1 − auth_fraction (fusion signal)
    alert: bool  # True if auth_fraction < threshold
    osnma: OSNMALayerResult  # raw layer result


@dataclass(frozen=True)
class IntegrityScore:
    """Pillar 2 — integrity-layer base fault posterior (GM-RAIM + IMM + INS + CoopRAIM + Huh)."""

    base_posterior: tuple[float, float, float, float]  # [P_nom, P_mp, P_hw, P_spoof]
    gmm: GMMResult
    imm: IMMResult
    ins: INSCouplingResult
    coop_raim: CoopRAIMResult
    huh: HuhSelectionResult  # Layer 9 — D-optimal satellite subset


@dataclass(frozen=True)
class StructuralScore:
    """Pillar 3 — graph-structure anomaly intensity."""

    structure_intensity: float  # max(ρ_F−1, 0) + rmt_anomaly
    spectral: SpectralResult
    structural: StructuralMonitorResult
    phase: PhaseTransitionResult  # Layer 10 — Duminil-Copin percolation monitor


@dataclass(frozen=True)
class EpochDiagnosis:
    """Per-epoch diagnostic output from ResilienceTwin (4-pillar architecture)."""

    t: int
    fault_posterior: tuple[float, float, float, float]  # [P_nom, P_mp, P_hw, P_spoof]
    diagnosis: FaultClass
    confidence: float  # max(fault_posterior)
    entropy: FaultEntropyResult  # Pillar 4 — intervention
    auth: AuthenticationScore  # Pillar 1 — authentication
    integrity: IntegrityScore  # Pillar 2 — integrity
    structure: StructuralScore  # Pillar 3 — structure
    cn0_anomaly: CN0AnomalyResult | None = None  # C/N0 anomaly result; None if unavailable
