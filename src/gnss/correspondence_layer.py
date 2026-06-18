"""Correspondence Layer — dual mapping to physical hypothesis and cryptographic trust state.

Implements the Langlands-inspired duality for GNSS fault detection:

    Physical side  (geometric / integrity):
        Coherent SNR = n·mean(Δf)² / var(Δf) → PhysicalHypothesis
        High SNR indicates common-mode meaconing (spoofing); low SNR is nominal.

    Cryptographic side (algebraic / authentication):
        OSNMA per-satellite auth flags → CryptoTrustState

Invariant:
    CorrespondenceState always carries BOTH sides.
    Neither physical alone nor crypto alone is sufficient for a decision.

Coherence check:
    is_coherent() exposes contradictions between the two sides.
    Example: physical=SPOOFED and crypto=AUTHENTICATED is incoherent —
    an attacker cannot spoof physical signals while simultaneously satisfying
    the ECDSA/TESLA authentication chain.  This contradiction is itself a
    strong spoofing indicator (AUTH_MISMATCH syndrome edge).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class PhysicalHypothesis(str, Enum):
    """Physical-domain fault hypothesis for one satellite."""

    GENUINE = "genuine"
    SPOOFED = "spoofed"
    JAMMED = "jammed"
    FAULT = "fault"
    UNCERTAIN = "uncertain"


class CryptoTrustState(str, Enum):
    """Cryptographic authentication trust level."""

    AUTHENTICATED = "authenticated"  # ECDSA/TESLA chain verified
    UNAUTHENTICATED = "unauthenticated"  # authentication check failed
    UNKNOWN = "unknown"  # authentication data unavailable
    COMPROMISED = "compromised"  # key compromise suspected


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

# Coherent SNR threshold to declare SPOOFED hypothesis.
# Coherent SNR = n · mean(Δf)² / var(Δf).
# Threshold=5 corresponds to |mean| > sqrt(5·var/n) ≈ 0.27 Hz for n=6, var=0.09.
_COHERENT_SNR_THRESHOLD: float = 5.0

# Minimum OSNMA auth fraction to declare a satellite AUTHENTICATED.
# Not used per-satellite (per-sat flag is binary), reserved for future ensemble logic.
_AUTH_FRACTION_MIN: float = 0.80


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorrespondenceState:
    """Per-satellite dual state: physical hypothesis × cryptographic trust state.

    Invariant: Both `physical` and `crypto` must be explicitly assessed.
    The default PhysicalHypothesis.UNCERTAIN and CryptoTrustState.UNKNOWN are valid
    states — they represent genuinely uncertain assessments, not missing data.
    """

    satellite_id: str
    epoch: int
    physical: PhysicalHypothesis
    physical_score: float  # anomaly evidence ∈ [0, 1]; higher = more anomalous
    crypto: CryptoTrustState
    crypto_confidence: float  # auth confidence ∈ [0, 1]

    def is_coherent(self) -> bool:
        """True when physical and crypto assessments are mutually consistent.

        Incoherent cases:
            physical=SPOOFED  + crypto=AUTHENTICATED → impossible without key compromise
            physical=GENUINE  + crypto=COMPROMISED   → impossible without signal substitution

        All other combinations are coherent or unknown.
        """
        if self.physical == PhysicalHypothesis.SPOOFED:
            # Spoofing while passing ECDSA/TESLA authentication is physically impossible.
            return self.crypto != CryptoTrustState.AUTHENTICATED
        if self.physical == PhysicalHypothesis.GENUINE:
            # Genuine signal from a key-compromised chain is a contradiction.
            return self.crypto != CryptoTrustState.COMPROMISED
        # JAMMED / FAULT / UNCERTAIN are coherent with all crypto states.
        return True


@dataclass(frozen=True)
class EpochCorrespondence:
    """All per-satellite correspondence states for one epoch.

    Attributes:
        epoch:               Epoch index.
        states:              Per-satellite CorrespondenceState tuple.
        auth_fraction:       Fraction of satellites with AUTHENTICATED crypto state.
        mean_physical_score: Mean anomaly score across all satellites ∈ [0, 1].
        incoherence_count:   Number of satellites with incoherent state pairs.
    """

    epoch: int
    states: tuple[CorrespondenceState, ...]
    auth_fraction: float
    mean_physical_score: float
    incoherence_count: int


# ---------------------------------------------------------------------------
# Assessor
# ---------------------------------------------------------------------------


class CorrespondenceAssessor:
    """Assess physical and cryptographic state for an epoch.

    Physical score is derived from Doppler coherence SNR:
        coherent_snr = n · mean(Δf)² / var(Δf)
        physical_score = clip(coherent_snr / (2 · threshold), 0, 1)

    Per-satellite physical hypothesis:
        physical_score > 0.5 → SPOOFED (common-mode meaconing bias detected)
        else                 → GENUINE

    Crypto state comes from per-satellite OSNMA authentication flags:
        flag=True  → AUTHENTICATED, confidence=1.0
        flag=False → UNAUTHENTICATED, confidence=0.0
        flag=None  → UNKNOWN, confidence=0.5
    """

    def assess_epoch(
        self,
        epoch: int,
        satellite_ids: list[str],
        doppler_deviations: np.ndarray,  # shape (n,)
        osnma_auth_per_sat: list[bool] | None = None,
    ) -> EpochCorrespondence:
        """Compute dual correspondence state for all satellites in one epoch.

        Args:
            epoch:               Epoch index.
            satellite_ids:       Ordered list of n satellite identifiers.
            doppler_deviations:  (n,) Doppler residuals [Hz].
            osnma_auth_per_sat:  Per-satellite OSNMA auth flags; None if unavailable.
        """
        n = len(satellite_ids)
        if len(doppler_deviations) != n:
            raise ValueError(f"doppler_deviations length {len(doppler_deviations)} != n_sats {n}")

        # Physical: coherent SNR across all satellites.
        # coherent_snr ≫ 1 → common-mode meaconing bias (spoofing).
        mean_dev = float(doppler_deviations.mean())
        var_dev = max(float(doppler_deviations.var()), 1e-9)
        coherent_snr = n * mean_dev**2 / var_dev
        # Normalize: score=0.5 at threshold, score=1.0 at 2×threshold.
        physical_score = min(coherent_snr / (2.0 * _COHERENT_SNR_THRESHOLD), 1.0)
        physical_hyp = (
            PhysicalHypothesis.SPOOFED if physical_score > 0.5 else PhysicalHypothesis.GENUINE
        )

        # Crypto: per-satellite auth flags.
        auth_flags: list[bool | None] = (
            list(osnma_auth_per_sat) if osnma_auth_per_sat is not None else [None] * n
        )

        states: list[CorrespondenceState] = []
        n_auth = 0
        for i, sat_id in enumerate(satellite_ids):
            flag: bool | None = auth_flags[i] if i < len(auth_flags) else None
            if flag is None:
                crypto = CryptoTrustState.UNKNOWN
                crypto_conf = 0.5
            elif flag:
                crypto = CryptoTrustState.AUTHENTICATED
                crypto_conf = 1.0
                n_auth += 1
            else:
                crypto = CryptoTrustState.UNAUTHENTICATED
                crypto_conf = 0.0

            states.append(
                CorrespondenceState(
                    satellite_id=sat_id,
                    epoch=epoch,
                    physical=physical_hyp,
                    physical_score=physical_score,
                    crypto=crypto,
                    crypto_confidence=crypto_conf,
                )
            )

        auth_fraction = n_auth / n if n > 0 else 0.0
        incoherence_count = sum(1 for s in states if not s.is_coherent())

        return EpochCorrespondence(
            epoch=epoch,
            states=tuple(states),
            auth_fraction=auth_fraction,
            mean_physical_score=physical_score,
            incoherence_count=incoherence_count,
        )
