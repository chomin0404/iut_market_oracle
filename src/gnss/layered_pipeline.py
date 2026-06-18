"""Layered Pipeline — orchestrator for the 5-layer GNSS defense architecture.

Pipeline order (one call to process_epoch() runs all five layers in sequence):

    1. Signal Layer      → SignalFeature per satellite
    2. Correspondence    → EpochCorrespondence (dual physical + crypto state)
    3. Syndrome Graph    → SyndromeEdge list (violated consistency checks, append-only)
    4. Decoder           → DecoderResult (MAP fault class via log-linear belief propagation)
    5. Certificate       → AlarmCertificate (machine-readable invariant proof)

State
-----
The pipeline is stateful across epochs:
    - SyndromeGraph accumulates all edges over the run (append-only).
    - process_epoch() must be called with monotonically increasing epoch indices.

Usage::

    pipeline = LayeredPipeline(satellite_ids=["G01", "G02", ..., "G08"])
    result = pipeline.process_epoch(
        epoch=0,
        gps_tow=518400.0,
        doppler_hz=np.zeros(8),
    )
    assert result.certificate.is_valid()
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import numpy as np

from gnss.certificate import AlarmCertificate, build_certificate
from gnss.correspondence_layer import CorrespondenceAssessor, EpochCorrespondence
from gnss.decoder import DecoderResult, FactorGraphDecoder
from gnss.signal_layer import SignalFeature, SignalFeatureExtractor, SignalObservation
from gnss.syndrome_graph import SyndromeGraph

# ---------------------------------------------------------------------------
# Default RAIM threshold: chi²(0.95, df=4) = 9.488
# Typical usage: df = n_sats − 4 (four state unknowns: 3D position + clock).
# The caller supplies the actual chi² test statistic from their RAIM implementation.
# ---------------------------------------------------------------------------
_RAIM_THRESHOLD_DEFAULT: float = 9.488  # chi²(0.95, df=4)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LayeredEpochResult:
    """Full 5-layer pipeline output for one epoch.

    Attributes:
        epoch:              Epoch index.
        gps_tow:            GPS Time of Week [s].
        signal_features:    Per-satellite SignalFeature tuple (Layer 1).
        correspondence:     EpochCorrespondence (Layer 2).
        n_syndrome_edges:   Number of violated constraints added this epoch (Layer 3).
        decoder:            DecoderResult from log-linear BP (Layer 4).
        certificate:        AlarmCertificate (Layer 5).
    """

    epoch: int
    gps_tow: float
    signal_features: tuple[SignalFeature, ...]
    correspondence: EpochCorrespondence
    n_syndrome_edges: int
    decoder: DecoderResult
    certificate: AlarmCertificate


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class LayeredPipeline:
    """5-layer GNSS defense pipeline.

    Stateful across epochs: the internal SyndromeGraph grows monotonically.
    Call process_epoch() once per observation epoch in increasing epoch order.

    Args:
        satellite_ids:    Ordered list of satellite PRN identifiers.
        run_id:           Optional 8-char hex run identifier.
                          Generated automatically if not provided.
        raim_threshold:   Chi² decision boundary for the RAIM geometry check.
                          Default: chi²(0.95, df=4) = 9.488.
    """

    def __init__(
        self,
        satellite_ids: list[str],
        run_id: str | None = None,
        raim_threshold: float = _RAIM_THRESHOLD_DEFAULT,
    ) -> None:
        self._satellite_ids = list(satellite_ids)
        self._run_id = run_id or uuid.uuid4().hex[:8]
        self._raim_threshold = raim_threshold
        self._extractor = SignalFeatureExtractor()
        self._correspondence = CorrespondenceAssessor()
        self._syndrome_graph = SyndromeGraph()
        self._decoder = FactorGraphDecoder()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def syndrome_graph(self) -> SyndromeGraph:
        """Access to the underlying syndrome graph for inspection."""
        return self._syndrome_graph

    @property
    def n_satellites(self) -> int:
        return len(self._satellite_ids)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process_epoch(
        self,
        epoch: int,
        gps_tow: float,
        doppler_hz: np.ndarray,
        elevations_rad: np.ndarray | None = None,
        osnma_auth_per_sat: list[bool] | None = None,
        cn0_db_hz: np.ndarray | None = None,
        carrier_phases: np.ndarray | None = None,
        prev_carrier_phases: np.ndarray | None = None,
        raim_chi2: float = 0.0,
    ) -> LayeredEpochResult:
        """Run all five layers for one observation epoch.

        Args:
            epoch:               Epoch index (must be new; raises on duplicate).
            gps_tow:             GPS Time of Week [s].
            doppler_hz:          (n,) Doppler deviations [Hz].  Required.
            elevations_rad:      (n,) elevation angles [rad].  Optional.
            osnma_auth_per_sat:  Per-satellite OSNMA auth flags.  Optional.
            cn0_db_hz:           (n,) C/N₀ values [dB-Hz].  Optional.
            carrier_phases:      (n,) carrier phase [cycles].  Optional.
            prev_carrier_phases: (n,) previous-epoch carrier phases.  Optional.
            raim_chi2:           RAIM chi² test statistic (default 0 = no RAIM check).

        Returns:
            LayeredEpochResult with all five layers' outputs.

        Raises:
            ValueError: If epoch has already been processed (append-only syndrome graph).
            ValueError: If doppler_hz length does not match n_satellites.
        """
        n = self.n_satellites
        if len(doppler_hz) != n:
            raise ValueError(f"doppler_hz length {len(doppler_hz)} != n_satellites {n}")

        # ---- Layer 1: Signal --------------------------------------------
        observations = [
            SignalObservation(
                satellite_id=self._satellite_ids[i],
                epoch=epoch,
                gps_tow=gps_tow,
                doppler_hz=float(doppler_hz[i]),
                cn0_db_hz=float(cn0_db_hz[i]) if cn0_db_hz is not None else None,
                carrier_phase_cycles=(
                    float(carrier_phases[i]) if carrier_phases is not None else None
                ),
                elevation_deg=(
                    float(np.degrees(elevations_rad[i])) if elevations_rad is not None else None
                ),
            )
            for i in range(n)
        ]
        signal_features = self._extractor.extract_batch(observations)

        # ---- Layer 2: Correspondence ------------------------------------
        correspondence = self._correspondence.assess_epoch(
            epoch=epoch,
            satellite_ids=self._satellite_ids,
            doppler_deviations=doppler_hz,
            osnma_auth_per_sat=osnma_auth_per_sat,
        )

        # ---- Layer 3: Syndrome Graph ------------------------------------
        syndrome_edges = self._syndrome_graph.add_epoch(
            epoch=epoch,
            satellite_ids=self._satellite_ids,
            correspondences=list(correspondence.states),
            doppler_deviations=doppler_hz,
            raim_chi2=raim_chi2,
            raim_threshold=self._raim_threshold,
            cn0_values=cn0_db_hz,
            carrier_phases=carrier_phases,
            prev_carrier_phases=prev_carrier_phases,
        )

        # ---- Layer 4: Decoder -------------------------------------------
        decoder_result = self._decoder.decode(
            satellite_ids=self._satellite_ids,
            syndrome_edges=syndrome_edges,
            elevations_rad=elevations_rad,
        )

        # ---- Layer 5: Certificate ---------------------------------------
        certificate = build_certificate(
            epoch=epoch,
            gps_tow=gps_tow,
            run_id=self._run_id,
            decoder_result=decoder_result,
            syndrome_edges=syndrome_edges,
            correspondence_incoherence_count=correspondence.incoherence_count,
        )

        return LayeredEpochResult(
            epoch=epoch,
            gps_tow=gps_tow,
            signal_features=tuple(signal_features),
            correspondence=correspondence,
            n_syndrome_edges=len(syndrome_edges),
            decoder=decoder_result,
            certificate=certificate,
        )
