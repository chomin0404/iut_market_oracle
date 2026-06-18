"""Signal Layer — raw observation ingestion and directed sparse feature extraction.

Converts per-satellite signal measurements into a uniform feature vector:
    [doppler_hz, code_phase_chips, carrier_phase_cycles, cn0_db_hz, aoa_deg, iq_phase_rad]

Missing values are encoded as NaN and tracked via quality_mask.
The feature vector dimension is fixed at FEATURE_DIM = 6.

Invariant: feature_vector.shape == (FEATURE_DIM,) for all outputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURE_DIM: int = 6
FEATURE_NAMES: tuple[str, ...] = (
    "doppler_hz",
    "code_phase_chips",
    "carrier_phase_cycles",
    "cn0_db_hz",
    "aoa_deg",
    "iq_phase_rad",
)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SignalObservation:
    """Raw per-satellite signal observation for one epoch.

    Attributes:
        satellite_id:          Satellite PRN identifier (e.g. "G01").
        epoch:                 Epoch index (monotonically increasing).
        gps_tow:               GPS Time of Week [s].
        doppler_hz:            Doppler frequency deviation [Hz]. Always required.
        code_phase_chips:      Receiver code phase [chips]. Optional.
        carrier_phase_cycles:  Integrated carrier phase [cycles]. Optional.
        cn0_db_hz:             Carrier-to-noise density ratio [dB-Hz]. Optional.
        aoa_deg:               Angle-of-arrival measurement [degrees]. Optional.
        iq_phase_rad:          IQ discriminator phase [radians]. Optional.
        elevation_deg:         Satellite elevation angle [degrees]. Optional.
        pr_residual_m:         Pseudorange residual [m]. Optional.
    """

    satellite_id: str
    epoch: int
    gps_tow: float  # GPS Time of Week [s]
    doppler_hz: float
    code_phase_chips: float | None = None
    carrier_phase_cycles: float | None = None
    cn0_db_hz: float | None = None
    aoa_deg: float | None = None
    iq_phase_rad: float | None = None
    elevation_deg: float | None = None
    pr_residual_m: float | None = None


@dataclass(frozen=True)
class SignalFeature:
    """Directed sparse feature derived from a SignalObservation.

    Attributes:
        satellite_id:   Satellite PRN identifier.
        epoch:          Epoch index.
        gps_tow:        GPS Time of Week [s].
        feature_vector: shape (FEATURE_DIM,), dtype float64.
                        NaN where the measurement is unavailable.
        quality_mask:   shape (FEATURE_DIM,), dtype bool.
                        True where the feature is valid (not NaN).
    """

    satellite_id: str
    epoch: int
    gps_tow: float
    feature_vector: np.ndarray  # shape (FEATURE_DIM,)
    quality_mask: np.ndarray  # shape (FEATURE_DIM,), bool


# ---------------------------------------------------------------------------
# Feature extractor
# ---------------------------------------------------------------------------


class SignalFeatureExtractor:
    """Convert SignalObservation → SignalFeature.

    Packs the six signal components into a fixed-length vector.
    Optional fields become NaN; quality_mask tracks availability.
    """

    def extract(self, obs: SignalObservation) -> SignalFeature:
        """Extract feature vector from one observation."""

        def _f(v: float | None) -> float:
            return float(v) if v is not None else np.nan

        raw = np.array(
            [
                obs.doppler_hz,
                _f(obs.code_phase_chips),
                _f(obs.carrier_phase_cycles),
                _f(obs.cn0_db_hz),
                _f(obs.aoa_deg),
                _f(obs.iq_phase_rad),
            ],
            dtype=float,
        )
        mask = ~np.isnan(raw)
        return SignalFeature(
            satellite_id=obs.satellite_id,
            epoch=obs.epoch,
            gps_tow=obs.gps_tow,
            feature_vector=raw,
            quality_mask=mask,
        )

    def extract_batch(self, observations: list[SignalObservation]) -> list[SignalFeature]:
        """Extract features from a list of observations (one per satellite)."""
        return [self.extract(obs) for obs in observations]
