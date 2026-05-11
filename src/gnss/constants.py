"""Shared GNSS signal and physical constants (T1300 / T1500).

These values are used by both the simulation layer (spoof_sim, multi_sensor_sim)
and the production scoring layer (resilience_twin, mvp).  Defining them here
prevents the production scoring code from depending on simulation modules.
"""

# ---------------------------------------------------------------------------
# RF / carrier constants
# ---------------------------------------------------------------------------

_SPEED_OF_LIGHT: float = 2.998e8  # m/s
_L1_FREQ: float = 1575.42e6  # Hz  (GPS L1 carrier)

# ---------------------------------------------------------------------------
# Doppler / signal noise
# ---------------------------------------------------------------------------

_DOPPLER_NOISE_STD: float = 0.30  # Hz — genuine Doppler measurement noise 1-σ
_GRAPH_SIGMA: float = 1.50  # Hz — Gaussian kernel bandwidth σ for similarity graph

# ---------------------------------------------------------------------------
# INS coupling noise
# ---------------------------------------------------------------------------

_INS_VEL_STD: float = 0.05  # m/s — INS velocity error 1-σ
_INS_CLOCK_STD: float = 0.01  # m/s equivalent — INS clock error 1-σ

# ---------------------------------------------------------------------------
# Attack-window Dirichlet prior
# ---------------------------------------------------------------------------

_DIRICHLET_ALPHA: float = 2.0  # symmetric Dirichlet concentration parameter
