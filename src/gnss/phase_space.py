"""Backward-compatible re-export stub for gnss.phase_space.

Implementation lives in gnss.experimental.phase_space.
"""

from gnss.experimental.phase_space import (  # noqa: F401
    EMBED_DIM,
    LYA_ALARM_THRESH,
    MAX_ITER,
    MIN_POINTS,
    THEILER_WINDOW,
    TIME_DELAY,
    PhaseSpaceResult,
    max_lyapunov_exponent,
    takens_embed,
)

__all__ = [
    "EMBED_DIM",
    "LYA_ALARM_THRESH",
    "MAX_ITER",
    "MIN_POINTS",
    "THEILER_WINDOW",
    "TIME_DELAY",
    "PhaseSpaceResult",
    "max_lyapunov_exponent",
    "takens_embed",
]
