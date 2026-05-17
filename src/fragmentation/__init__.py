"""Growth-fragmentation drone swarm control module.

Public API
----------
FragConfig          — simulation configuration (Pydantic)
FragResult          — full simulation output (Pydantic)
EigenResult         — Malthus parameter + eigenfunction (Pydantic)
ParticleSnapshot    — single Gillespie event snapshot (Pydantic)
estimate_malthus    — GFE spectral analysis
simulate            — Marcus-Lushnikov Gillespie simulation
run_pipeline        — convenience: eigen → simulate → return FragResult

References
----------
Bertoin J. (2006). Random Fragmentation and Coagulation Processes. Cambridge UP.
Norris J.R. (1999). Smoluchowski coagulation equation. Ann. Appl. Probab.
Villani C. (2008). Optimal Transport: Old and New. Springer.
"""

from fragmentation.eigenanalysis import estimate_malthus
from fragmentation.marcus_lushnikov import simulate
from fragmentation.schemas import EigenResult, FragConfig, FragResult, ParticleSnapshot


def run_pipeline(config: FragConfig, control_u: float = 0.0) -> FragResult:
    """End-to-end pipeline: eigenanalysis → Gillespie simulation.

    Parameters
    ----------
    config : FragConfig
        Simulation configuration.
    control_u : float
        Control input u ≥ 0 (uniform, event-driven).

    Returns
    -------
    FragResult
        Full result with Malthus parameter, trajectory, and W₂ cost.
    """
    eigen = estimate_malthus(config)
    return simulate(config, eigen, control_u=control_u)


__all__ = [
    "FragConfig",
    "FragResult",
    "EigenResult",
    "ParticleSnapshot",
    "estimate_malthus",
    "simulate",
    "run_pipeline",
]
