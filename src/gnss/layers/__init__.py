"""GNSS Resilience Twin — layers sub-package.

Re-exports all layer classes and result types so that consumers can use
either the flat ``gnss.resilience_twin`` namespace (backward-compatible)
or the structured ``gnss.layers.*`` namespace.
"""

from gnss.layers.authentication import OSNMALayer, OSNMALayerResult
from gnss.layers.integrity import (
    CoopRAIMLayer,
    CoopRAIMResult,
    GMMRaim,
    GMMResult,
    HuhSelectionResult,
    HuhSubsetSelector,
    IMMKalman,
    IMMResult,
    INSCouplingLayer,
    INSCouplingResult,
)
from gnss.layers.intervention import FaultEntropyMonitor, FaultEntropyResult
from gnss.layers.structure import (
    DuminilCopinPhaseMonitor,
    PhaseTransitionResult,
    SpectralMonitor,
    SpectralResult,
    StructuralDependencyMonitor,
    StructuralMonitorResult,
    _lcc_curve_batch,
)

__all__ = [
    # authentication
    "OSNMALayer",
    "OSNMALayerResult",
    # integrity
    "CoopRAIMLayer",
    "CoopRAIMResult",
    "GMMRaim",
    "GMMResult",
    "HuhSelectionResult",
    "HuhSubsetSelector",
    "IMMKalman",
    "IMMResult",
    "INSCouplingLayer",
    "INSCouplingResult",
    # intervention
    "FaultEntropyMonitor",
    "FaultEntropyResult",
    # structure
    "DuminilCopinPhaseMonitor",
    "PhaseTransitionResult",
    "SpectralMonitor",
    "SpectralResult",
    "StructuralDependencyMonitor",
    "StructuralMonitorResult",
    "_lcc_curve_batch",
]
