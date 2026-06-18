"""Correlation peak monitor — re-export stub.

Implementation has moved to gnss.layers.correlation_monitor.
This module re-exports all public names for backward compatibility.
"""

from gnss.layers.correlation_monitor import (  # noqa: F401
    GLRT_THRESH,
    GLRT_WINDOW,
    EL_RMS_THRESH,
    CorrelationMonitor,
    CorrelationMonitorResult,
)

__all__ = [
    "EL_RMS_THRESH",
    "GLRT_THRESH",
    "GLRT_WINDOW",
    "CorrelationMonitor",
    "CorrelationMonitorResult",
]
