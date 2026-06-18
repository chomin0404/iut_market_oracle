"""Multi-sensor fusion for GNSS integrity — re-export stub.

Implementation has moved to gnss.layers.sensor_fusion.
This module re-exports all public names for backward compatibility.
"""

from gnss.layers.sensor_fusion import (  # noqa: F401
    BARO_CHI2_THRESH,
    VO_CHI2_THRESH_2D,
    BarometerResult,
    FixedLagSmoother,
    SensorFusionLayer,
    SensorFusionResult,
    VisualOdometryResult,
    check_barometer,
    check_visual_odometry,
)

__all__ = [
    "BARO_CHI2_THRESH",
    "VO_CHI2_THRESH_2D",
    "BarometerResult",
    "FixedLagSmoother",
    "SensorFusionLayer",
    "SensorFusionResult",
    "VisualOdometryResult",
    "check_barometer",
    "check_visual_odometry",
]
