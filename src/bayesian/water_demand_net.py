"""Backward-compatible re-export shim.

The implementation has moved to :mod:`bayesian.research.water_demand_net`.
This module re-exports the public API so that existing imports continue to work.
"""

from bayesian.research.water_demand_net import (
    _CPT_DEMAND_LEVEL,
    _CPT_TEMPERATURE,
    _CPT_USAGE_TIER,
    _P_DAY_TYPE,
    _P_SEASON,
    build_fukuoka_water_demand_net,
    calibrate_usage_tier_cpt,
)

__all__ = [
    "_CPT_DEMAND_LEVEL",
    "_CPT_TEMPERATURE",
    "_CPT_USAGE_TIER",
    "_P_DAY_TYPE",
    "_P_SEASON",
    "build_fukuoka_water_demand_net",
    "calibrate_usage_tier_cpt",
]
