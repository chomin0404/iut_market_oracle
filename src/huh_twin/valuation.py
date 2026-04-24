"""Public valuation API for the huh_twin package.

Re-exports the DCF primitives from valuation.dcf so that callers can import
from a single, stable namespace::

    from huh_twin.valuation import DCFInputs, dcf_valuation, ...
"""

from valuation.dcf import (
    DCFInputs,
    DCFResult,
    dcf_valuation,
    discount_cash_flows,
    gordon_terminal_value,
    project_fcfs,
    reverse_dcf_implied_growth,
)

__all__ = [
    "DCFInputs",
    "DCFResult",
    "dcf_valuation",
    "discount_cash_flows",
    "gordon_terminal_value",
    "project_fcfs",
    "reverse_dcf_implied_growth",
]
