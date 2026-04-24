"""Public reporting API for the huh_twin package.

Re-exports from huh_twin.report so callers can import from a single namespace::

    from huh_twin.reporting import build_markdown_report, write_markdown_report
"""

from huh_twin.report import (
    build_markdown_report,
    markdown_table,
    render_next_actions,
    render_posterior_section,
    render_sensitivity_section,
    render_valuation_section,
    write_markdown_report,
)

__all__ = [
    "build_markdown_report",
    "markdown_table",
    "render_next_actions",
    "render_posterior_section",
    "render_sensitivity_section",
    "render_valuation_section",
    "write_markdown_report",
]
