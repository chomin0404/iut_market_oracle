"""Tests for src/huh_twin/report.py.

Structural invariants
---------------------
markdown_table
    - Header row contains every column name.
    - Separator row has exactly len(headers) `---` cells.
    - Data rows match the supplied content verbatim.
    - Empty rows list → header + separator only (no data rows).

render_* functions
    - Empty input → fallback paragraph containing "No … available."
    - Non-empty input → section heading present + table structure present.
    - Numeric values formatted to declared precision.

build_markdown_report
    - Title rendered as H1.
    - All four section headings present.
    - Ends with a single newline.

write_markdown_report
    - Written file content matches the supplied string exactly.
    - Returns the resolved Path.
    - Parent directories are created automatically.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from huh_twin.report import (
    _format_float,
    build_markdown_report,
    markdown_table,
    render_next_actions,
    render_posterior_section,
    render_sensitivity_section,
    render_valuation_section,
    write_markdown_report,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

POSTERIOR_ROW = {
    "name": "growth_rate",
    "prior_mean": 0.10,
    "posterior_mean": 0.12,
    "posterior_std": 0.02,
    "observation_count": 3,
}

VALUATION = {
    "enterprise_value": 1234.5678,
    "terminal_value": 987.6543,
    "discounted_terminal_value": 800.1111,
}

SENSITIVITY_ROW_ONE_WAY = {
    "variable": "growth_rate",
    "value": 0.15,
    "enterprise_value": 1500.0,
    "pct_change_vs_base": 0.05,
}

SENSITIVITY_ROW_TWO_WAY = {
    "x_name": "growth_rate",
    "x_value": 0.12,
    "enterprise_value": 1400.0,
    "pct_change_vs_base": -0.02,
}


# ---------------------------------------------------------------------------
# _format_float
# ---------------------------------------------------------------------------


class TestFormatFloat:
    def test_default_four_decimal_places(self) -> None:
        assert _format_float(1.23456789) == "1.2346"

    def test_custom_digits(self) -> None:
        assert _format_float(1.23456789, digits=2) == "1.23"

    def test_zero_digits(self) -> None:
        assert _format_float(3.7, digits=0) == "4"

    def test_negative_value(self) -> None:
        assert _format_float(-0.05, digits=4) == "-0.0500"

    def test_zero(self) -> None:
        assert _format_float(0.0) == "0.0000"


# ---------------------------------------------------------------------------
# markdown_table
# ---------------------------------------------------------------------------


class TestMarkdownTable:
    def test_header_contains_all_columns(self) -> None:
        table = markdown_table(["A", "B", "C"], [])
        first_line = table.splitlines()[0]
        assert "A" in first_line and "B" in first_line and "C" in first_line

    def test_separator_has_correct_cell_count(self) -> None:
        table = markdown_table(["X", "Y"], [["1", "2"]])
        sep_line = table.splitlines()[1]
        # Count `---` occurrences
        assert sep_line.count("---") == 2

    def test_data_row_content(self) -> None:
        table = markdown_table(["Name", "Value"], [["alpha", "1.0000"]])
        lines = table.splitlines()
        assert "alpha" in lines[2]
        assert "1.0000" in lines[2]

    def test_multiple_rows(self) -> None:
        rows = [["r1c1", "r1c2"], ["r2c1", "r2c2"]]
        table = markdown_table(["H1", "H2"], rows)
        lines = table.splitlines()
        # header + separator + 2 data rows = 4 lines
        assert len(lines) == 4

    def test_empty_rows_gives_two_lines(self) -> None:
        table = markdown_table(["A", "B"], [])
        assert len(table.splitlines()) == 2

    def test_pipe_delimiters_present(self) -> None:
        table = markdown_table(["Col"], [["val"]])
        for line in table.splitlines():
            assert line.startswith("|") and line.endswith("|")

    def test_single_column(self) -> None:
        table = markdown_table(["Only"], [["one"]])
        lines = table.splitlines()
        assert "Only" in lines[0]
        assert "one" in lines[2]


# ---------------------------------------------------------------------------
# render_posterior_section
# ---------------------------------------------------------------------------


class TestRenderPosteriorSection:
    def test_empty_returns_fallback(self) -> None:
        out = render_posterior_section([])
        assert "No posterior results available." in out

    def test_heading_present(self) -> None:
        out = render_posterior_section([POSTERIOR_ROW])
        assert "## Posterior Summary" in out

    def test_name_appears(self) -> None:
        out = render_posterior_section([POSTERIOR_ROW])
        assert "growth_rate" in out

    def test_prior_mean_formatted(self) -> None:
        out = render_posterior_section([POSTERIOR_ROW])
        assert "0.1000" in out

    def test_posterior_mean_formatted(self) -> None:
        out = render_posterior_section([POSTERIOR_ROW])
        assert "0.1200" in out

    def test_observation_count_appears(self) -> None:
        out = render_posterior_section([POSTERIOR_ROW])
        assert "3" in out

    def test_multiple_rows(self) -> None:
        second = {**POSTERIOR_ROW, "name": "discount_rate"}
        out = render_posterior_section([POSTERIOR_ROW, second])
        assert "growth_rate" in out
        assert "discount_rate" in out

    def test_ends_with_newline(self) -> None:
        assert render_posterior_section([POSTERIOR_ROW]).endswith("\n")


# ---------------------------------------------------------------------------
# render_valuation_section
# ---------------------------------------------------------------------------


class TestRenderValuationSection:
    def test_empty_returns_fallback(self) -> None:
        out = render_valuation_section({})
        assert "No valuation results available." in out

    def test_heading_present(self) -> None:
        out = render_valuation_section(VALUATION)
        assert "## Valuation Summary" in out

    def test_enterprise_value_formatted(self) -> None:
        out = render_valuation_section(VALUATION)
        assert "1234.5678" in out

    def test_terminal_value_formatted(self) -> None:
        out = render_valuation_section(VALUATION)
        assert "987.6543" in out

    def test_discounted_terminal_value_formatted(self) -> None:
        out = render_valuation_section(VALUATION)
        assert "800.1111" in out

    def test_all_three_metric_labels_present(self) -> None:
        out = render_valuation_section(VALUATION)
        assert "Enterprise Value" in out
        assert "Terminal Value" in out
        assert "Discounted Terminal Value" in out

    def test_ends_with_newline(self) -> None:
        assert render_valuation_section(VALUATION).endswith("\n")


# ---------------------------------------------------------------------------
# render_sensitivity_section
# ---------------------------------------------------------------------------


class TestRenderSensitivitySection:
    def test_empty_returns_fallback(self) -> None:
        out = render_sensitivity_section([])
        assert "No sensitivity results available." in out

    def test_heading_present(self) -> None:
        out = render_sensitivity_section([SENSITIVITY_ROW_ONE_WAY])
        assert "## Sensitivity Summary" in out

    def test_one_way_variable_name(self) -> None:
        out = render_sensitivity_section([SENSITIVITY_ROW_ONE_WAY])
        assert "growth_rate" in out

    def test_one_way_value_formatted(self) -> None:
        out = render_sensitivity_section([SENSITIVITY_ROW_ONE_WAY])
        assert "0.1500" in out

    def test_one_way_pct_change_formatted_with_percent(self) -> None:
        # 0.05 * 100 = 5.00%
        out = render_sensitivity_section([SENSITIVITY_ROW_ONE_WAY])
        assert "5.00%" in out

    def test_two_way_falls_back_to_x_name(self) -> None:
        out = render_sensitivity_section([SENSITIVITY_ROW_TWO_WAY])
        assert "growth_rate" in out

    def test_two_way_falls_back_to_x_value(self) -> None:
        out = render_sensitivity_section([SENSITIVITY_ROW_TWO_WAY])
        assert "0.1200" in out

    def test_negative_pct_change_formatted(self) -> None:
        # -0.02 * 100 = -2.00%
        out = render_sensitivity_section([SENSITIVITY_ROW_TWO_WAY])
        assert "-2.00%" in out

    def test_multiple_rows(self) -> None:
        out = render_sensitivity_section([SENSITIVITY_ROW_ONE_WAY, SENSITIVITY_ROW_TWO_WAY])
        assert out.count("|") > 4  # more than a 2-row table

    def test_ends_with_newline(self) -> None:
        assert render_sensitivity_section([SENSITIVITY_ROW_ONE_WAY]).endswith("\n")


# ---------------------------------------------------------------------------
# render_next_actions
# ---------------------------------------------------------------------------


class TestRenderNextActions:
    def test_empty_returns_fallback(self) -> None:
        out = render_next_actions([])
        assert "No actions recorded." in out

    def test_heading_present(self) -> None:
        out = render_next_actions(["Do X"])
        assert "## Next Actions" in out

    def test_action_rendered_as_bullet(self) -> None:
        out = render_next_actions(["Do X"])
        assert "- Do X" in out

    def test_multiple_actions_all_present(self) -> None:
        out = render_next_actions(["Step 1", "Step 2", "Step 3"])
        assert "- Step 1" in out
        assert "- Step 2" in out
        assert "- Step 3" in out

    def test_ends_with_newline(self) -> None:
        assert render_next_actions(["Do X"]).endswith("\n")


# ---------------------------------------------------------------------------
# build_markdown_report
# ---------------------------------------------------------------------------


class TestBuildMarkdownReport:
    def _build(self, **overrides) -> str:
        defaults: dict = dict(
            title="Test Report",
            posterior_rows=[POSTERIOR_ROW],
            valuation=VALUATION,
            sensitivity_rows=[SENSITIVITY_ROW_ONE_WAY],
            next_actions=["Review results"],
        )
        defaults.update(overrides)
        return build_markdown_report(**defaults)

    def test_title_rendered_as_h1(self) -> None:
        out = self._build(title="My Report")
        assert "# My Report" in out

    def test_all_section_headings_present(self) -> None:
        out = self._build()
        assert "## Posterior Summary" in out
        assert "## Valuation Summary" in out
        assert "## Sensitivity Summary" in out
        assert "## Next Actions" in out

    def test_ends_with_single_newline(self) -> None:
        out = self._build()
        assert out.endswith("\n")
        assert not out.endswith("\n\n")

    def test_empty_sections_show_fallbacks(self) -> None:
        out = self._build(
            posterior_rows=[],
            valuation={},
            sensitivity_rows=[],
            next_actions=[],
        )
        assert "No posterior results available." in out
        assert "No valuation results available." in out
        assert "No sensitivity results available." in out
        assert "No actions recorded." in out

    def test_title_appears_before_sections(self) -> None:
        out = self._build(title="First")
        title_pos = out.index("# First")
        posterior_pos = out.index("## Posterior")
        assert title_pos < posterior_pos

    def test_section_order(self) -> None:
        out = self._build()
        positions = {
            heading: out.index(heading)
            for heading in [
                "## Posterior Summary",
                "## Valuation Summary",
                "## Sensitivity Summary",
                "## Next Actions",
            ]
        }
        ordered = sorted(positions, key=lambda k: positions[k])
        assert ordered == [
            "## Posterior Summary",
            "## Valuation Summary",
            "## Sensitivity Summary",
            "## Next Actions",
        ]


# ---------------------------------------------------------------------------
# write_markdown_report
# ---------------------------------------------------------------------------


class TestWriteMarkdownReport:
    def test_returns_path(self, tmp_path: Path) -> None:
        p = write_markdown_report(tmp_path / "report.md", "# Hello\n")
        assert isinstance(p, Path)

    def test_file_content_matches(self, tmp_path: Path) -> None:
        content = "# Title\n\nBody text.\n"
        p = write_markdown_report(tmp_path / "report.md", content)
        assert p.read_text(encoding="utf-8") == content

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b" / "c" / "report.md"
        write_markdown_report(nested, "x\n")
        assert nested.exists()

    def test_returned_path_resolves_to_written_file(self, tmp_path: Path) -> None:
        dest = tmp_path / "out.md"
        returned = write_markdown_report(dest, "y\n")
        assert returned.resolve() == dest.resolve()

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        dest = str(tmp_path / "str_path.md")
        write_markdown_report(dest, "z\n")
        assert Path(dest).exists()

    def test_overwrites_existing_file(self, tmp_path: Path) -> None:
        dest = tmp_path / "report.md"
        write_markdown_report(dest, "first\n")
        write_markdown_report(dest, "second\n")
        assert dest.read_text(encoding="utf-8") == "second\n"
