from pathlib import Path

from huh_twin.reporting import build_markdown_report, write_markdown_report


def sample_posterior_rows() -> list[dict[str, object]]:
    return [
        {
            "name": "growth_rate",
            "prior_mean": 0.10,
            "posterior_mean": 0.115,
            "posterior_std": 0.018,
            "observation_count": 3,
        },
        {
            "name": "discount_rate",
            "prior_mean": 0.10,
            "posterior_mean": 0.094,
            "posterior_std": 0.009,
            "observation_count": 3,
        },
    ]


def sample_valuation() -> dict[str, float]:
    return {
        "enterprise_value": 1850.0,
        "terminal_value": 2200.0,
        "discounted_terminal_value": 1360.0,
    }


def sample_sensitivity_rows() -> list[dict[str, object]]:
    return [
        {
            "variable": "growth_rate",
            "value": 0.08,
            "enterprise_value": 1600.0,
            "pct_change_vs_base": -0.05,
        },
        {
            "variable": "growth_rate",
            "value": 0.12,
            "enterprise_value": 1950.0,
            "pct_change_vs_base": 0.06,
        },
    ]


def test_build_markdown_report_contains_sections() -> None:
    content = build_markdown_report(
        title="Weekly Research Report",
        posterior_rows=sample_posterior_rows(),
        valuation=sample_valuation(),
        sensitivity_rows=sample_sensitivity_rows(),
        next_actions=["Implement reverse DCF diagnostics", "Add regime tests"],
    )

    assert "# Weekly Research Report" in content
    assert "## Posterior Summary" in content
    assert "## Valuation Summary" in content
    assert "## Sensitivity Summary" in content
    assert "## Next Actions" in content


def test_build_markdown_report_handles_empty_inputs() -> None:
    content = build_markdown_report(
        title="Empty Report",
        posterior_rows=[],
        valuation={},
        sensitivity_rows=[],
        next_actions=[],
    )

    assert "No posterior results available." in content
    assert "No valuation results available." in content
    assert "No sensitivity results available." in content
    assert "No actions recorded." in content


def test_write_markdown_report_creates_file(tmp_path: Path) -> None:
    content = build_markdown_report(
        title="File Output Report",
        posterior_rows=sample_posterior_rows(),
        valuation=sample_valuation(),
        sensitivity_rows=sample_sensitivity_rows(),
        next_actions=["Review moat assumptions"],
    )

    output_path = write_markdown_report(tmp_path / "report.md", content)

    assert output_path.exists()
    text = output_path.read_text(encoding="utf-8")
    assert "# File Output Report" in text
    assert "Review moat assumptions" in text
