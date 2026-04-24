from __future__ import annotations

from pathlib import Path
from typing import Any


def _format_float(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    row_lines = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, separator_line, *row_lines])


def render_posterior_section(posterior_rows: list[dict[str, Any]]) -> str:
    if not posterior_rows:
        return "## Posterior Summary\n\nNo posterior results available.\n"

    rows = []
    for row in posterior_rows:
        rows.append(
            [
                str(row["name"]),
                _format_float(float(row["prior_mean"])),
                _format_float(float(row["posterior_mean"])),
                _format_float(float(row["posterior_std"])),
                str(row["observation_count"]),
            ]
        )

    table = markdown_table(
        headers=["Name", "Prior Mean", "Posterior Mean", "Posterior Std", "Obs Count"],
        rows=rows,
    )
    return f"## Posterior Summary\n\n{table}\n"


def render_valuation_section(valuation: dict[str, Any]) -> str:
    if not valuation:
        return "## Valuation Summary\n\nNo valuation results available.\n"

    rows = [
        ["Enterprise Value", _format_float(float(valuation["enterprise_value"]))],
        ["Terminal Value", _format_float(float(valuation["terminal_value"]))],
        ["Discounted Terminal Value", _format_float(float(valuation["discounted_terminal_value"]))],
    ]
    table = markdown_table(headers=["Metric", "Value"], rows=rows)
    return f"## Valuation Summary\n\n{table}\n"


def render_sensitivity_section(sensitivity_rows: list[dict[str, Any]]) -> str:
    if not sensitivity_rows:
        return "## Sensitivity Summary\n\nNo sensitivity results available.\n"

    rows = []
    for row in sensitivity_rows:
        rows.append(
            [
                str(row.get("variable", row.get("x_name", ""))),
                _format_float(float(row.get("value", row.get("x_value", 0.0)))),
                _format_float(float(row["enterprise_value"])),
                _format_float(float(row["pct_change_vs_base"]) * 100.0, digits=2) + "%",
            ]
        )

    table = markdown_table(
        headers=["Variable", "Value", "Enterprise Value", "Delta vs Base"],
        rows=rows,
    )
    return f"## Sensitivity Summary\n\n{table}\n"


def render_next_actions(next_actions: list[str]) -> str:
    if not next_actions:
        return "## Next Actions\n\n- No actions recorded.\n"

    lines = "\n".join(f"- {action}" for action in next_actions)
    return f"## Next Actions\n\n{lines}\n"


def build_markdown_report(
    title: str,
    posterior_rows: list[dict[str, Any]],
    valuation: dict[str, Any],
    sensitivity_rows: list[dict[str, Any]],
    next_actions: list[str],
) -> str:
    parts = [
        f"# {title}\n",
        render_posterior_section(posterior_rows),
        render_valuation_section(valuation),
        render_sensitivity_section(sensitivity_rows),
        render_next_actions(next_actions),
    ]
    return "\n".join(parts).strip() + "\n"


def write_markdown_report(path: str | Path, content: str) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    return output_path
