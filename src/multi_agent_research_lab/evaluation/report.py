"""Benchmark report rendering."""

from multi_agent_research_lab.core.schemas import BenchmarkMetrics


def render_markdown_report(metrics: list[BenchmarkMetrics]) -> str:
    """Render benchmark metrics to markdown."""

    lines = [
        "# Benchmark Report",
        "",
        "## Summary",
        "",
        _render_summary(metrics),
        "",
        "## Metrics",
        "",
        "| Run | Latency (s) | Cost (USD) | Quality | Notes |",
        "|---|---:|---:|---:|---|",
    ]
    for item in metrics:
        cost = "" if item.estimated_cost_usd is None else f"{item.estimated_cost_usd:.4f}"
        quality = "" if item.quality_score is None else f"{item.quality_score:.1f}"
        lines.append(
            f"| {item.run_name} | {item.latency_seconds:.2f} | {cost} | "
            f"{quality} | {_escape_table_cell(item.notes)} |"
        )
    lines.extend(["", "## Notes", "", _render_notes(metrics)])
    return "\n".join(lines) + "\n"


def _render_summary(metrics: list[BenchmarkMetrics]) -> str:
    if not metrics:
        return "No benchmark runs were recorded."

    fastest = min(metrics, key=lambda item: item.latency_seconds)
    scored = [item for item in metrics if item.quality_score is not None]
    if not scored:
        return f"Recorded {len(metrics)} run(s). Fastest run: `{fastest.run_name}`."

    best_quality = max(scored, key=lambda item: item.quality_score or 0.0)
    return (
        f"Recorded {len(metrics)} run(s). Fastest run: `{fastest.run_name}` "
        f"({fastest.latency_seconds:.2f}s). Best heuristic quality: "
        f"`{best_quality.run_name}` ({best_quality.quality_score:.1f}/10)."
    )


def _render_notes(metrics: list[BenchmarkMetrics]) -> str:
    if not metrics:
        return "- Add benchmark runs to compare baseline and multi-agent behavior."
    return "\n".join(f"- `{item.run_name}`: {item.notes or 'No notes.'}" for item in metrics)


def _escape_table_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")
