"""Benchmark utilities for single-agent vs multi-agent runs."""

from collections.abc import Callable
from time import perf_counter

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState

Runner = Callable[[str], ResearchState]


def run_benchmark(
    run_name: str, query: str, runner: Runner
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Measure latency and return benchmark metrics."""

    started = perf_counter()
    state = runner(query)
    latency = perf_counter() - started
    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=_estimate_cost(state),
        quality_score=_estimate_quality(state),
        notes=_build_notes(state),
    )
    return state, metrics


def _estimate_cost(state: ResearchState) -> float | None:
    costs = [
        event.get("payload", {}).get("cost_usd")
        for event in state.trace
        if event.get("payload", {}).get("cost_usd") is not None
    ]
    if not costs:
        return None
    return float(sum(costs))


def _estimate_quality(state: ResearchState) -> float:
    score = 0.0
    if state.research_notes:
        score += 2.0
    if state.analysis_notes:
        score += 2.0
    if state.final_answer:
        score += 3.0
        word_count = len(state.final_answer.split())
        if word_count >= 250:
            score += 1.0
    if state.sources:
        score += 1.0
    if state.sources and not all(source.metadata.get("fallback") for source in state.sources):
        score += 1.0
    if state.errors:
        score -= min(2.0, len(state.errors) * 0.5)
    return max(0.0, min(10.0, score))


def _build_notes(state: ResearchState) -> str:
    citation_coverage = _citation_coverage(state)
    status = "ok" if state.final_answer and not state.errors else "needs review"
    notes = (
        f"status={status}; "
        f"routes={'>'.join(state.route_history) or 'none'}; "
        f"sources={len(state.sources)}; "
        f"citation_coverage={citation_coverage:.0%}; "
        f"errors={len(state.errors)}"
    )
    trace_url = _langsmith_trace_url(state)
    if trace_url:
        notes = f"{notes}; trace={trace_url}"
    return notes


def _citation_coverage(state: ResearchState) -> float:
    if not state.final_answer:
        return 0.0
    if not state.sources:
        return 0.0
    real_sources = [source for source in state.sources if not source.metadata.get("fallback")]
    return len(real_sources) / len(state.sources)


def _langsmith_trace_url(state: ResearchState) -> str | None:
    for event in reversed(state.trace):
        if event.get("name") != "langsmith.run_finished":
            continue
        url = event.get("payload", {}).get("url")
        if isinstance(url, str) and url:
            return url
    return None
