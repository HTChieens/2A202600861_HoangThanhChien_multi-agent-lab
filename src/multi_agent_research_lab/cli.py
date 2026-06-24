"""Command-line entrypoint for the lab starter."""

from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import LabError
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.storage import LocalArtifactStore
from multi_agent_research_lab.utils.timer import elapsed_timer

app = typer.Typer(help="Multi-Agent Research Lab starter CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


def _run_single_agent_baseline(request: ResearchQuery) -> ResearchState:
    state = ResearchState(request=request)
    llm = LLMClient()
    system_prompt = (
        "You are a careful single-agent research assistant. Answer the user's query clearly, "
        "state important assumptions, and separate established facts from uncertainty. "
        "If live web search or citations are required, say what should be verified."
    )
    user_prompt = (
        f"Research query: {request.query}\n"
        f"Audience: {request.audience}\n"
        f"Maximum sources desired later in the full system: {request.max_sources}\n\n"
        "Write a concise but useful baseline answer. Use headings or bullets only when they "
        "make the answer easier to scan."
    )

    with elapsed_timer() as elapsed:
        response = llm.complete(system_prompt=system_prompt, user_prompt=user_prompt)
        latency_seconds = elapsed()

    state.final_answer = response.content
    state.add_trace_event(
        "baseline.complete",
        {
            "model": llm.model,
            "latency_seconds": latency_seconds,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "cost_usd": response.cost_usd,
        },
    )
    return state


def _run_multi_agent_workflow(request: ResearchQuery) -> ResearchState:
    state = ResearchState(request=request)
    workflow = MultiAgentWorkflow()
    return workflow.run(state)


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
    max_sources: Annotated[int, typer.Option("--max-sources", help="Desired citation budget")] = 5,
    audience: Annotated[
        str, typer.Option("--audience", help="Target audience")
    ] = "technical learners",
) -> None:
    """Run a single-agent LLM baseline."""

    _init()
    request = ResearchQuery(query=query, max_sources=max_sources, audience=audience)
    try:
        state = _run_single_agent_baseline(request)
    except LabError as exc:
        console.print(Panel.fit(str(exc), title="Baseline failed", style="red"))
        raise typer.Exit(code=1) from exc

    console.print(Panel.fit(state.final_answer or "", title="Single-Agent Baseline"))
    console.print(state.model_dump_json(indent=2))


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
    max_sources: Annotated[int, typer.Option("--max-sources", help="Desired citation budget")] = 5,
    audience: Annotated[
        str, typer.Option("--audience", help="Target audience")
    ] = "technical learners",
) -> None:
    """Run the multi-agent workflow."""

    _init()
    request = ResearchQuery(query=query, max_sources=max_sources, audience=audience)
    try:
        result = _run_multi_agent_workflow(request)
    except LabError as exc:
        console.print(Panel.fit(str(exc), title="Workflow failed", style="red"))
        raise typer.Exit(code=1) from exc
    console.print(result.model_dump_json(indent=2))


@app.command()
def benchmark(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
    output: Annotated[
        str, typer.Option("--output", "-o", help="Report path under reports/")
    ] = "benchmark_report.md",
    max_sources: Annotated[int, typer.Option("--max-sources", help="Desired citation budget")] = 5,
    audience: Annotated[
        str, typer.Option("--audience", help="Target audience")
    ] = "technical learners",
) -> None:
    """Run baseline and multi-agent benchmark, then write a markdown report."""

    _init()

    def baseline_runner(raw_query: str) -> ResearchState:
        request = ResearchQuery(query=raw_query, max_sources=max_sources, audience=audience)
        return _run_single_agent_baseline(request)

    def multi_agent_runner(raw_query: str) -> ResearchState:
        request = ResearchQuery(query=raw_query, max_sources=max_sources, audience=audience)
        return _run_multi_agent_workflow(request)

    try:
        _, baseline_metrics = run_benchmark("baseline", query, baseline_runner)
        _, multi_agent_metrics = run_benchmark("multi-agent", query, multi_agent_runner)
    except LabError as exc:
        console.print(Panel.fit(str(exc), title="Benchmark failed", style="red"))
        raise typer.Exit(code=1) from exc

    report = render_markdown_report([baseline_metrics, multi_agent_metrics])
    path = LocalArtifactStore().write_text(output, report)
    console.print(Panel.fit(str(path), title="Benchmark report written"))


if __name__ == "__main__":
    app()
