"""Researcher agent."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import LabError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(
        self,
        search_client: SearchClient | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.search_client = search_client or SearchClient()
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`.

        Uses the configured search client when available. If no search provider is
        implemented yet, it creates a transparent planning source so downstream agents
        can still run and the trace shows the missing integration.
        """

        sources = self._collect_sources(state)
        state.sources = sources
        state.research_notes = self._build_notes(state)
        state.agent_results.append(
            AgentResult(
                agent=AgentName.RESEARCHER,
                content=state.research_notes,
                metadata={"source_count": len(state.sources)},
            )
        )
        state.add_trace_event(
            "researcher.complete",
            {"source_count": len(state.sources), "used_fallback": self._used_fallback(sources)},
        )
        return state

    def _collect_sources(self, state: ResearchState) -> list[SourceDocument]:
        try:
            return self.search_client.search(
                state.request.query,
                max_results=state.request.max_sources,
            )
        except LabError as exc:
            state.errors.append(f"researcher.search_fallback: {exc}")
            return [
                SourceDocument(
                    title="Search integration not configured",
                    url=None,
                    snippet=(
                        "No external search provider returned documents. Treat these notes as "
                        "a research plan and verify claims with live sources before submission."
                    ),
                    metadata={"fallback": True, "query": state.request.query},
                )
            ]

    def _build_notes(self, state: ResearchState) -> str:
        source_lines = "\n".join(f"- {source.title}: {source.snippet}" for source in state.sources)
        system_prompt = (
            "You are the Researcher in a multi-agent research system. Convert sources into "
            "concise research notes. Preserve uncertainty and mention missing evidence."
        )
        user_prompt = (
            f"Question: {state.request.query}\n"
            f"Audience: {state.request.audience}\n\n"
            f"Sources:\n{source_lines}\n\n"
            "Write research notes with key facts, useful context, and verification gaps."
        )

        try:
            response = self.llm_client.complete(system_prompt, user_prompt)
            state.add_trace_event(
                "researcher.llm_usage",
                {
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                },
            )
            return response.content
        except LabError as exc:
            state.errors.append(f"researcher.llm_fallback: {exc}")
            return (
                f"Research notes for: {state.request.query}\n\n"
                f"Available source summary:\n{source_lines}\n\n"
                "Verification gaps: external search or citations are not configured yet, so "
                "claims should be checked against current primary or high-quality secondary "
                "sources."
            )

    @staticmethod
    def _used_fallback(sources: list[SourceDocument]) -> bool:
        return any(source.metadata.get("fallback") is True for source in sources)
