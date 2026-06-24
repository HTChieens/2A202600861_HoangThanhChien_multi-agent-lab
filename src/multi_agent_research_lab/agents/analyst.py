"""Analyst agent."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import LabError, ValidationError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes`.

        Extract key claims, implications, and evidence gaps from research notes.
        """

        if not state.research_notes:
            raise ValidationError("AnalystAgent requires research_notes before running.")

        state.analysis_notes = self._analyze(state)
        state.agent_results.append(
            AgentResult(
                agent=AgentName.ANALYST,
                content=state.analysis_notes,
                metadata={"source_count": len(state.sources)},
            )
        )
        state.add_trace_event(
            "analyst.complete",
            {
                "source_count": len(state.sources),
                "analysis_length": len(state.analysis_notes),
            },
        )
        return state

    def _analyze(self, state: ResearchState) -> str:
        system_prompt = (
            "You are the Analyst in a multi-agent research system. Turn research notes into "
            "structured insights. Separate strong evidence, weak evidence, and implications."
        )
        user_prompt = (
            f"Question: {state.request.query}\n"
            f"Research notes:\n{state.research_notes}\n\n"
            "Return: 1) key claims, 2) why they matter, 3) evidence gaps or caveats."
        )

        try:
            response = self.llm_client.complete(system_prompt, user_prompt)
            state.add_trace_event(
                "analyst.llm_usage",
                {
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                },
            )
            return response.content
        except LabError as exc:
            state.errors.append(f"analyst.llm_fallback: {exc}")
            return (
                "Key claims:\n"
                f"- The research question is: {state.request.query}\n"
                "- Current notes identify the topic and available evidence, but live source "
                "coverage may be incomplete.\n\n"
                "Why it matters:\n"
                "- The final answer should explain the main ideas, current direction, and "
                "practical tradeoffs for the target audience.\n\n"
                "Evidence gaps and caveats:\n"
                "- Verify recency-sensitive claims with current sources.\n"
                "- Avoid presenting fallback notes as fully sourced facts."
            )
