"""Writer agent."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import LabError, ValidationError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer`.

        Synthesize a clear response from research and analysis notes.
        """

        if not state.research_notes and not state.analysis_notes:
            raise ValidationError(
                "WriterAgent requires research_notes or analysis_notes before running."
            )

        state.final_answer = self._write(state)
        state.agent_results.append(
            AgentResult(
                agent=AgentName.WRITER,
                content=state.final_answer,
                metadata={"answer_length": len(state.final_answer)},
            )
        )
        state.add_trace_event(
            "writer.complete",
            {"answer_length": len(state.final_answer), "source_count": len(state.sources)},
        )
        return state

    def _write(self, state: ResearchState) -> str:
        source_list = "\n".join(
            f"- {source.title}{f' ({source.url})' if source.url else ''}"
            for source in state.sources
        )
        system_prompt = (
            "You are the Writer in a multi-agent research system. Produce a polished answer "
            "for the requested audience. Be direct, cite available sources by title, and do "
            "not overstate unsupported claims."
        )
        user_prompt = (
            f"Question: {state.request.query}\n"
            f"Audience: {state.request.audience}\n\n"
            f"Research notes:\n{state.research_notes or 'None'}\n\n"
            f"Analysis notes:\n{state.analysis_notes or 'None'}\n\n"
            f"Available sources:\n{source_list or 'No external sources'}\n\n"
            "Write the final answer."
        )

        try:
            response = self.llm_client.complete(system_prompt, user_prompt)
            state.add_trace_event(
                "writer.llm_usage",
                {
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                },
            )
            return response.content
        except LabError as exc:
            state.errors.append(f"writer.llm_fallback: {exc}")
            return (
                f"# {state.request.query}\n\n"
                f"{state.analysis_notes or state.research_notes}\n\n"
                "Source note: this answer was generated from the current agent state. "
                "External citations should be added after configuring a search provider."
            )
