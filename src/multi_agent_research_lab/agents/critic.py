"""Optional critic agent for answer review."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import LabError, ValidationError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class CriticAgent(BaseAgent):
    """Optional fact-checking and safety-review agent."""

    name = "critic"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Validate final answer and append findings.

        The critic does not rewrite the answer. It records review findings so the
        supervisor or a later workflow can decide whether to revise.
        """

        if not state.final_answer:
            raise ValidationError("CriticAgent requires final_answer before running.")

        findings = self._review(state)
        state.agent_results.append(
            AgentResult(
                agent=AgentName.CRITIC,
                content=findings,
                metadata={"source_count": len(state.sources)},
            )
        )
        state.add_trace_event(
            "critic.complete",
            {
                "source_count": len(state.sources),
                "finding_length": len(findings),
            },
        )
        return state

    def _review(self, state: ResearchState) -> str:
        final_answer = state.final_answer
        if final_answer is None:
            raise ValidationError("CriticAgent requires final_answer before reviewing.")

        system_prompt = (
            "You are the Critic in a multi-agent research system. Review the final answer "
            "for unsupported claims, missing citations, clarity, and caveats."
        )
        user_prompt = (
            f"Question: {state.request.query}\n"
            f"Sources available: {len(state.sources)}\n\n"
            f"Final answer:\n{final_answer}\n\n"
            "Return concise findings and a pass/revise recommendation."
        )

        try:
            response = self.llm_client.complete(system_prompt, user_prompt)
            state.add_trace_event(
                "critic.llm_usage",
                {
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                },
            )
            return response.content
        except LabError as exc:
            state.errors.append(f"critic.llm_fallback: {exc}")
            findings = []
            if not state.sources:
                findings.append("- No external sources are attached; citation coverage is weak.")
            if "verify" not in final_answer.lower() and not state.sources:
                findings.append("- Add an explicit verification caveat for unsupported claims.")
            if len(final_answer.split()) < 100:
                findings.append("- The answer may be too short for a research summary.")
            if not findings:
                findings.append(
                    "- Basic checks passed; review currentness-sensitive claims manually."
                )
            return "Critic findings:\n" + "\n".join(findings)
