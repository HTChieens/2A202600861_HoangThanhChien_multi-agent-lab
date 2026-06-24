"""Supervisor / router agent."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def run(self, state: ResearchState) -> ResearchState:
        """Update `state.route_history` with the next route.

        Routes are intentionally simple and inspectable:
        researcher -> analyst -> writer -> critic -> done.
        If the run reaches the iteration cap, route to writer when possible and then stop.
        """

        route = self._choose_route(state)
        state.record_route(route)
        state.add_trace_event(
            "supervisor.route",
            {
                "route": route,
                "iteration": state.iteration,
                "has_sources": bool(state.sources),
                "has_research_notes": bool(state.research_notes),
                "has_analysis_notes": bool(state.analysis_notes),
                "has_final_answer": bool(state.final_answer),
            },
        )
        state.agent_results.append(
            AgentResult(
                agent=AgentName.SUPERVISOR,
                content=f"Next route: {route}",
                metadata={"iteration": state.iteration},
            )
        )
        return state

    def _choose_route(self, state: ResearchState) -> str:
        if state.iteration >= self.settings.max_iterations:
            return "writer" if not state.final_answer else "done"

        if not state.research_notes:
            return "researcher"
        if not state.analysis_notes:
            return "analyst"
        if not state.final_answer:
            return "writer"
        if "critic" not in state.route_history:
            return "critic"
        return "done"
