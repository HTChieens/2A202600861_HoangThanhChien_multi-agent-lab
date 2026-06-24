"""Multi-agent workflow orchestration."""

from multi_agent_research_lab.agents import (
    AnalystAgent,
    CriticAgent,
    ResearcherAgent,
    SupervisorAgent,
    WriterAgent,
)
from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import LangSmithTraceClient


class MultiAgentWorkflow:
    """Builds and runs the multi-agent workflow.

    Keep orchestration here; keep agent internals in `agents/`.
    """

    def __init__(
        self,
        *,
        supervisor: SupervisorAgent | None = None,
        researcher: ResearcherAgent | None = None,
        analyst: AnalystAgent | None = None,
        writer: WriterAgent | None = None,
        critic: CriticAgent | None = None,
        settings: Settings | None = None,
        trace_client: LangSmithTraceClient | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.supervisor = supervisor or SupervisorAgent(settings=self.settings)
        self.researcher = researcher or ResearcherAgent()
        self.analyst = analyst or AnalystAgent()
        self.writer = writer or WriterAgent()
        self.critic = critic or CriticAgent()
        self.trace_client = trace_client or LangSmithTraceClient(self.settings)

    def build(self) -> dict[str, BaseAgent]:
        """Create the executable node registry.

        This project keeps the workflow dependency-light. The returned mapping mirrors
        the nodes one would register in LangGraph and can be swapped for a compiled
        graph later without changing the agent contracts.
        """

        return {
            "researcher": self.researcher,
            "analyst": self.analyst,
            "writer": self.writer,
            "critic": self.critic,
        }

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the workflow and return final state."""

        nodes = self.build()
        state.add_trace_event(
            "workflow.start",
            {"query": state.request.query, "max_iterations": self.settings.max_iterations},
        )
        remote_run_id = self.trace_client.create_workflow_run(state)

        max_steps = self.settings.max_iterations + 2
        try:
            for _ in range(max_steps):
                state = self.supervisor.run(state)
                route = state.route_history[-1]

                if route == "done":
                    state.add_trace_event("workflow.done", {"iteration": state.iteration})
                    self.trace_client.finish_workflow_run(state, remote_run_id)
                    return state

                agent = nodes.get(route)
                if agent is None:
                    raise AgentExecutionError(f"Supervisor selected unknown route: {route}")

                state.add_trace_event("workflow.node_start", {"agent": agent.name})
                state = agent.run(state)
                state.add_trace_event("workflow.node_end", {"agent": agent.name})

            state.errors.append("workflow.max_steps_exceeded")
            state.add_trace_event(
                "workflow.stopped",
                {"reason": "max_steps_exceeded", "max_steps": max_steps},
            )
            if state.final_answer:
                self.trace_client.finish_workflow_run(state, remote_run_id)
                return state
            raise AgentExecutionError("Workflow stopped before producing a final answer.")
        except Exception as exc:
            self.trace_client.finish_workflow_run(state, remote_run_id, error=str(exc))
            raise
