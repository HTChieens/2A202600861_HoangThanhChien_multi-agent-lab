"""Tracing helpers.

This module intentionally avoids binding the app to one tracing vendor. The
returned dictionaries are JSON-serializable and can be forwarded to LangSmith,
Langfuse, OpenTelemetry, or a local artifact store.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from importlib import import_module
from time import perf_counter
from typing import Any
from uuid import UUID, uuid4

from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.core.state import ResearchState


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Create a JSON-serializable span dictionary."""

    started = perf_counter()
    span: dict[str, Any] = {
        "name": name,
        "attributes": attributes or {},
        "started_at": datetime.now(UTC).isoformat(),
        "ended_at": None,
        "duration_seconds": None,
        "status": "ok",
        "error": None,
    }
    try:
        yield span
    except Exception as exc:
        span["status"] = "error"
        span["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        span["ended_at"] = datetime.now(UTC).isoformat()
        span["duration_seconds"] = perf_counter() - started


def add_span_to_state(state: ResearchState, span: dict[str, Any]) -> None:
    """Append a completed span to `ResearchState.trace`."""

    state.add_trace_event(
        str(span["name"]),
        {
            "attributes": span.get("attributes", {}),
            "started_at": span.get("started_at"),
            "ended_at": span.get("ended_at"),
            "duration_seconds": span.get("duration_seconds"),
            "status": span.get("status"),
            "error": span.get("error"),
        },
    )


class LangSmithTraceClient:
    """Small optional LangSmith adapter.

    The rest of the app should not import LangSmith directly. If the API key or
    SDK is missing, this adapter becomes a no-op and records that locally.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.enabled = bool(settings.langsmith_api_key)
        self._client: Any | None = None
        self._error: str | None = None

        if not self.enabled:
            return

        try:
            langsmith_module = import_module("langsmith")
            client_class = langsmith_module.Client
            self._client = client_class(api_key=settings.langsmith_api_key)
        except Exception as exc:  # pragma: no cover - depends on optional SDK/env.
            self.enabled = False
            self._error = f"{type(exc).__name__}: {exc}"

    def create_workflow_run(self, state: ResearchState) -> UUID | None:
        """Create a root LangSmith run for one workflow execution."""

        if not self.enabled or self._client is None:
            self._record_disabled(state)
            return None

        run_id = uuid4()
        try:
            self._client.create_run(
                id=run_id,
                name="multi-agent-workflow",
                run_type="chain",
                project_name=self.settings.langsmith_project,
                inputs={
                    "query": state.request.query,
                    "audience": state.request.audience,
                    "max_sources": state.request.max_sources,
                },
                start_time=datetime.now(UTC),
                extra={"metadata": {"app": "multi-agent-research-lab"}},
            )
            state.add_trace_event(
                "langsmith.run_created",
                {"run_id": str(run_id), "project": self.settings.langsmith_project},
            )
            return run_id
        except Exception as exc:  # pragma: no cover - depends on provider/network.
            self._record_error(state, "create_run", exc)
            return None

    def finish_workflow_run(
        self,
        state: ResearchState,
        run_id: UUID | None,
        *,
        error: str | None = None,
    ) -> None:
        """Update the root LangSmith run with final outputs and trace events."""

        if run_id is None or not self.enabled or self._client is None:
            return

        try:
            self._client.update_run(
                run_id,
                end_time=datetime.now(UTC),
                error=error,
                outputs={
                    "final_answer": state.final_answer,
                    "route_history": state.route_history,
                    "errors": state.errors,
                },
                events=state.trace,
                extra={
                    "metadata": {
                        "source_count": len(state.sources),
                        "iteration": state.iteration,
                    }
                },
            )
            self._client._flush_run_ops_buffer()
            url = self._get_run_url(run_id)
            state.add_trace_event(
                "langsmith.run_finished",
                {"run_id": str(run_id), "url": url},
            )
        except Exception as exc:  # pragma: no cover - depends on provider/network.
            self._record_error(state, "update_run", exc)

    def _get_run_url(self, run_id: UUID) -> str | None:
        if self._client is None:
            return None
        try:
            run = self._client.read_run(run_id)
            return str(self._client.get_run_url(run=run))
        except Exception:
            return None

    def _record_disabled(self, state: ResearchState) -> None:
        if self._error is None:
            return
        state.add_trace_event("langsmith.disabled", {"reason": self._error})

    @staticmethod
    def _record_error(state: ResearchState, operation: str, exc: Exception) -> None:
        state.add_trace_event(
            "langsmith.error",
            {"operation": operation, "error": f"{type(exc).__name__}: {exc}"},
        )
