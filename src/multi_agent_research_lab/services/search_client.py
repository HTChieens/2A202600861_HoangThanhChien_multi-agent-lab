"""Search client abstraction for ResearcherAgent."""

import json
from urllib.error import URLError
from urllib.request import Request, urlopen

from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError, ValidationError
from multi_agent_research_lab.core.schemas import SourceDocument


class SearchClient:
    """Provider-agnostic search client.

    Uses Tavily when `TAVILY_API_KEY` is configured. Without a provider, it returns a
    transparent local source so the rest of the lab can run offline.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search for documents relevant to a query."""

        if not query.strip():
            raise ValidationError("query must not be empty")
        if max_results < 1:
            raise ValidationError("max_results must be at least 1")

        if self.settings.tavily_api_key:
            return self._search_tavily(query=query, max_results=max_results)
        return self._local_mock_results(query=query, max_results=max_results)

    def _search_tavily(self, query: str, max_results: int) -> list[SourceDocument]:
        payload = {
            "api_key": self.settings.tavily_api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",
            "include_answer": False,
        }
        request = Request(
            "https://api.tavily.com/search",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.settings.timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (OSError, URLError, json.JSONDecodeError) as exc:
            raise AgentExecutionError(f"Search provider call failed: {exc}") from exc

        results = data.get("results", [])
        documents = [
            SourceDocument(
                title=str(item.get("title") or "Untitled source"),
                url=item.get("url"),
                snippet=str(item.get("content") or item.get("snippet") or ""),
                metadata={
                    "provider": "tavily",
                    "score": item.get("score"),
                },
            )
            for item in results[:max_results]
        ]
        if not documents:
            return self._local_mock_results(query=query, max_results=max_results)
        return documents

    @staticmethod
    def _local_mock_results(query: str, max_results: int) -> list[SourceDocument]:
        templates = [
            (
                "Local research planning note",
                "Use this offline fallback to outline the topic, decompose subquestions, "
                "and identify claims that need external verification.",
            ),
            (
                "Citation coverage reminder",
                "Before final submission, replace local fallback notes with current primary "
                "sources or high-quality secondary sources.",
            ),
            (
                "Evaluation checklist",
                "Check freshness, authority, claim-source alignment, and uncertainty for "
                "each major point in the answer.",
            ),
        ]
        return [
            SourceDocument(
                title=title,
                url=None,
                snippet=f"{snippet} Query: {query}",
                metadata={"provider": "local_mock", "fallback": True},
            )
            for title, snippet in templates[:max_results]
        ]
