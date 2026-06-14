"""Web search integration for research agents."""

from __future__ import annotations

from deep_researcher.config import ResearchConfig
from deep_researcher.models import SourceDocument


def tavily_search(
    query: str,
    config: ResearchConfig,
    *,
    max_results: int | None = None,
) -> list[SourceDocument]:
    """Search the web with Tavily and normalize results."""

    if not config.tavily_api_key:
        return [
            SourceDocument(
                title="Tavily API key not configured",
                content=(
                    "Web search was skipped because TAVILY_API_KEY is not set. "
                    "Add a Tavily key in the environment or upload local source "
                    "documents to run a fully grounded investigation."
                ),
                source_type="system_notice",
                metadata={"web_search_skipped": True},
            )
        ]

    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=config.tavily_api_key)
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=max_results or config.max_web_results,
            include_answer=False,
            include_raw_content=True,
        )
    except Exception as exc:  # pragma: no cover - network/client specific
        return [
            SourceDocument(
                title="Tavily search error",
                content=f"Web search failed for query '{query}': {exc}",
                source_type="system_notice",
                metadata={"web_search_error": str(exc)},
            )
        ]

    documents: list[SourceDocument] = []
    for result in response.get("results", []):
        content = result.get("raw_content") or result.get("content") or ""
        if not content.strip():
            continue
        documents.append(
            SourceDocument(
                title=result.get("title") or result.get("url") or "Web result",
                url=result.get("url"),
                content=content,
                source_type="web",
                score=result.get("score"),
                metadata={
                    "published_date": result.get("published_date"),
                    "query": query,
                },
            )
        )

    return documents
