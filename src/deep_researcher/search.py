"""Web search integration for research agents."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from deep_researcher.config import ResearchConfig
from deep_researcher.models import SourceDocument


SOURCE_SEARCH_QUERIES = {
    "research_papers": "{query} research paper study arxiv pubmed scholarly evidence",
    "news_articles": "{query} latest news article analysis Reuters AP BBC",
    "reports": "{query} report whitepaper government industry pdf findings",
    "apis": "{query} API dataset endpoint documentation data source",
}


def tavily_search(
    query: str,
    config: ResearchConfig,
    *,
    max_results: int | None = None,
    source_lane: str = "web",
) -> list[SourceDocument]:
    """Search the web with Tavily and normalize results."""

    if not config.tavily_api_key:
        return [
            SourceDocument(
                title=f"Tavily API key not configured for {source_lane}",
                content=(
                    "Web search was skipped because TAVILY_API_KEY is not set. "
                    "Add a Tavily key in the environment or upload local source "
                    "documents to run a fully grounded investigation. "
                    f"Skipped source lane: {source_lane}."
                ),
                source_type="system_notice",
                metadata={"web_search_skipped": True, "source_lane": source_lane},
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
                metadata={"web_search_error": str(exc), "source_lane": source_lane},
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
                source_type=source_lane,
                score=result.get("score"),
                metadata={
                    "published_date": result.get("published_date"),
                    "query": query,
                    "source_lane": source_lane,
                },
            )
        )

    return documents


def parallel_tavily_search(
    query: str,
    config: ResearchConfig,
    *,
    source_queries: dict[str, str] | None = None,
) -> list[SourceDocument]:
    """Run source-specific Tavily searches concurrently."""

    lanes = source_queries or SOURCE_SEARCH_QUERIES
    max_workers = max(1, min(len(lanes), 4))
    per_lane_results = max(1, config.max_web_results // max(1, len(lanes)))

    documents: list[SourceDocument] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                tavily_search,
                template.format(query=query),
                config,
                max_results=per_lane_results,
                source_lane=lane,
            ): lane
            for lane, template in lanes.items()
        }
        for future in as_completed(futures):
            lane = futures[future]
            try:
                documents.extend(future.result())
            except Exception as exc:  # pragma: no cover - defensive future boundary
                documents.append(
                    SourceDocument(
                        title=f"Parallel Tavily search error for {lane}",
                        content=f"Parallel web search failed for source lane '{lane}': {exc}",
                        source_type="system_notice",
                        metadata={"web_search_error": str(exc), "source_lane": lane},
                    )
                )

    return documents
