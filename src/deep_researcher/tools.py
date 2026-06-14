"""Safe allowlisted tools used by agents.

The registry gives agents executable capabilities without exposing arbitrary
shell access. Each tool is a named Python callable with a narrow purpose.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from deep_researcher.config import ResearchConfig
from deep_researcher.models import SourceDocument
from deep_researcher.retrieval import retrieve_relevant_context
from deep_researcher.search import parallel_tavily_search


@dataclass(frozen=True, slots=True)
class SafeTool:
    name: str
    description: str
    handler: Callable[..., Any]


class SafeToolRegistry:
    """Minimal allowlist for agent tool execution."""

    def __init__(self) -> None:
        self._tools: dict[str, SafeTool] = {}

    def register(self, tool: SafeTool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def execute(self, name: str, **kwargs: Any) -> Any:
        if name not in self._tools:
            raise KeyError(f"Tool is not allowlisted: {name}")
        return self._tools[name].handler(**kwargs)

    def list_tools(self) -> list[dict[str, str]]:
        return [
            {"name": tool.name, "description": tool.description}
            for tool in self._tools.values()
        ]


def build_default_tool_registry(config: ResearchConfig) -> SafeToolRegistry:
    registry = SafeToolRegistry()
    registry.register(
        SafeTool(
            name="parallel_tavily_search",
            description="Run parallel Tavily source-lane searches for the research query.",
            handler=lambda query: parallel_tavily_search(query, config),
        )
    )
    registry.register(
        SafeTool(
            name="faiss_top_k_retrieval",
            description="Retrieve top-k context chunks from the combined source corpus using FAISS.",
            handler=lambda query, sources: retrieve_relevant_context(query, sources, config),
        )
    )
    registry.register(
        SafeTool(
            name="markdown_report_export",
            description="Prepare a Markdown report payload for download or file export.",
            handler=_markdown_report_export,
        )
    )
    return registry


def _markdown_report_export(report: str, filename: str = "deep_research_report.md") -> dict[str, str]:
    return {"filename": filename, "content": report}
