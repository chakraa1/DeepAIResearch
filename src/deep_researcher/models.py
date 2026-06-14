"""Typed models shared by the agents and Streamlit UI."""

from __future__ import annotations

from typing import Any, TypedDict

from pydantic import BaseModel, Field


class SourceDocument(BaseModel):
    """Normalized source chunk used by retrieval and synthesis agents."""

    title: str = "Untitled source"
    url: str | None = None
    content: str
    source_type: str = "document"
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def citation_label(self) -> str:
        if self.url:
            return f"{self.title} ({self.url})"
        return self.title


class SourceAssessment(BaseModel):
    """Critical notes about a source or group of sources."""

    source: str
    credibility: str
    relevance: str
    caveats: str


class ResearchState(TypedDict, total=False):
    """LangGraph state passed among all research agents."""

    query: str
    local_documents: list[SourceDocument]
    sub_questions: list[str]
    tavily_sources: list[SourceDocument]
    retrieved_context: list[SourceDocument]
    source_assessments: list[SourceAssessment]
    source_validation_summary: str
    synthesis: str
    contradictions: list[str]
    insights: list[str]
    hypotheses: list[str]
    report: str
    logs: list[str]
    errors: list[str]
