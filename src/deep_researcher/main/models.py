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


class ResearchPlan(BaseModel):
    """Typed planning output from the Query Planning Agent."""

    sub_questions: list[str] = Field(default_factory=list)
    focus_areas: list[str] = Field(default_factory=list)
    evidence_needs: list[str] = Field(default_factory=list)


class ResearchState(TypedDict, total=False):
    """LangGraph state passed among all research agents."""

    query: str
    thread_id: str
    local_documents: list[SourceDocument]
    research_plan: ResearchPlan
    sub_questions: list[str]
    tavily_sources: list[SourceDocument]
    retrieved_context: list[SourceDocument]
    relevant_context_summary: str
    source_assessments: list[SourceAssessment]
    source_validation_summary: str
    synthesis: str
    contradictions: list[str]
    insights: list[str]
    hypotheses: list[str]
    reproducible_snippet: str
    report: str
    report_revision_edits: list[str]
    report_reflection_attempts: int
    report_reflection_notes: list[str]
    human_review_decision: str
    logs: list[str]
    errors: list[str]
