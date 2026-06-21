"""Typed models shared by the cybersecurity agents and Streamlit UI."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field

Severity = Literal["critical", "high", "medium", "low", "informational"]
FindingStatus = Literal["open", "needs_review", "mitigated"]


class SourceDocument(BaseModel):
    """Normalized evidence chunk used by retrieval and security agents."""

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
    """Typed planning output from the Security Planning Agent."""

    sub_questions: list[str] = Field(default_factory=list)
    focus_areas: list[str] = Field(default_factory=list)
    evidence_needs: list[str] = Field(default_factory=list)


class SecurityFinding(BaseModel):
    """A normalized security finding emitted by one specialist agent."""

    id: str
    agent: str
    severity: Severity
    category: str
    title: str
    evidence: str
    affected_assets: list[str] = Field(default_factory=list)
    recommended_fix: str
    confidence: float = Field(default=0.75, ge=0.0, le=1.0)
    source_refs: list[str] = Field(default_factory=list)
    status: FindingStatus = "open"


class PolicyGap(BaseModel):
    """Framework control gap for ISO 27001, NIST CSF, SOC 2, or similar."""

    framework: str
    control: str
    status: Literal["pass", "gap", "needs_review"]
    finding: str
    remediation: str


class IncidentStep(BaseModel):
    """Actionable incident-response instruction."""

    phase: Literal["triage", "containment", "eradication", "recovery", "communications", "lessons_learned"]
    action: str
    owner: str
    priority: Severity


class EvaluationMetric(BaseModel):
    """Single deterministic evaluation metric for the security agent loop."""

    name: str
    score: float = Field(ge=0.0, le=1.0)
    passed: bool
    notes: str


class EvaluationSummary(BaseModel):
    """Scorecard used to improve future agent runs."""

    overall_score: float = Field(ge=0.0, le=1.0)
    metrics: list[EvaluationMetric] = Field(default_factory=list)
    improvement_actions: list[str] = Field(default_factory=list)


class EvaluationCase(BaseModel):
    """Offline eval case for repeatable loop engineering."""

    name: str
    query: str
    documents: list[SourceDocument]
    expected_keywords: list[str] = Field(default_factory=list)
    required_agents: list[str] = Field(default_factory=list)
    min_findings: int = 1


class ResearchState(TypedDict, total=False):
    """LangGraph state passed among all cybersecurity agents."""

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
    log_findings: list[SecurityFinding]
    threat_findings: list[SecurityFinding]
    vulnerability_findings: list[SecurityFinding]
    policy_gaps: list[PolicyGap]
    incident_steps: list[IncidentStep]
    findings: list[SecurityFinding]
    risk_score: float
    evaluation_summary: EvaluationSummary
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
