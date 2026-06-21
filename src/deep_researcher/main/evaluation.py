"""Deterministic evaluation harness for CyberSecurityAIAgent.

The scorecard is intentionally lightweight and local-first. It evaluates the
whole agent state, not only an LLM answer, so it can run in Streamlit Cloud
without paid judge models or external telemetry.
"""

from __future__ import annotations

from statistics import mean
from typing import Any

from deep_researcher.config import ResearchConfig
from deep_researcher.main.models import (
    EvaluationCase,
    EvaluationMetric,
    EvaluationSummary,
    SourceDocument,
)

AUTHORIZED_CVE_DOMAINS = (
    "nvd.nist.gov",
    "cve.org",
    "cisa.gov",
    "mitre.org",
)


def evaluate_state(state: dict[str, Any]) -> EvaluationSummary:
    """Score a completed security-agent run."""

    findings = state.get("findings", [])
    policy_gaps = state.get("policy_gaps", [])
    incident_steps = state.get("incident_steps", [])
    retrieved_context = state.get("retrieved_context", [])

    active_agents = {
        getattr(finding, "agent", "")
        for finding in findings
        if getattr(finding, "agent", "")
    }
    source_urls = [
        str(getattr(source, "url", "") or "")
        for source in retrieved_context
    ]

    actionability = _fraction(
        bool(getattr(finding, "recommended_fix", "").strip())
        and len(getattr(finding, "evidence", "").strip()) >= 12
        for finding in findings
    )
    severity_triage = 1.0 if any(getattr(finding, "severity", "") in {"critical", "high"} for finding in findings) else 0.6
    if not findings:
        severity_triage = 0.0

    cve_needed = any("CVE-" in item for item in [state.get("query", ""), *[getattr(f, "evidence", "") for f in findings]])
    authorized_cve_score = 1.0
    if cve_needed:
        authorized_cve_score = 1.0 if any(any(domain in url for domain in AUTHORIZED_CVE_DOMAINS) for url in source_urls) else 0.25

    incident_phases = {getattr(step, "phase", "") for step in incident_steps}
    required_phases = {"triage", "containment", "eradication", "recovery", "communications", "lessons_learned"}
    incident_score = len(incident_phases & required_phases) / len(required_phases)

    metrics = [
        EvaluationMetric(
            name="specialist_agent_coverage",
            score=min(len(active_agents) / 5, 1.0),
            passed={"Log Monitor Agent", "Threat Intelligence Agent", "Vulnerability Scanner Agent"} <= active_agents,
            notes=f"Findings emitted by {len(active_agents)} specialist agent(s).",
        ),
        EvaluationMetric(
            name="finding_actionability",
            score=actionability,
            passed=actionability >= 0.8,
            notes="Findings should include evidence and a concrete recommended fix.",
        ),
        EvaluationMetric(
            name="severity_triage",
            score=severity_triage,
            passed=severity_triage >= 0.6,
            notes="The run should triage high-risk signals when present.",
        ),
        EvaluationMetric(
            name="authorized_cve_sources",
            score=authorized_cve_score,
            passed=authorized_cve_score >= 0.8,
            notes="CVE-related analysis should be grounded in NVD, CVE.org, CISA, or MITRE sources.",
        ),
        EvaluationMetric(
            name="incident_response_completeness",
            score=incident_score,
            passed=incident_score >= 0.8,
            notes=f"Covered phases: {', '.join(sorted(incident_phases)) or 'none'}.",
        ),
        EvaluationMetric(
            name="policy_mapping",
            score=1.0 if policy_gaps else 0.0,
            passed=bool(policy_gaps),
            notes="Policy Checker Agent should map risk to ISO, NIST, or SOC 2 controls.",
        ),
    ]
    score = round(mean(metric.score for metric in metrics), 2)
    return EvaluationSummary(
        overall_score=score,
        metrics=metrics,
        improvement_actions=_improvement_actions(metrics),
    )


def run_offline_evaluation(config: ResearchConfig | None = None) -> list[dict[str, Any]]:
    """Run repeatable local eval cases through the full LangGraph workflow."""

    from deep_researcher.main.workflow import DeepResearchWorkflow

    workflow = DeepResearchWorkflow(config or ResearchConfig(openai_api_key=None, tavily_api_key=None))
    results: list[dict[str, Any]] = []
    for case in default_eval_cases():
        state = workflow.run(
            case.query,
            local_documents=case.documents,
            thread_id=f"eval-{case.name.lower().replace(' ', '-')}",
        )
        summary = state.get("evaluation_summary") or evaluate_state(state)
        report_text = state.get("report", "")
        findings = state.get("findings", [])
        keyword_hits = sum(
            1 for keyword in case.expected_keywords if keyword.lower() in report_text.lower()
        )
        required_agents = {
            getattr(finding, "agent", "")
            for finding in findings
        }
        passed = (
            summary.overall_score >= 0.65
            and len(findings) >= case.min_findings
            and keyword_hits >= max(1, len(case.expected_keywords) // 2)
            and set(case.required_agents) <= required_agents
        )
        results.append(
            {
                "case": case.name,
                "passed": passed,
                "score": summary.overall_score,
                "finding_count": len(findings),
                "keyword_hits": keyword_hits,
                "required_agents_met": sorted(set(case.required_agents) & required_agents),
                "improvement_actions": summary.improvement_actions,
            }
        )
    return results


def default_eval_cases() -> list[EvaluationCase]:
    """Security scenarios used by the Streamlit eval tab and pytest."""

    return [
        EvaluationCase(
            name="Brute force and SQL injection",
            query="Assess suspicious login failures and SQL injection attempts for an internet banking API.",
            documents=[
                SourceDocument(
                    title="auth-api.log",
                    source_type="log",
                    content=(
                        "2026-06-20T10:03:11Z failed password for admin from 203.0.113.44\n"
                        "2026-06-20T10:03:12Z failed password for admin from 203.0.113.44\n"
                        "2026-06-20T10:03:13Z failed password for admin from 203.0.113.44\n"
                        "2026-06-20T10:04:01Z GET /accounts?id=1 UNION SELECT card_number FROM cards\n"
                    ),
                )
            ],
            expected_keywords=["brute force", "SQL injection", "containment"],
            required_agents=["Log Monitor Agent", "Incident Response Agent"],
            min_findings=2,
        ),
        EvaluationCase(
            name="Container and database hardening",
            query="Review Docker and database configuration for a PCI workloads platform.",
            documents=[
                SourceDocument(
                    title="Dockerfile",
                    source_type="dockerfile",
                    content=(
                        "FROM python:3.12\n"
                        "ENV MYSQL_ALLOW_EMPTY_PASSWORD=yes\n"
                        "ENV DEBUG=true\n"
                        "COPY . /app\n"
                        "CMD [\"python\", \"app.py\"]\n"
                    ),
                ),
                SourceDocument(
                    title="postgresql.conf",
                    source_type="database_config",
                    content="listen_addresses='0.0.0.0'\nssl = off\n",
                ),
            ],
            expected_keywords=["Docker", "database", "Policy"],
            required_agents=["Vulnerability Scanner Agent", "Policy Checker Agent"],
            min_findings=2,
        ),
    ]


def _fraction(values) -> float:
    items = list(values)
    if not items:
        return 0.0
    return sum(1 for item in items if item) / len(items)


def _improvement_actions(metrics: list[EvaluationMetric]) -> list[str]:
    actions = []
    for metric in metrics:
        if metric.passed:
            continue
        if metric.name == "specialist_agent_coverage":
            actions.append("Add or upload evidence that exercises all specialist agents before relying on the assessment.")
        elif metric.name == "authorized_cve_sources":
            actions.append("Enable Tavily or add NVD/CVE.org/CISA/MITRE evidence for CVE-specific analysis.")
        elif metric.name == "incident_response_completeness":
            actions.append("Ensure the response plan covers triage, containment, eradication, recovery, communications, and lessons learned.")
        elif metric.name == "policy_mapping":
            actions.append("Map each major risk to at least one ISO 27001, NIST CSF, or SOC 2 control.")
        else:
            actions.append(f"Improve metric: {metric.name}.")
    return actions or ["Keep collecting production feedback and compare scorecards across runs."]
