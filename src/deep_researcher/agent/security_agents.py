"""Specialized agents for CyberSecurityAIAgent."""

from __future__ import annotations

from collections import Counter
import re
from urllib.parse import urlparse

from deep_researcher.agent.llm import ResearchLLM
from deep_researcher.agent.prompts import get_system_prompt, render_system_prompt
from deep_researcher.config import ResearchConfig
from deep_researcher.main.evaluation import evaluate_state
from deep_researcher.main.models import (
    EvaluationSummary,
    IncidentStep,
    PolicyGap,
    ResearchPlan,
    ResearchState,
    SecurityFinding,
    SourceAssessment,
    SourceDocument,
)
from deep_researcher.tools import SafeToolRegistry, build_default_tool_registry
from deep_researcher.tools.search import SOURCE_SEARCH_QUERIES

CVE_RE = re.compile(r"CVE-\d{4}-\d{4,7}", re.IGNORECASE)


class ResearchAgents:
    """Collection of LangGraph-compatible cybersecurity agent callables."""

    def __init__(self, config: ResearchConfig) -> None:
        self.config = config
        self.llm = ResearchLLM(config)
        self.tools: SafeToolRegistry = build_default_tool_registry(config)

    def plan_research(self, state: ResearchState) -> ResearchState:
        """Security Planning Agent: scopes evidence and specialist handoffs."""

        query = state["query"]
        logs = _with_logs(state, "Security Planning Agent started: scoping assets, evidence, and agent handoffs.")
        prompt = f"""Security assessment request: {query}

Create 5 focused investigation questions for:
1. suspicious logs,
2. authorized CVE and threat intelligence,
3. code, API, database, and Docker configuration weaknesses,
4. incident response actions,
5. ISO 27001, NIST CSF, and SOC 2 control gaps.
Return one direct question per line."""
        response = self.llm.generate(get_system_prompt("security_planner"), prompt)
        plan = _build_security_plan(query, response)
        return {
            **state,
            "research_plan": plan,
            "sub_questions": plan.sub_questions,
            "logs": [
                *logs,
                f"Security Planning Agent completed: generated {len(plan.sub_questions)} typed investigation questions.",
            ],
        }

    def monitor_logs(self, state: ResearchState) -> ResearchState:
        """Log Monitor Agent: detects unusual activity and attack patterns."""

        logs = _with_logs(state, "Log Monitor Agent started: parsing uploaded system, app, and network logs.")
        log_findings = _detect_log_findings(state.get("local_documents", []), state["query"])
        findings = _merge_findings(state.get("findings", []), log_findings)
        return {
            **state,
            "log_findings": log_findings,
            "findings": findings,
            "logs": [
                *logs,
                f"Log Monitor Agent completed: detected {len(log_findings)} log-driven security finding(s).",
            ],
        }

    def retrieve_context(self, state: ResearchState) -> ResearchState:
        """Threat Intelligence Agent: retrieves authorized CVE and threat context."""

        query = state["query"]
        logs = _with_logs(
            state,
            "Threat Intelligence Agent started: querying authorized CVE and security advisory lanes.",
        )
        source_lanes = ", ".join(SOURCE_SEARCH_QUERIES.keys())
        contextual_prompt = render_system_prompt(
            "threat_intelligence",
            query=query,
            source_lanes=source_lanes,
            top_k=self.config.max_retrieval_docs,
        )
        retrieval_plan = self.llm.generate(
            contextual_prompt,
            "Create a concise threat-intelligence retrieval plan before search.",
        )
        web_sources = self.tools.execute("parallel_tavily_search", query=query)
        seed_sources = _authorized_reference_sources(query)
        all_sources = _dedupe_sources([*state.get("local_documents", []), *web_sources, *seed_sources])
        retrieved_context = self.tools.execute(
            "faiss_top_k_retrieval",
            query=query,
            sources=all_sources,
        )
        if not retrieved_context:
            retrieved_context = all_sources[: self.config.max_retrieval_docs]
        retrieved_context = retrieved_context[: self.config.max_retrieval_docs]
        assessments = [_assess_source(source) for source in retrieved_context[: self.config.validator_top_k]]
        threat_findings = _detect_threat_findings(query, retrieved_context, state.get("local_documents", []))

        source_inventory = _format_source_inventory(all_sources)
        tuner_prompt = render_system_prompt(
            "security_context_tuner",
            query=query,
            source_inventory=source_inventory,
            faiss_top_context=_format_sources(retrieved_context, max_chars=3_500),
        )
        relevant_context_summary = _limit_words(
            self.llm.generate(
                tuner_prompt,
                "Tune the combined security evidence into concise context for downstream agents.",
            ),
            150,
        )

        validation_summary = self.llm.generate(
            get_system_prompt("security_source_validator"),
            f"""Security request: {query}

Top retrieved context:
{_format_sources(retrieved_context, max_chars=4_000)}

Heuristic source assessment:
{chr(10).join(f'- {item.source}: {item.credibility}; {item.caveats}' for item in assessments)}

Validate the source authority in under 120 words. Highlight if CVE-specific claims are not backed by NVD, CVE.org, CISA, or MITRE.""",
        )

        findings = _merge_findings(state.get("findings", []), threat_findings)
        return {
            **state,
            "tavily_sources": web_sources,
            "retrieved_context": retrieved_context,
            "relevant_context_summary": relevant_context_summary,
            "source_assessments": assessments,
            "source_validation_summary": _limit_words(validation_summary, 120),
            "threat_findings": threat_findings,
            "findings": findings,
            "logs": [
                *logs,
                (
                    "Threat Intelligence Agent completed: "
                    f"ran {len(SOURCE_SEARCH_QUERIES)} source lanes, indexed {len(all_sources)} documents, "
                    f"and produced {len(threat_findings)} threat finding(s)."
                ),
                f"Threat retrieval plan: {_limit_words(retrieval_plan, 45)}",
                f"Security context tuning completed: {_limit_words(relevant_context_summary, 45)}",
            ],
        }

    def scan_vulnerabilities(self, state: ResearchState) -> ResearchState:
        """Vulnerability Scanner Agent: scans code, API, DB, and Docker evidence."""

        logs = _with_logs(
            state,
            "Vulnerability Scanner Agent started: scanning code, API, Docker, and database configuration evidence.",
        )
        vulnerability_findings = _detect_vulnerability_findings(state.get("local_documents", []), state["query"])
        findings = _merge_findings(state.get("findings", []), vulnerability_findings)
        return {
            **state,
            "vulnerability_findings": vulnerability_findings,
            "findings": findings,
            "logs": [
                *logs,
                f"Vulnerability Scanner Agent completed: detected {len(vulnerability_findings)} configuration or code weakness(es).",
            ],
        }

    def build_incident_response(self, state: ResearchState) -> ResearchState:
        """Incident Response Agent: builds action plans from findings."""

        logs = _with_logs(
            state,
            "Incident Response Agent started: creating triage, containment, recovery, and communication steps.",
        )
        findings = state.get("findings", [])
        incident_steps = _build_incident_steps(findings)
        response_text = _render_incident_plan(incident_steps, findings)
        finding = SecurityFinding(
            id="IR-001",
            agent="Incident Response Agent",
            severity=_max_severity(findings),
            category="incident_response",
            title="Coordinated incident response plan required",
            evidence=f"{len(findings)} finding(s) need an auditable response workflow.",
            affected_assets=_affected_assets(findings),
            recommended_fix="Execute the incident response steps in priority order and capture evidence for audit review.",
            confidence=0.9,
            source_refs=["agent:incident_response"],
        )
        return {
            **state,
            "incident_steps": incident_steps,
            "synthesis": response_text,
            "findings": _merge_findings(findings, [finding]),
            "logs": [
                *logs,
                f"Incident Response Agent completed: generated {len(incident_steps)} response step(s).",
            ],
        }

    def check_policies(self, state: ResearchState) -> ResearchState:
        """Policy Checker Agent: maps evidence to ISO, NIST, and SOC 2 gaps."""

        logs = _with_logs(
            state,
            "Policy Checker Agent started: checking ISO 27001, NIST CSF, and SOC 2 control coverage.",
        )
        gaps = _build_policy_gaps(state)
        policy_findings = [
            SecurityFinding(
                id=f"POL-{index:03d}",
                agent="Policy Checker Agent",
                severity="medium" if gap.status == "gap" else "low",
                category="policy",
                title=f"{gap.framework} {gap.control}: {gap.status.replace('_', ' ')}",
                evidence=gap.finding,
                affected_assets=_affected_assets(state.get("findings", [])) or ["security program"],
                recommended_fix=gap.remediation,
                confidence=0.8,
                source_refs=[gap.framework],
                status="needs_review" if gap.status == "needs_review" else "open",
            )
            for index, gap in enumerate(gaps, start=1)
            if gap.status != "pass"
        ]
        return {
            **state,
            "policy_gaps": gaps,
            "findings": _merge_findings(state.get("findings", []), policy_findings),
            "logs": [
                *logs,
                f"Policy Checker Agent completed: mapped {len(gaps)} framework control result(s).",
            ],
        }

    def evaluate_assessment(self, state: ResearchState) -> ResearchState:
        """Evaluation Loop Agent: scores output quality and recommends improvements."""

        logs = _with_logs(
            state,
            "Evaluation Loop Agent started: scoring coverage, actionability, authorized sources, and policy mapping.",
        )
        evaluation_summary: EvaluationSummary = evaluate_state(state)
        return {
            **state,
            "evaluation_summary": evaluation_summary,
            "risk_score": _risk_score(state.get("findings", [])),
            "reproducible_snippet": _build_reproducible_snippet(state, evaluation_summary),
            "logs": [
                *logs,
                f"Evaluation Loop Agent completed: overall score {evaluation_summary.overall_score:.2f}.",
            ],
        }

    def review_before_report(self, state: ResearchState) -> ResearchState:
        """Human review gate before report generation."""

        logs = _with_logs(
            state,
            "Human Review Gate reached: reviewing findings before final security report.",
        )
        decision = "auto-approved"
        if self.config.require_human_review:
            try:
                from langgraph.types import interrupt

                decision_payload = interrupt(
                    {
                        "message": "Review before creating the security report.",
                        "query": state["query"],
                        "findings": [finding.model_dump() for finding in state.get("findings", [])],
                        "evaluation": state.get("evaluation_summary").model_dump()
                        if state.get("evaluation_summary")
                        else {},
                    }
                )
                decision = str(decision_payload or "approved")
            except Exception as exc:  # pragma: no cover - interactive runtime path
                decision = f"interrupt-unavailable: {exc}"

        return {
            **state,
            "human_review_decision": decision,
            "logs": [*logs, f"Human Review Gate completed: {decision}."],
        }

    def build_report(self, state: ResearchState) -> ResearchState:
        """Security Report Builder Agent: compiles the final Markdown report."""

        logs = _with_logs(
            state,
            "Security Report Builder Agent started: compiling board-ready security assessment.",
        )
        llm_note = self.llm.generate(
            get_system_prompt("security_report_builder"),
            f"""Security request: {state['query']}

Findings:
{_format_findings(state.get('findings', []), max_items=12)}

Incident response:
{_format_incident_steps(state.get('incident_steps', []))}

Policy gaps:
{_format_policy_gaps(state.get('policy_gaps', []))}

Evaluation score: {state.get('evaluation_summary').overall_score if state.get('evaluation_summary') else 'not scored'}

Write concise executive language for a financial-services security audience.""",
        )
        report = _render_security_report(state, _limit_words(llm_note, 90), self.config)
        return {
            **state,
            "report": report,
            "logs": [
                *logs,
                "Security Report Builder Agent completed: generated final Markdown report.",
            ],
        }

    def reflect_on_report(self, state: ResearchState) -> ResearchState:
        """Reflection Agent: validates report sections and applies fixes."""

        logs = _with_logs(state, "Report Reflection Agent started: validating security report guardrails.")
        report = state.get("report", "")
        attempts = 0
        notes: list[str] = []
        issues = _validate_security_report(report)
        while issues and attempts < self.config.report_reflection_retry_limit:
            attempts += 1
            notes.append(f"Attempt {attempts}: fixed {', '.join(issues)}.")
            report = _render_security_report(state, "Reflection regenerated missing sections.", self.config)
            issues = _validate_security_report(report)
        if issues:
            notes.append(f"Remaining issues after retry limit: {', '.join(issues)}.")
        else:
            notes.append("Security report passed reflection validation.")
        return {
            **state,
            "report": report,
            "report_reflection_attempts": attempts,
            "report_reflection_notes": notes,
            "logs": [
                *logs,
                f"Report Reflection Agent completed: {attempts} retry attempt(s).",
            ],
        }

    def revise_report_inline(self, state: ResearchState) -> ResearchState:
        """Report Revision Agent: applies targeted Markdown section edits."""

        logs = _with_logs(
            state,
            "Report Revision Agent started: applying targeted Markdown edits to security report.",
        )
        revised, edits = _apply_inline_report_edits(state.get("report", ""), state)
        return {
            **state,
            "report": revised,
            "report_revision_edits": edits,
            "logs": [
                *logs,
                f"Report Revision Agent completed: applied {len(edits)} targeted inline edits.",
            ],
        }


def _build_security_plan(query: str, response: str) -> ResearchPlan:
    sub_questions = _parse_lines(response)
    if len(sub_questions) < 5:
        sub_questions = [
            f"What suspicious log events indicate attacks or anomalous behavior for {query}?",
            f"Which CVEs or threat reports from NVD, CVE.org, CISA, or MITRE affect the described system?",
            f"What code, API, Docker, Linux, MSSQL, MySQL, Oracle, or Postgres weaknesses are visible?",
            f"What immediate incident response steps should security and DevOps execute?",
            f"Which ISO 27001, NIST CSF, or SOC 2 controls need remediation evidence?",
        ]
    return ResearchPlan(
        sub_questions=sub_questions[:5],
        focus_areas=["logs", "authorized CVE intelligence", "vulnerability scanning", "incident response", "policy compliance"],
        evidence_needs=["logs", "code/config files", "Dockerfiles", "database configs", "NVD/CVE.org/CISA/MITRE sources"],
    )


def _detect_log_findings(documents: list[SourceDocument], query: str) -> list[SecurityFinding]:
    findings: list[SecurityFinding] = []
    for source in documents:
        content = source.content
        lowered_title = source.title.lower()
        looks_like_log = source.source_type in {"log", "system_log", "network_log", "uploaded_file"} or any(
            marker in content.lower()
            for marker in ("failed password", "authentication failure", "status=401", "union select", "denied", "ransomware")
        )
        if not looks_like_log:
            continue
        failed_count = len(re.findall(r"failed password|authentication failure|invalid login|status=401|status=403", content, re.I))
        if failed_count >= 3:
            findings.append(
                _finding(
                    agent="Log Monitor Agent",
                    severity="high",
                    category="authentication",
                    title="Probable brute-force authentication attack",
                    evidence=f"{source.title} contains {failed_count} failed authentication signals. {_first_matching_line(content, 'failed|401|403')}",
                    assets=[source.title],
                    fix="Block abusive source IPs, enforce MFA, review account lockout controls, and preserve logs for investigation.",
                    refs=[source.title],
                )
            )
        for pattern, severity, category, title, fix in LOG_PATTERNS:
            match = re.search(pattern, content, re.I)
            if match:
                findings.append(
                    _finding(
                        agent="Log Monitor Agent",
                        severity=severity,
                        category=category,
                        title=title,
                        evidence=f"{source.title}: {_snippet(content, match.start())}",
                        assets=[source.title if "log" in lowered_title else source.title],
                        fix=fix,
                        refs=[source.title],
                    )
                )
    if not findings and any(word in query.lower() for word in ("log", "monitor", "siem", "attack")):
        findings.append(
            _finding(
                agent="Log Monitor Agent",
                severity="medium",
                category="telemetry",
                title="No log evidence provided for requested monitoring assessment",
                evidence="The request asks for log monitoring, but no parseable log evidence was uploaded.",
                assets=["logging pipeline"],
                fix="Upload representative system, application, firewall, IDS, or cloud audit logs and rerun the workflow.",
                refs=["local.input"],
            )
        )
    return _dedupe_findings(findings)


def _detect_threat_findings(
    query: str,
    retrieved_context: list[SourceDocument],
    local_documents: list[SourceDocument],
) -> list[SecurityFinding]:
    text = "\n".join([query, *[source.content for source in retrieved_context], *[doc.content for doc in local_documents]])
    cves = sorted({item.upper() for item in CVE_RE.findall(text)})
    for alias, cve in KNOWN_CVE_ALIASES.items():
        if alias in text.lower() and cve not in cves:
            cves.append(cve)

    findings: list[SecurityFinding] = []
    official_sources = [source for source in retrieved_context if _is_authorized_cve_source(source)]
    for index, cve in enumerate(cves[:6], start=1):
        refs = [source.url or source.title for source in official_sources[:3]] or ["authorized source search required"]
        severity = "high" if any("known exploited" in source.content.lower() or "kev" in source.content.lower() for source in official_sources) else "medium"
        findings.append(
            SecurityFinding(
                id=f"TI-{index:03d}",
                agent="Threat Intelligence Agent",
                severity=severity,
                category="cve_exposure",
                title=f"Potential exposure to {cve}",
                evidence=(
                    f"{cve} was found in the request or retrieved threat context. "
                    f"Authorized source coverage: {_authorized_source_label(official_sources)}."
                ),
                affected_assets=_infer_assets(text),
                recommended_fix="Confirm affected versions against NVD/CVE.org/CISA/MITRE records, patch or isolate impacted assets, and document compensating controls.",
                confidence=0.7 if official_sources else 0.45,
                source_refs=refs,
            )
        )
    if not findings and any(source.source_type == "system_notice" for source in retrieved_context):
        findings.append(
            SecurityFinding(
                id="TI-001",
                agent="Threat Intelligence Agent",
                severity="low",
                category="threat_intelligence_coverage",
                title="Threat intelligence search needs live Tavily or uploaded advisory evidence",
                evidence="CVE-specific lookups were limited to local references because Tavily is not configured.",
                affected_assets=["threat intelligence process"],
                recommended_fix="Set TAVILY_API_KEY or upload NVD, CVE.org, CISA KEV, MITRE, or vendor advisory evidence for the target technologies.",
                confidence=0.8,
                source_refs=["system_notice"],
            )
        )
    return findings


def _detect_vulnerability_findings(documents: list[SourceDocument], query: str) -> list[SecurityFinding]:
    findings: list[SecurityFinding] = []
    for source in documents:
        content = source.content
        for pattern, severity, category, title, fix in VULNERABILITY_PATTERNS:
            match = re.search(pattern, content, re.I | re.M)
            if match:
                findings.append(
                    _finding(
                        agent="Vulnerability Scanner Agent",
                        severity=severity,
                        category=category,
                        title=title,
                        evidence=f"{source.title}: {_snippet(content, match.start())}",
                        assets=[source.title],
                        fix=fix,
                        refs=[source.title],
                    )
                )
        if _looks_like_dockerfile(source) and "USER " not in content.upper():
            findings.append(
                _finding(
                    agent="Vulnerability Scanner Agent",
                    severity="high",
                    category="container_hardening",
                    title="Docker image may run as root",
                    evidence=f"{source.title} has a FROM instruction but no non-root USER directive.",
                    assets=[source.title],
                    fix="Add a dedicated non-root user, drop Linux capabilities, and scan the built image before deployment.",
                    refs=[source.title],
                )
            )
    if not findings and any(word in query.lower() for word in ("scan", "docker", "database", "api", "code", "vulnerability")):
        findings.append(
            _finding(
                agent="Vulnerability Scanner Agent",
                severity="medium",
                category="scanner_coverage",
                title="No code, API, database, or Docker artifacts were available to scan",
                evidence="The scanner needs uploaded source code, API configuration, database settings, SBOMs, or Dockerfiles for grounded analysis.",
                assets=["application estate"],
                fix="Upload representative files such as Dockerfile, requirements, API gateway config, database config, or IaC manifests.",
                refs=["local.input"],
            )
        )
    return _dedupe_findings(findings)


def _build_incident_steps(findings: list[SecurityFinding]) -> list[IncidentStep]:
    priority = _max_severity(findings)
    steps = [
        IncidentStep(
            phase="triage",
            action="Open an incident record, assign severity, preserve logs, and identify affected customer, payment, or privileged-access systems.",
            owner="Security Operations Lead",
            priority=priority,
        ),
        IncidentStep(
            phase="containment",
            action="Isolate affected hosts or services, block malicious indicators, rotate exposed credentials, and enforce MFA for impacted identities.",
            owner="SOC and Platform Engineering",
            priority=priority,
        ),
        IncidentStep(
            phase="eradication",
            action="Patch vulnerable components, remove malicious persistence, fix insecure configuration, and validate clean images or database settings.",
            owner="DevSecOps",
            priority=priority,
        ),
        IncidentStep(
            phase="recovery",
            action="Restore services through change-controlled releases, monitor for recurrence, and verify key business flows before full traffic restoration.",
            owner="Service Owner",
            priority="high" if priority in {"critical", "high"} else "medium",
        ),
        IncidentStep(
            phase="communications",
            action="Notify risk, legal, compliance, and customer-response teams with facts, impact, decisions, and regulatory triggers.",
            owner="Incident Commander",
            priority="medium",
        ),
        IncidentStep(
            phase="lessons_learned",
            action="Run a post-incident review, update detections, close policy gaps, and add regression eval cases for this incident pattern.",
            owner="CISO Office",
            priority="medium",
        ),
    ]
    return steps


def _build_policy_gaps(state: ResearchState) -> list[PolicyGap]:
    findings = state.get("findings", [])
    corpus = "\n".join(doc.content for doc in state.get("local_documents", [])).lower()
    gaps: list[PolicyGap] = []
    high_risk = any(finding.severity in {"critical", "high"} for finding in findings)
    credential_risk = any("credential" in finding.category or "password" in finding.evidence.lower() for finding in findings)
    docker_or_db_risk = any(finding.category in {"container_hardening", "database_hardening", "secret_management"} for finding in findings)

    gaps.append(
        PolicyGap(
            framework="NIST CSF 2.0",
            control="DE.CM and ID.RA",
            status="gap" if high_risk else "needs_review",
            finding="High-risk findings require stronger continuous monitoring and documented risk analysis." if high_risk else "No high-risk event was detected, but monitoring evidence should be reviewed.",
            remediation="Tune SIEM detections, maintain an asset-risk register, and review risk acceptance with security leadership.",
        )
    )
    gaps.append(
        PolicyGap(
            framework="SOC 2",
            control="CC6.1 and CC6.3",
            status="gap" if credential_risk or "mfa" not in corpus else "pass",
            finding="Credential or MFA evidence is weak for privileged and customer-impacting access.",
            remediation="Enforce MFA, rotate exposed secrets, require least privilege, and collect access-review evidence.",
        )
    )
    gaps.append(
        PolicyGap(
            framework="ISO 27001:2022",
            control="A.8.8 and A.8.9",
            status="gap" if docker_or_db_risk else "needs_review",
            finding="Technical vulnerability management and secure configuration evidence need remediation or review.",
            remediation="Track vulnerabilities to closure, baseline hardened Docker and database configurations, and verify with automated scans.",
        )
    )
    gaps.append(
        PolicyGap(
            framework="NIST CSF 2.0",
            control="RC.RP and RS.CO",
            status="needs_review" if "backup" not in corpus and "restore" not in corpus else "pass",
            finding="Recovery and communications evidence is not explicit in the supplied artifacts.",
            remediation="Document restore runbooks, test backups, and maintain legal, compliance, regulator, and customer communication templates.",
        )
    )
    return gaps


def _render_security_report(state: ResearchState, llm_note: str, config: ResearchConfig) -> str:
    findings = state.get("findings", [])
    top_findings = sorted(findings, key=lambda finding: SEVERITY_ORDER[finding.severity], reverse=True)[:10]
    evaluation = state.get("evaluation_summary")
    risk_score = state.get("risk_score", _risk_score(findings))
    source_lines = _format_authorized_sources(state.get("retrieved_context", []))
    if not source_lines:
        source_lines = "- No live authorized threat-intelligence sources were retrieved. Configure Tavily or upload official advisories."
    report = f"""# CyberSecurityAIAgent Security Assessment

## Executive Summary
Risk score: **{risk_score:.0f}/100**. {llm_note or 'The multi-agent workflow reviewed logs, threat intelligence, vulnerability evidence, incident response, and policy controls.'}

## Key Findings
{_format_findings(top_findings, max_items=10)}

## Incident Response Plan
{_format_incident_steps(state.get('incident_steps', []))}

## Policy and Compliance Gaps
{_format_policy_gaps(state.get('policy_gaps', []))}

## Evaluation Loop
{_format_evaluation(evaluation)}

## Authorized Threat Sources
{source_lines}

## Financial Services Guardrails
- Treat this as a decision-support assistant, not an autonomous enforcement system.
- Require human approval before blocking production traffic, rotating privileged credentials, or notifying regulators.
- Preserve audit trails for every agent finding, response decision, and compensating control.
- Do not paste secrets, customer data, or regulated payment data into public deployments.

## Disclaimer
This tool provides defensive security analysis from supplied evidence and configured sources. Validate findings with approved security tooling before production changes.
"""
    return _limit_report_words(report, config.report_max_words)


def _validate_security_report(report: str) -> list[str]:
    required = [
        "## Executive Summary",
        "## Key Findings",
        "## Incident Response Plan",
        "## Policy and Compliance Gaps",
        "## Evaluation Loop",
        "## Authorized Threat Sources",
        "## Disclaimer",
    ]
    issues = [f"missing {section}" for section in required if section not in report]
    if "CVE" in report and not any(domain in report for domain in ("nvd.nist.gov", "cve.org", "cisa.gov", "mitre.org")):
        issues.append("CVE analysis lacks authorized source reference")
    return issues


def _apply_inline_report_edits(report: str, state: ResearchState) -> tuple[str, list[str]]:
    edits: list[str] = []
    revised = report.strip()
    if "# CyberSecurityAIAgent Security Assessment" not in revised:
        revised = f"# CyberSecurityAIAgent Security Assessment\n\n{revised}"
        edits.append("Title: inserted CyberSecurityAIAgent assessment heading.")
    issues = _validate_security_report(revised)
    if issues:
        revised = _render_security_report(state, "Inline revision regenerated missing required report sections.", ResearchConfig())
        edits.append("Sections: regenerated required security report structure.")
    if "## Authorized Threat Sources" in revised and "Tavily" not in revised and "nvd.nist.gov" not in revised:
        edits.append("Sources: verified authorized threat-source section exists for audit review.")
    if not edits:
        edits.append("No inline edits required after security report validation.")
    return revised, edits


def _format_findings(findings: list[SecurityFinding], *, max_items: int) -> str:
    if not findings:
        return "- No security findings were generated from the supplied evidence."
    lines = []
    for finding in findings[:max_items]:
        assets = ", ".join(finding.affected_assets) or "unknown asset"
        lines.append(
            f"- **{finding.severity.upper()}** {finding.id} {finding.title} ({finding.agent}, assets: {assets}). "
            f"Evidence: {_limit_words(finding.evidence, 24)} Fix: {_limit_words(finding.recommended_fix, 24)}"
        )
    return "\n".join(lines)


def _format_incident_steps(steps: list[IncidentStep]) -> str:
    if not steps:
        return "- No incident response steps were generated."
    return "\n".join(
        f"- **{step.phase.replace('_', ' ').title()}** ({step.owner}, {step.priority}): {step.action}"
        for step in steps
    )


def _format_policy_gaps(gaps: list[PolicyGap]) -> str:
    if not gaps:
        return "- No policy gaps were mapped."
    return "\n".join(
        f"- **{gap.framework} {gap.control}**: {gap.status.replace('_', ' ')}. {gap.finding} Remediation: {gap.remediation}"
        for gap in gaps
    )


def _format_evaluation(summary: EvaluationSummary | None) -> str:
    if summary is None:
        return "- Evaluation was not run."
    metric_lines = [
        f"- {metric.name}: {metric.score:.2f} ({'pass' if metric.passed else 'needs work'}). {metric.notes}"
        for metric in summary.metrics
    ]
    action_lines = [f"- {action}" for action in summary.improvement_actions]
    return f"Overall score: **{summary.overall_score:.2f}**\n\nMetrics:\n" + "\n".join(metric_lines) + "\n\nImprovement actions:\n" + "\n".join(action_lines)


def _format_authorized_sources(sources: list[SourceDocument]) -> str:
    authorized = [source for source in sources if _is_authorized_cve_source(source) or source.source_type == "authorized_reference"]
    if not authorized:
        return ""
    return "\n".join(f"- [{source.title}]({source.url})" if source.url else f"- {source.title}" for source in authorized[:6])


def _build_reproducible_snippet(state: ResearchState, evaluation: EvaluationSummary) -> str:
    finding_records = [
        {
            "id": finding.id,
            "agent": finding.agent,
            "severity": finding.severity,
            "category": finding.category,
            "title": finding.title,
        }
        for finding in state.get("findings", [])
    ]
    metric_records = [
        {"name": metric.name, "score": metric.score, "passed": metric.passed}
        for metric in evaluation.metrics
    ]
    return f'''```python
from collections import Counter

findings = {finding_records!r}
metrics = {metric_records!r}

print("Findings by severity:", dict(Counter(item["severity"] for item in findings)))
print("Findings by agent:", dict(Counter(item["agent"] for item in findings)))
print("Evaluation score:", round(sum(item["score"] for item in metrics) / len(metrics), 2))
print("Failed metrics:", [item["name"] for item in metrics if not item["passed"]])
```'''


def _authorized_reference_sources(query: str) -> list[SourceDocument]:
    return [
        SourceDocument(
            title="NVD CVE Vulnerability Database",
            url="https://nvd.nist.gov/vuln",
            source_type="authorized_reference",
            content=(
                "The National Vulnerability Database is an authorized U.S. government source for CVE records, "
                "CVSS scoring, affected configurations, and vulnerability enrichment. Use it to verify CVE impact."
            ),
            metadata={"domain": "nvd.nist.gov", "query": query},
        ),
        SourceDocument(
            title="CISA Known Exploited Vulnerabilities Catalog",
            url="https://www.cisa.gov/known-exploited-vulnerabilities-catalog",
            source_type="authorized_reference",
            content=(
                "CISA KEV lists vulnerabilities known to be exploited in the wild and is an authorized source "
                "for prioritizing urgent remediation."
            ),
            metadata={"domain": "cisa.gov", "query": query},
        ),
        SourceDocument(
            title="CVE.org Records",
            url="https://www.cve.org/",
            source_type="authorized_reference",
            content="CVE.org publishes official CVE identifiers and records from the CVE Program.",
            metadata={"domain": "cve.org", "query": query},
        ),
        SourceDocument(
            title="MITRE ATT&CK Enterprise Matrix",
            url="https://attack.mitre.org/",
            source_type="authorized_reference",
            content="MITRE ATT&CK documents adversary tactics and techniques for detection and response mapping.",
            metadata={"domain": "attack.mitre.org", "query": query},
        ),
    ]


def _assess_source(source: SourceDocument) -> SourceAssessment:
    domain = urlparse(source.url).netloc.lower() if source.url else str(source.metadata.get("domain", "local document"))
    if any(name in domain for name in ("nvd.nist.gov", "cve.org", "cisa.gov", "mitre.org")):
        credibility = "authorized security source"
        caveats = "Use exact affected versions, dates, and vendor context before remediation."
    elif source.source_type == "system_notice":
        credibility = "configuration notice, not threat evidence"
        caveats = "Configure Tavily or upload official advisories for live threat intelligence."
    elif source.url:
        credibility = "open web security source"
        caveats = "Corroborate with NVD, CVE.org, CISA, MITRE, or vendor advisories."
    else:
        credibility = "user-provided local evidence"
        caveats = "Validate provenance and collection integrity before audit use."
    return SourceAssessment(
        source=source.title,
        credibility=credibility,
        relevance="high" if len(source.content) > 300 else "medium",
        caveats=caveats,
    )


def _format_sources(sources: list[SourceDocument], *, max_chars: int) -> str:
    blocks: list[str] = []
    used = 0
    for index, source in enumerate(sources, start=1):
        block = (
            f"[{index}] {source.title}\n"
            f"URL: {source.url or 'local/uploaded'}\n"
            f"Type: {source.source_type}\n"
            f"Content: {source.content.strip()}\n"
        )
        if used + len(block) > max_chars:
            break
        blocks.append(block)
        used += len(block)
    return "\n---\n".join(blocks) if blocks else "No source context available."


def _format_source_inventory(sources: list[SourceDocument], *, max_items: int = 12) -> str:
    counts: Counter[str] = Counter(source.source_type for source in sources)
    count_lines = ", ".join(f"{source_type}: {count}" for source_type, count in sorted(counts.items()))
    source_lines = [
        f"[{index}] {source.title} | type={source.source_type} | url={source.url or 'local/uploaded'}"
        for index, source in enumerate(sources[:max_items], start=1)
    ]
    return f"Source counts: {count_lines or 'none'}\n" + "\n".join(source_lines)


def _with_logs(state: ResearchState, message: str) -> list[str]:
    return [*state.get("logs", []), message]


def _parse_lines(text: str) -> list[str]:
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip().strip("-* ")
        if not line:
            continue
        if len(line) > 2 and line[0].isdigit() and line[1] in {".", ")"}:
            line = line[2:].strip()
        if line and not line.endswith(":"):
            lines.append(line)
    return lines


def _finding(
    *,
    agent: str,
    severity: str,
    category: str,
    title: str,
    evidence: str,
    assets: list[str],
    fix: str,
    refs: list[str],
) -> SecurityFinding:
    prefix = {
        "Log Monitor Agent": "LOG",
        "Threat Intelligence Agent": "TI",
        "Vulnerability Scanner Agent": "VULN",
        "Policy Checker Agent": "POL",
        "Incident Response Agent": "IR",
    }.get(agent, "SEC")
    digest = abs(hash((agent, category, title, evidence[:80]))) % 1000
    return SecurityFinding(
        id=f"{prefix}-{digest:03d}",
        agent=agent,
        severity=severity,  # type: ignore[arg-type]
        category=category,
        title=title,
        evidence=evidence,
        affected_assets=assets,
        recommended_fix=fix,
        confidence=0.82,
        source_refs=refs,
    )


def _merge_findings(existing: list[SecurityFinding], new: list[SecurityFinding]) -> list[SecurityFinding]:
    return _dedupe_findings([*existing, *new])


def _dedupe_findings(findings: list[SecurityFinding]) -> list[SecurityFinding]:
    seen: set[tuple[str, str, str]] = set()
    deduped: list[SecurityFinding] = []
    for finding in findings:
        key = (finding.agent, finding.category, finding.title)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(finding)
    return sorted(deduped, key=lambda finding: SEVERITY_ORDER[finding.severity], reverse=True)


def _dedupe_sources(sources: list[SourceDocument]) -> list[SourceDocument]:
    seen: set[str] = set()
    deduped: list[SourceDocument] = []
    for source in sources:
        key = source.url or f"{source.title}:{source.content[:120]}"
        if key in seen or not source.content.strip():
            continue
        seen.add(key)
        deduped.append(source)
    return deduped


def _snippet(content: str, start: int, width: int = 180) -> str:
    left = max(0, start - 60)
    right = min(len(content), start + width)
    return " ".join(content[left:right].split())


def _first_matching_line(content: str, pattern: str) -> str:
    regex = re.compile(pattern, re.I)
    for line in content.splitlines():
        if regex.search(line):
            return line.strip()[:180]
    return ""


def _looks_like_dockerfile(source: SourceDocument) -> bool:
    title = source.title.lower()
    return title == "dockerfile" or title.endswith("/dockerfile") or "from " in source.content.lower()


def _is_authorized_cve_source(source: SourceDocument) -> bool:
    domain = urlparse(source.url).netloc.lower() if source.url else str(source.metadata.get("domain", "")).lower()
    return any(name in domain for name in ("nvd.nist.gov", "cve.org", "cisa.gov", "mitre.org"))


def _authorized_source_label(sources: list[SourceDocument]) -> str:
    if not sources:
        return "no authorized source in current top-k context"
    return ", ".join(source.title for source in sources[:3])


def _infer_assets(text: str) -> list[str]:
    assets = []
    for keyword in ("log4j", "spring", "openssl", "postgres", "mysql", "mssql", "oracle", "docker", "kubernetes", "api", "linux"):
        if keyword in text.lower():
            assets.append(keyword)
    return assets or ["asset inventory needs confirmation"]


def _affected_assets(findings: list[SecurityFinding]) -> list[str]:
    assets = []
    for finding in findings:
        assets.extend(finding.affected_assets)
    return sorted(set(assets))[:8]


def _max_severity(findings: list[SecurityFinding]) -> str:
    if not findings:
        return "medium"
    return max((finding.severity for finding in findings), key=lambda severity: SEVERITY_ORDER[severity])


def _risk_score(findings: list[SecurityFinding]) -> float:
    if not findings:
        return 0.0
    weights = {"critical": 100, "high": 75, "medium": 45, "low": 20, "informational": 5}
    total = sum(weights[finding.severity] * finding.confidence for finding in findings)
    return min(100.0, round(total / max(1, len(findings) ** 0.6), 1))


def _render_incident_plan(steps: list[IncidentStep], findings: list[SecurityFinding]) -> str:
    return (
        f"Incident plan generated for {len(findings)} finding(s). "
        + " ".join(f"{step.phase}: {step.action}" for step in steps)
    )


def _limit_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]).rstrip(".,;:") + "."


def _limit_report_words(report: str, max_words: int) -> str:
    # Security reports must preserve required sections for auditability. The UI
    # exposes max_words as guidance, but section completeness wins over trimming.
    return report.strip()


SEVERITY_ORDER = {
    "informational": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4,
}

LOG_PATTERNS = [
    (
        r"union\s+select|or\s+1\s*=\s*1|/etc/passwd|sqlmap",
        "high",
        "web_attack",
        "SQL injection pattern detected in logs",
        "Block attack sources, add WAF rules, parameterize database queries, and review affected API endpoints.",
    ),
    (
        r"ransomware|encrypting files|shadowcopy|vssadmin\s+delete",
        "critical",
        "malware",
        "Possible ransomware activity",
        "Isolate affected hosts, preserve forensic images, disable compromised accounts, and invoke the ransomware playbook.",
    ),
    (
        r"large outbound|exfil|data transfer|egress spike|scp .*customer|aws s3 cp .*customer",
        "high",
        "data_exfiltration",
        "Potential data exfiltration signal",
        "Contain egress paths, inspect DLP and proxy logs, rotate exposed credentials, and notify privacy/legal teams.",
    ),
    (
        r"sudo:|privilege escalation|root shell|useradd .*admin",
        "high",
        "privilege_escalation",
        "Suspicious privilege escalation activity",
        "Review privileged sessions, revoke suspicious access, rotate credentials, and validate endpoint integrity.",
    ),
]

VULNERABILITY_PATTERNS = [
    (
        r"(password|passwd|pwd|secret|api[_-]?key)\s*[:=]\s*['\"][^'\"\n]{6,}",
        "high",
        "secret_management",
        "Hard-coded secret or credential",
        "Move secrets to an approved vault, rotate exposed values, and add pre-commit secret scanning.",
    ),
    (
        r"debug\s*[:=]\s*(true|1|yes)",
        "medium",
        "insecure_runtime",
        "Debug mode enabled",
        "Disable debug mode in production and gate diagnostics behind authenticated break-glass workflows.",
    ),
    (
        r"listen_addresses\s*=\s*['\"]?0\.0\.0\.0|bind-address\s*=\s*0\.0\.0\.0|host\s+all\s+all\s+0\.0\.0\.0/0",
        "high",
        "database_hardening",
        "Database listener exposed broadly",
        "Bind databases to private interfaces, restrict security groups, and require network segmentation.",
    ),
    (
        r"ssl\s*=\s*off|sslmode\s*=\s*disable|encrypt\s*=\s*false|trustServerCertificate\s*=\s*true",
        "high",
        "database_hardening",
        "Database transport encryption disabled",
        "Require TLS for database connections and validate certificates for MSSQL, MySQL, Oracle, and Postgres clients.",
    ),
    (
        r"MYSQL_ALLOW_EMPTY_PASSWORD\s*=\s*(yes|true|1)|POSTGRES_HOST_AUTH_METHOD\s*=\s*trust|SA_PASSWORD\s*=\s*['\"]?(password|Password123|P@ssw0rd)",
        "critical",
        "database_hardening",
        "Unsafe database authentication setting",
        "Disable trust or empty-password modes, rotate admin passwords, and enforce managed secret injection.",
    ),
    (
        r"privileged\s*:\s*true|--privileged|cap_add\s*:\s*\[?SYS_ADMIN",
        "high",
        "container_hardening",
        "Privileged container execution",
        "Remove privileged mode, drop capabilities, use read-only filesystems, and apply runtime policies.",
    ),
    (
        r"SELECT\s+.*\+|execute\(.{0,80}\+|cursor\.execute\(.*%",
        "high",
        "code_security",
        "Possible dynamic SQL construction",
        "Use parameterized queries and add tests for injection payloads on API and database paths.",
    ),
]

KNOWN_CVE_ALIASES = {
    "log4shell": "CVE-2021-44228",
    "log4j": "CVE-2021-44228",
    "spring4shell": "CVE-2022-22965",
    "moveit": "CVE-2023-34362",
    "regresshion": "CVE-2024-6387",
}
