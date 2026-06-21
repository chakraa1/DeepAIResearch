from __future__ import annotations

import pytest

from deep_researcher.config import OPENROUTER_BASE_URL, ResearchConfig
from deep_researcher.config.concepts import coverage_table_rows, load_concepts, score_concept_coverage
from deep_researcher.main.evaluation import evaluate_state, run_offline_evaluation
from deep_researcher.main.models import ResearchPlan, SourceDocument
from deep_researcher.main.workflow import DeepResearchWorkflow
from deep_researcher.agent.prompts import get_system_prompt, render_system_prompt
from deep_researcher.tools.embeddings import HashEmbeddings, build_embeddings
from deep_researcher.tools.search import SOURCE_SEARCH_QUERIES, parallel_tavily_search, tavily_search
from deep_researcher.tools import build_default_tool_registry


def test_hash_embeddings_are_deterministic() -> None:
    embedder = HashEmbeddings(dimensions=32)

    first = embedder.embed_query("CVE brute force Docker database hardening")
    second = embedder.embed_query("CVE brute force Docker database hardening")

    assert first == second
    assert len(first) == 32


def test_concepts_json_loads_and_scores_security_alignment() -> None:
    concepts = load_concepts()
    score = score_concept_coverage()
    rows = coverage_table_rows()

    assert len(concepts) == 17
    assert score.chapter == "Orion Tutorial - Consolidated Agent Curriculum and Design Patterns"
    assert score.max_score == 10
    assert 0 < score.score <= 10
    assert score.implemented_count >= 16
    assert len(rows) == 17
    assert any("Threat Intelligence Agent" in row["Evidence"] for row in rows)
    assert any(row["Concept"] == "1.3 Agent Graph & Smart Routing" for row in rows)


def test_safe_tool_registry_allows_known_tools_and_rejects_unknown() -> None:
    registry = build_default_tool_registry(ResearchConfig(tavily_api_key=None))
    tool_names = {tool["name"] for tool in registry.list_tools()}

    assert {"parallel_tavily_search", "faiss_top_k_retrieval", "markdown_report_export"} <= tool_names
    export = registry.execute("markdown_report_export", report="# Report")
    assert export["filename"] == "cybersecurity_ai_agent_report.md"
    with pytest.raises(KeyError):
        registry.execute("shell", command="rm -rf /")


def test_openrouter_env_sets_base_url_and_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-or-v1-test")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    config = ResearchConfig.from_env()

    assert config.llm_provider == "openrouter"
    assert config.llm_base_url == OPENROUTER_BASE_URL
    assert config.openai_model == "openai/gpt-4o-mini"
    assert not config.uses_direct_openai_api


def test_default_config_uses_cybersecurity_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("CHECKPOINT_THREAD_ID", raising=False)

    config = ResearchConfig.from_env()

    assert config.llm_provider == "openrouter"
    assert config.llm_base_url == OPENROUTER_BASE_URL
    assert config.openai_model == "openai/gpt-4o-mini"
    assert config.checkpoint_thread_id == "cybersecurity-agent-default"
    assert config.report_max_words == 650


def test_openrouter_uses_local_hash_embeddings() -> None:
    config = ResearchConfig(
        openai_api_key="sk-or-v1-test",
        llm_provider="openrouter",
        openai_base_url=OPENROUTER_BASE_URL,
    )

    assert isinstance(build_embeddings(config), HashEmbeddings)


def test_tavily_search_returns_notice_without_key() -> None:
    config = ResearchConfig(tavily_api_key=None)

    results = tavily_search("Log4j CVE-2021-44228", config)

    assert results
    assert results[0].source_type == "system_notice"
    assert "TAVILY_API_KEY" in results[0].content
    assert "NVD" in results[0].content


def test_parallel_tavily_search_runs_authorized_security_lanes_without_key() -> None:
    config = ResearchConfig(tavily_api_key=None)

    results = parallel_tavily_search("Log4j CVE-2021-44228", config)

    lanes = {result.metadata["source_lane"] for result in results}
    assert lanes == set(SOURCE_SEARCH_QUERIES)
    assert {"nvd_cve", "cve_org", "cisa_kev", "mitre_attack"} <= lanes


def test_cached_yaml_prompts_include_security_roles() -> None:
    assert "Role: Security Planning Agent" in get_system_prompt("security_planner")
    assert "Role: Threat Intelligence Agent" in get_system_prompt("threat_intelligence")
    assert "authorized CVE" in get_system_prompt("threat_intelligence")
    assert "Role: Security Report Builder Agent" in get_system_prompt("security_report_builder")
    assert "Do not provide exploit instructions" in get_system_prompt("security_report_builder")


def test_security_context_tuner_prompt_renders_placeholders() -> None:
    prompt = render_system_prompt(
        "security_context_tuner",
        query="internet banking CVE exposure",
        source_inventory="Source counts: nvd_cve: 1, log: 1",
        faiss_top_context="[1] NVD CVE Vulnerability Database\nContent: CVE records and CVSS scoring.",
    )

    assert "Role: Security Context Tuning Agent" in prompt
    assert "internet banking CVE exposure" in prompt
    assert "Source counts: nvd_cve: 1, log: 1" in prompt
    assert "CVE records and CVSS scoring" in prompt
    assert "{source_inventory}" not in prompt


def test_evaluate_state_scores_policy_and_incident_coverage() -> None:
    state = {
        "query": "Assess CVE-2021-44228",
        "findings": [],
        "policy_gaps": [],
        "incident_steps": [],
        "retrieved_context": [
            SourceDocument(
                title="NVD CVE Vulnerability Database",
                url="https://nvd.nist.gov/vuln/detail/CVE-2021-44228",
                content="Authorized CVE source.",
                source_type="nvd_cve",
            )
        ],
    }

    summary = evaluate_state(state)

    assert summary.overall_score < 1
    assert any(metric.name == "authorized_cve_sources" and metric.passed for metric in summary.metrics)
    assert summary.improvement_actions


def test_offline_workflow_generates_security_report_from_local_evidence() -> None:
    pytest.importorskip("faiss")
    pytest.importorskip("langgraph")

    config = ResearchConfig(
        openai_api_key=None,
        tavily_api_key=None,
        max_web_results=2,
        max_retrieval_docs=3,
        validator_top_k=3,
        report_max_words=650,
    )
    workflow = DeepResearchWorkflow(config)
    documents = [
        SourceDocument(
            title="auth-api.log",
            source_type="log",
            content="""2026-06-20 failed password for admin from 203.0.113.44
2026-06-20 failed password for admin from 203.0.113.44
2026-06-20 failed password for admin from 203.0.113.44
GET /accounts?id=1 UNION SELECT card_number FROM cards
""",
        ),
        SourceDocument(
            title="Dockerfile",
            source_type="dockerfile",
            content="""FROM python:3.12
ENV MYSQL_ALLOW_EMPTY_PASSWORD=yes
ENV DEBUG=true
CMD ["python", "app.py"]
""",
        ),
        SourceDocument(
            title="postgresql.conf",
            source_type="database_config",
            content="""listen_addresses='0.0.0.0'
ssl = off
""",
        ),
    ]

    result = workflow.run(
        "Assess internet banking logs, Docker, Postgres, and Log4j CVE-2021-44228 exposure.",
        local_documents=documents,
        thread_id="test-cybersecurity-thread",
    )

    assert result["thread_id"] == "test-cybersecurity-thread"
    assert isinstance(result["research_plan"], ResearchPlan)
    assert result["research_plan"].sub_questions
    assert result["report"].startswith("# CyberSecurityAIAgent Security Assessment")
    for section in (
        "## Executive Summary",
        "## Key Findings",
        "## Incident Response Plan",
        "## Policy and Compliance Gaps",
        "## Evaluation Loop",
        "## Authorized Threat Sources",
        "## Disclaimer",
    ):
        assert section in result["report"]
    agents = {finding.agent for finding in result["findings"]}
    assert "Log Monitor Agent" in agents
    assert "Threat Intelligence Agent" in agents
    assert "Vulnerability Scanner Agent" in agents
    assert "Incident Response Agent" in agents
    assert "Policy Checker Agent" in agents
    assert any("brute-force" in finding.title.lower() or "sql injection" in finding.title.lower() for finding in result["findings"])
    assert any("database" in finding.category or "container" in finding.category for finding in result["findings"])
    assert result["incident_steps"]
    assert result["policy_gaps"]
    assert result["evaluation_summary"].overall_score > 0
    assert result["reproducible_snippet"].startswith("```python")
    assert result["human_review_decision"] == "auto-approved"
    assert result["report_reflection_notes"]
    assert result["report_revision_edits"]
    assert any("Threat Intelligence Agent completed" in log for log in result["logs"])
    assert any("Evaluation Loop Agent completed" in log for log in result["logs"])


def test_offline_eval_suite_runs_cases() -> None:
    pytest.importorskip("faiss")
    pytest.importorskip("langgraph")

    rows = run_offline_evaluation(
        ResearchConfig(openai_api_key=None, tavily_api_key=None, max_web_results=1, max_retrieval_docs=3)
    )

    assert len(rows) >= 2
    assert all("score" in row for row in rows)
    assert any(row["passed"] for row in rows)
