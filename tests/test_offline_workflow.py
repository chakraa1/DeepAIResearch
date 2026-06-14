from __future__ import annotations

import pytest

from deep_researcher.config import OPENROUTER_BASE_URL, ResearchConfig
from deep_researcher.config.concepts import coverage_table_rows, load_concepts, score_concept_coverage
from deep_researcher.main.models import ResearchPlan, SourceDocument
from deep_researcher.main.workflow import DeepResearchWorkflow
from deep_researcher.agent.prompts import get_system_prompt, render_system_prompt
from deep_researcher.tools.embeddings import HashEmbeddings, build_embeddings
from deep_researcher.tools.search import parallel_tavily_search, tavily_search
from deep_researcher.tools import build_default_tool_registry


def test_hash_embeddings_are_deterministic() -> None:
    embedder = HashEmbeddings(dimensions=32)

    first = embedder.embed_query("AI agents improve research workflows")
    second = embedder.embed_query("AI agents improve research workflows")

    assert first == second
    assert len(first) == 32


def test_concepts_json_loads_and_scores_alignment() -> None:
    concepts = load_concepts()
    score = score_concept_coverage()
    rows = coverage_table_rows()

    assert len(concepts) == 17
    assert score.chapter == "Orion Tutorial - Consolidated Agent Curriculum and Design Patterns"
    assert score.max_score == 10
    assert 0 < score.score <= 10
    assert score.implemented_count >= 16
    assert score.partial_count == 1
    assert len(rows) == 17
    assert any(row["Concept"] == "1.3 Agent Graph & Smart Routing" for row in rows)
    assert not any(row["Concept"] == "1.5 Code Generation" for row in rows)
    for concept in (
        "1.2 Tools & Shell Command Execution",
        "1.4 Structured Planning",
        "2.2 AI Code Review with Retry Limit",
        "2.4 Inline Edit",
        "3.4 Human Approval Gate",
        "3.6 State Checkpointing & Time Travel",
    ):
        row = next(row for row in rows if row["Concept"] == concept)
        assert row["Status"] == "Implemented"


def test_safe_tool_registry_allows_known_tools_and_rejects_unknown() -> None:
    registry = build_default_tool_registry(ResearchConfig(tavily_api_key=None))
    tool_names = {tool["name"] for tool in registry.list_tools()}

    assert {"parallel_tavily_search", "faiss_top_k_retrieval", "markdown_report_export"} <= tool_names
    export = registry.execute("markdown_report_export", report="# Report")
    assert export["filename"] == "deep_research_report.md"
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


def test_openrouter_uses_local_hash_embeddings() -> None:
    config = ResearchConfig(
        openai_api_key="sk-or-v1-test",
        llm_provider="openrouter",
        openai_base_url=OPENROUTER_BASE_URL,
    )

    assert isinstance(build_embeddings(config), HashEmbeddings)


def test_tavily_search_returns_notice_without_key() -> None:
    config = ResearchConfig(tavily_api_key=None)

    results = tavily_search("AI safety", config)

    assert results
    assert results[0].source_type == "system_notice"
    assert "TAVILY_API_KEY" in results[0].content


def test_parallel_tavily_search_runs_all_source_lanes_without_key() -> None:
    config = ResearchConfig(tavily_api_key=None)

    results = parallel_tavily_search("AI safety", config)

    lanes = {result.metadata["source_lane"] for result in results}
    assert lanes == {"research_papers", "news_articles", "reports", "apis"}


def test_cached_yaml_prompt_includes_explicit_role() -> None:
    prompt = get_system_prompt("report_builder")

    assert "Role: Report Builder Agent" in prompt
    assert "SOURCE_LINKS" in prompt


def test_reproducible_snippet_prompt_requires_python_code() -> None:
    prompt = get_system_prompt("reproducible_snippet")

    assert "Role: Reproducible Snippet Agent" in prompt
    assert "fenced python code block" in prompt


def test_report_revision_prompt_targets_inline_markdown_edits() -> None:
    prompt = get_system_prompt("report_revision")

    assert "Role: Report Revision Inline Edit Agent" in prompt
    assert "targeted section-level edits" in prompt
    assert "Do not rewrite the whole report" in prompt


def test_contextual_retriever_prompt_renders_query_placeholders() -> None:
    prompt = render_system_prompt(
        "contextual_retriever",
        query="enterprise AI agent adoption",
        source_lanes="research_papers, news_articles, reports, apis",
        top_k=3,
    )

    assert "Role: Contextual Retriever Agent" in prompt
    assert "Input query: enterprise AI agent adoption" in prompt
    assert "Source lanes: research_papers, news_articles, reports, apis" in prompt
    assert "FAISS top-k passed to LLM agents: 3" in prompt
    assert "{query}" not in prompt


def test_source_selector_prompt_renders_retrieved_context_sources() -> None:
    retrieved_context = """
[1] Agent Adoption Report
URL: https://example.com/report
Type: reports
Content: Evaluation quality is a recurring adoption bottleneck.
""".strip()

    prompt = render_system_prompt(
        "source_selector",
        query="enterprise AI agent adoption",
        top_k=3,
        relevant_context_summary="Reports and uploaded files agree that evaluation quality is the key bottleneck.",
        retrieved_context=retrieved_context,
    )

    assert "Role: FAISS Context Selector" in prompt
    assert "Research question: enterprise AI agent adoption" in prompt
    assert "FAISS top-k limit: 3" in prompt
    assert "Summarised relevant context from all sources:" in prompt
    assert "evaluation quality is the key bottleneck" in prompt
    assert "Retrieved context with sources:" in prompt
    assert "Agent Adoption Report" in prompt
    assert "https://example.com/report" in prompt
    assert "Evaluation quality is a recurring adoption bottleneck." in prompt
    assert "{retrieved_context}" not in prompt
    assert "{relevant_context_summary}" not in prompt


def test_relevant_context_tuner_prompt_combines_all_sources_and_faiss_top_k() -> None:
    prompt = render_system_prompt(
        "relevant_context_tuner",
        query="enterprise AI agent adoption",
        source_inventory="Source counts: reports: 1, uploaded_file: 1",
        faiss_top_context="[1] Agent Adoption Report\nContent: Oversight is required.",
    )

    assert "Role: Tuning to Relevant Context Agent" in prompt
    assert "Research question: enterprise AI agent adoption" in prompt
    assert "Source counts: reports: 1, uploaded_file: 1" in prompt
    assert "Agent Adoption Report" in prompt
    assert "Oversight is required." in prompt
    assert "{source_inventory}" not in prompt


def test_offline_workflow_generates_report_from_local_sources() -> None:
    pytest.importorskip("faiss")
    pytest.importorskip("langgraph")

    config = ResearchConfig(
        openai_api_key=None,
        tavily_api_key=None,
        max_web_results=1,
        max_retrieval_docs=3,
        validator_top_k=3,
    )
    workflow = DeepResearchWorkflow(config)
    documents = [
        SourceDocument(
            title="AI Agent Adoption Note",
            source_type="test_document",
            content=(
                "AI agents are being adopted for research, customer support, and software automation. "
                "However, organizations report risks around evaluation, source quality, and oversight. "
                "Recent pilots emphasize retrieval augmented generation, audit trails, and human review."
            ),
        )
    ]

    result = workflow.run(
        "What trends and risks shape enterprise AI agent adoption?",
        local_documents=documents,
        thread_id="test-thread-inline",
    )

    assert result["report"]
    assert result["thread_id"] == "test-thread-inline"
    assert isinstance(result["research_plan"], ResearchPlan)
    assert result["research_plan"].sub_questions
    assert result["report"].splitlines()[0].startswith("**")
    assert "## SOURCES" in result["report"]
    assert len(result["retrieved_context"]) <= 3
    assert result["retrieved_context"]
    assert result["relevant_context_summary"]
    assert result["reproducible_snippet"].startswith("```python")
    assert "sources =" in result["reproducible_snippet"]
    assert result["human_review_decision"] == "auto-approved"
    assert result["report_reflection_notes"]
    assert result["report_reflection_attempts"] >= 0
    assert result["report_revision_edits"]
    assert any("Sources section:" in edit for edit in result["report_revision_edits"])
    assert any("Contextual Retriever prompt plan:" in log for log in result["logs"])
    assert any("Tuning to Relevant Context completed:" in log for log in result["logs"])
    assert any("Reproducible Snippet Agent completed:" in log for log in result["logs"])
    assert any("Human Review Gate completed:" in log for log in result["logs"])
    assert any("Report Reflection Agent completed:" in log for log in result["logs"])
    assert result["logs"][-2].startswith("Report Revision Agent completed:")
