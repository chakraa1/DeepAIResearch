from __future__ import annotations

import pytest

from deep_researcher.config import ResearchConfig
from deep_researcher.embeddings import HashEmbeddings
from deep_researcher.models import SourceDocument
from deep_researcher.prompts import get_system_prompt, render_system_prompt
from deep_researcher.search import parallel_tavily_search, tavily_search
from deep_researcher.workflow import DeepResearchWorkflow


def test_hash_embeddings_are_deterministic() -> None:
    embedder = HashEmbeddings(dimensions=32)

    first = embedder.embed_query("AI agents improve research workflows")
    second = embedder.embed_query("AI agents improve research workflows")

    assert first == second
    assert len(first) == 32


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
    )

    assert result["report"]
    assert result["report"].splitlines()[0].startswith("**")
    assert "## SOURCES" in result["report"]
    assert len(result["retrieved_context"]) <= 3
    assert result["retrieved_context"]
    assert result["relevant_context_summary"]
    assert any("Contextual Retriever prompt plan:" in log for log in result["logs"])
    assert any("Tuning to Relevant Context completed:" in log for log in result["logs"])
    assert result["logs"][-1] == "Report Builder Agent completed: compiled the final rules-checked report."
