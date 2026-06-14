from __future__ import annotations

import pytest

from deep_researcher.config import ResearchConfig
from deep_researcher.embeddings import HashEmbeddings
from deep_researcher.models import SourceDocument
from deep_researcher.search import tavily_search
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


def test_offline_workflow_generates_report_from_local_sources() -> None:
    pytest.importorskip("faiss")
    pytest.importorskip("langgraph")

    config = ResearchConfig(
        openai_api_key=None,
        tavily_api_key=None,
        max_web_results=1,
        max_retrieval_docs=4,
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
    assert "Deep Research Report" in result["report"] or "Executive Summary" in result["report"]
    assert result["retrieved_context"]
    assert result["logs"][-1] == "Report Builder Agent compiled the final report."
