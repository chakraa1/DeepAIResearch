"""FAISS retrieval utilities."""

from __future__ import annotations

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from deep_researcher.config import ResearchConfig
from deep_researcher.embeddings import build_embeddings
from deep_researcher.models import SourceDocument


def split_sources(
    sources: list[SourceDocument],
    config: ResearchConfig,
) -> list[SourceDocument]:
    """Split long source documents into retrievable chunks."""

    if not sources:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    documents = [
        Document(
            page_content=source.content,
            metadata={
                "title": source.title,
                "url": source.url,
                "source_type": source.source_type,
                **source.metadata,
            },
        )
        for source in sources
        if source.content.strip()
    ]
    chunks = splitter.split_documents(documents)
    return [
        SourceDocument(
            title=chunk.metadata.get("title", "Untitled source"),
            url=chunk.metadata.get("url"),
            source_type=chunk.metadata.get("source_type", "document"),
            content=chunk.page_content,
            metadata=dict(chunk.metadata),
        )
        for chunk in chunks
    ]


def retrieve_relevant_context(
    query: str,
    sources: list[SourceDocument],
    config: ResearchConfig,
) -> list[SourceDocument]:
    """Build an in-memory FAISS index and return the top matching chunks."""

    chunks = split_sources(sources, config)
    if not chunks:
        return []

    documents = [
        Document(
            page_content=chunk.content,
            metadata={
                "title": chunk.title,
                "url": chunk.url,
                "source_type": chunk.source_type,
                **chunk.metadata,
            },
        )
        for chunk in chunks
    ]
    vector_store = FAISS.from_documents(documents, build_embeddings(config))
    matches = vector_store.similarity_search_with_score(
        query,
        k=min(config.max_retrieval_docs, len(documents)),
    )

    return [
        SourceDocument(
            title=document.metadata.get("title", "Untitled source"),
            url=document.metadata.get("url"),
            source_type=document.metadata.get("source_type", "document"),
            content=document.page_content,
            score=float(score),
            metadata=dict(document.metadata),
        )
        for document, score in matches
    ]
