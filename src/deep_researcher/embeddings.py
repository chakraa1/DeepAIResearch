"""Embedding helpers.

The app prefers OpenAI embeddings when an API key is configured, but a small
deterministic hash embedder keeps FAISS retrieval usable for offline demos and
tests.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

from langchain_core.embeddings import Embeddings

from deep_researcher.config import ResearchConfig

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


class HashEmbeddings(Embeddings):
    """Simple deterministic bag-of-words embeddings for local fallback mode."""

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = TOKEN_RE.findall(text.lower())
        counts = Counter(tokens)

        for token, count in counts.items():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = -1.0 if digest[4] % 2 else 1.0
            vector[index] += sign * (1.0 + math.log(count))

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]


def build_embeddings(config: ResearchConfig) -> Embeddings:
    """Create the best available embedding implementation."""

    if config.openai_api_key:
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(api_key=config.openai_api_key)

    return HashEmbeddings()
