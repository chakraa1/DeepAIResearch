"""Runtime configuration for the deep research assistant."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(slots=True)
class ResearchConfig:
    """Configuration values used across agents."""

    openai_api_key: str | None = None
    tavily_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    max_web_results: int = 5
    max_retrieval_docs: int = 8
    chunk_size: int = 1_000
    chunk_overlap: int = 150

    @classmethod
    def from_env(cls) -> "ResearchConfig":
        """Load settings from environment variables and a local .env file."""

        load_dotenv()
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY") or None,
            tavily_api_key=os.getenv("TAVILY_API_KEY") or None,
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            max_web_results=_read_int("MAX_WEB_RESULTS", 5),
            max_retrieval_docs=_read_int("MAX_RETRIEVAL_DOCS", 8),
        )

    @property
    def llm_enabled(self) -> bool:
        return bool(self.openai_api_key)

    @property
    def web_search_enabled(self) -> bool:
        return bool(self.tavily_api_key)


def _read_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default
