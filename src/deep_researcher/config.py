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
    max_web_results: int = 8
    max_retrieval_docs: int = 3
    validator_top_k: int = 3
    critical_analysis_word_limit: int = 200
    insight_word_limit: int = 200
    report_min_words: int = 200
    report_max_words: int = 300
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
            max_web_results=_read_int("MAX_WEB_RESULTS", 8),
            max_retrieval_docs=_read_int("MAX_RETRIEVAL_DOCS", 3),
            validator_top_k=_read_int("VALIDATOR_TOP_K", 3),
            critical_analysis_word_limit=_read_int("CRITICAL_ANALYSIS_WORD_LIMIT", 200),
            insight_word_limit=_read_int("INSIGHT_WORD_LIMIT", 200),
            report_min_words=_read_int("REPORT_MIN_WORDS", 200),
            report_max_words=_read_int("REPORT_MAX_WORDS", 300),
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
