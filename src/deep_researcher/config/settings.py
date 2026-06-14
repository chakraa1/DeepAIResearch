"""Runtime configuration for the deep research assistant."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass(slots=True)
class ResearchConfig:
    """Configuration values used across agents."""

    openai_api_key: str | None = None
    tavily_api_key: str | None = None
    llm_provider: str = "openrouter"
    openai_base_url: str | None = OPENROUTER_BASE_URL
    openai_model: str = "openai/gpt-4o-mini"
    max_web_results: int = 8
    max_retrieval_docs: int = 3
    validator_top_k: int = 3
    critical_analysis_word_limit: int = 200
    insight_word_limit: int = 200
    report_min_words: int = 200
    report_max_words: int = 300
    generate_code_snippet: bool = True
    report_reflection_retry_limit: int = 2
    require_human_review: bool = False
    checkpoint_thread_id: str = "deep-research-default"
    chunk_size: int = 1_000
    chunk_overlap: int = 150

    @classmethod
    def from_env(cls) -> "ResearchConfig":
        """Load settings from environment variables and a local .env file."""

        load_dotenv()
        provider = _normalize_provider(os.getenv("LLM_PROVIDER", "openrouter"))
        base_url = _resolve_base_url(provider, os.getenv("OPENAI_BASE_URL") or None)
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY") or None,
            tavily_api_key=os.getenv("TAVILY_API_KEY") or None,
            llm_provider=provider,
            openai_base_url=base_url,
            openai_model=os.getenv("OPENAI_MODEL") or _default_model(provider),
            max_web_results=_read_int("MAX_WEB_RESULTS", 8),
            max_retrieval_docs=_read_int("MAX_RETRIEVAL_DOCS", 3),
            validator_top_k=_read_int("VALIDATOR_TOP_K", 3),
            critical_analysis_word_limit=_read_int("CRITICAL_ANALYSIS_WORD_LIMIT", 200),
            insight_word_limit=_read_int("INSIGHT_WORD_LIMIT", 200),
            report_min_words=_read_int("REPORT_MIN_WORDS", 200),
            report_max_words=_read_int("REPORT_MAX_WORDS", 300),
            generate_code_snippet=_read_bool("GENERATE_CODE_SNIPPET", True),
            report_reflection_retry_limit=_read_int("REPORT_REFLECTION_RETRY_LIMIT", 2),
            require_human_review=_read_bool("REQUIRE_HUMAN_REVIEW", False),
            checkpoint_thread_id=os.getenv("CHECKPOINT_THREAD_ID", "deep-research-default"),
        )

    @property
    def llm_enabled(self) -> bool:
        return bool(self.openai_api_key)

    @property
    def web_search_enabled(self) -> bool:
        return bool(self.tavily_api_key)

    @property
    def llm_base_url(self) -> str | None:
        return _resolve_base_url(self.llm_provider, self.openai_base_url)

    @property
    def uses_direct_openai_api(self) -> bool:
        base_url = self.llm_base_url
        return self.llm_provider == "openai" and (
            not base_url or "api.openai.com" in base_url
        )


def _read_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _read_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _normalize_provider(provider: str | None) -> str:
    normalized = (provider or "openai").strip().lower()
    if normalized in {"openrouter", "open-router"}:
        return "openrouter"
    if normalized in {"custom", "custom_openai", "custom-openai"}:
        return "custom"
    return "openai"


def _resolve_base_url(provider: str, base_url: str | None) -> str | None:
    provider = _normalize_provider(provider)
    if provider == "openrouter":
        return base_url or OPENROUTER_BASE_URL
    if provider == "custom":
        return base_url
    return base_url or None


def _default_model(provider: str) -> str:
    if _normalize_provider(provider) == "openrouter":
        return "openai/gpt-4o-mini"
    return "gpt-4o-mini"
