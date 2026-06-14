"""Configuration and hackathon concept metadata."""

from deep_researcher.config.concepts import (
    coverage_table_rows,
    load_concepts,
    score_concept_coverage,
)
from deep_researcher.config.settings import OPENROUTER_BASE_URL, ResearchConfig

__all__ = [
    "OPENROUTER_BASE_URL",
    "ResearchConfig",
    "coverage_table_rows",
    "load_concepts",
    "score_concept_coverage",
]
