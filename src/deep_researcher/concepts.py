"""Hackathon concept coverage helpers.

The scoring here is intentionally explicit: each accelerator concept is mapped
to implementation evidence so judges can see why a point is claimed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import Literal

ConceptStatus = Literal["implemented", "partial", "not_targeted"]


@dataclass(frozen=True, slots=True)
class Concept:
    id: str
    title: str
    design_pattern: str
    description: str
    importance: str


@dataclass(frozen=True, slots=True)
class ConceptCoverage:
    concept: Concept
    status: ConceptStatus
    score_weight: float
    evidence: str
    improvement: str


@dataclass(frozen=True, slots=True)
class ConceptScore:
    chapter: str
    score: float
    max_score: int
    implemented_count: int
    partial_count: int
    total_count: int
    coverage: tuple[ConceptCoverage, ...]


COVERAGE_MAP: dict[str, tuple[ConceptStatus, str, str]] = {
    "1_1_llm_setup": (
        "implemented",
        "ResearchLLM initializes ChatOpenAI with OpenAI/OpenRouter/custom provider config and temperature.",
        "Expose more model hyperparameters such as max tokens and top-p.",
    ),
    "1_2_tools": (
        "partial",
        "The app uses Tavily, FAISS, PDF parsing, and file upload capabilities, but agents do not call shell subprocess tools.",
        "Add an explicit safe tool registry for source fetchers or report exporters.",
    ),
    "1_3_agent_graph": (
        "implemented",
        "DeepResearchWorkflow builds a LangGraph node network for planner, retriever, validator, analysis, insights, and report builder.",
        "Add conditional edges for retry or skip behavior based on state quality.",
    ),
    "1_4_structured_planning": (
        "partial",
        "Query Planning Agent creates sub-questions before retrieval, but it does not use with_structured_output yet.",
        "Return a typed Pydantic planning object from the planner agent.",
    ),
    "1_6_system_prompt": (
        "implemented",
        "System prompts are centralized in system_prompts.yaml with explicit roles and runtime placeholders.",
        "Add prompt version metadata and per-agent prompt tests.",
    ),
    "1_7_streaming": (
        "implemented",
        "DeepResearchWorkflow.stream emits graph node updates that Streamlit renders step by step.",
        "Add token-level streaming when provider APIs support it.",
    ),
    "1_8_multi_turn": (
        "partial",
        "Streamlit session_state stores the last report and state for a session, but durable conversation memory is not implemented.",
        "Persist research sessions and allow follow-up questions over prior state.",
    ),
    "2_1_structured_output": (
        "implemented",
        "Pydantic models and TypedDict state define source documents, assessments, and workflow state.",
        "Use structured LLM output for planner and source validation responses.",
    ),
    "2_2_self_correction_reflection": (
        "partial",
        "Report Builder applies deterministic rule enforcement after LLM generation, but there is no graph retry loop.",
        "Add a reflection node with retry counters for report validation failures.",
    ),
    "2_3_dynamic_rules": (
        "implemented",
        "Report word limits, top-k limits, provider selection, and report guardrails are config-driven and state-aware.",
        "Allow users to choose different report rule profiles from the UI.",
    ),
    "2_4_inline_edit": (
        "implemented",
        "Report Revision Agent applies targeted Markdown edits to the opening hook, body guardrails, and ## SOURCES section after report generation.",
        "Expose user-selected revision targets such as hook-only, sources-only, or length-only edits.",
    ),
    "3_1_codebase_rag": (
        "implemented",
        "FAISS retrieval indexes uploaded documents and Tavily results for grounded context selection.",
        "Add optional repository/code indexing for technical research tasks.",
    ),
    "3_2_orchestrator_state": (
        "implemented",
        "ResearchState carries sub-questions, sources, tuned context, assessments, synthesis, insights, report, and logs across agents.",
        "Persist state snapshots for comparison across runs.",
    ),
    "3_3_multi_agent": (
        "implemented",
        "Specialized LangGraph nodes divide scope across planning, retrieval, validation, analysis, insight generation, and reporting.",
        "Add a dedicated contradiction matrix agent.",
    ),
    "3_4_human_in_the_loop": (
        "partial",
        "The UI exposes transparent logs and downloadable reports, but no interrupt approval gate blocks graph execution.",
        "Add a review gate before Report Builder using LangGraph interrupts.",
    ),
    "3_5_parallel_generation": (
        "implemented",
        "Contextual Retriever runs parallel Tavily source lanes for papers, news, reports, and APIs before merging sources.",
        "Use LangGraph Send/reducers for graph-native parallel branches.",
    ),
    "3_6_time_travel": (
        "partial",
        "Session state keeps the latest workflow result, but LangGraph MemorySaver checkpointing is not enabled.",
        "Add MemorySaver checkpointer and thread IDs for replayable research runs.",
    ),
}


@lru_cache(maxsize=1)
def load_concept_payload() -> dict:
    concept_file = resources.files("deep_researcher").joinpath("concepts.json")
    with concept_file.open("r", encoding="utf-8") as file:
        return json.load(file)


@lru_cache(maxsize=1)
def load_concepts() -> tuple[Concept, ...]:
    payload = load_concept_payload()
    return tuple(
        Concept(
            id=item["id"],
            title=item["title"],
            design_pattern=item["design_pattern"],
            description=item["description"],
            importance=item["importance"],
        )
        for item in payload["concepts"]
    )


def score_concept_coverage() -> ConceptScore:
    concepts = load_concepts()
    coverage = tuple(_coverage_for(concept) for concept in concepts)
    raw_score = sum(item.score_weight for item in coverage)
    score = round((raw_score / len(coverage)) * 10, 1) if coverage else 0.0
    return ConceptScore(
        chapter=load_concept_payload()["chapter"],
        score=score,
        max_score=10,
        implemented_count=sum(item.status == "implemented" for item in coverage),
        partial_count=sum(item.status == "partial" for item in coverage),
        total_count=len(coverage),
        coverage=coverage,
    )


def coverage_table_rows() -> list[dict[str, str | float]]:
    score = score_concept_coverage()
    return [
        {
            "Concept": item.concept.title,
            "Pattern": item.concept.design_pattern,
            "Importance": item.concept.importance,
            "Status": item.status.replace("_", " ").title(),
            "Weight": item.score_weight,
            "Evidence": item.evidence,
            "Next improvement": item.improvement,
        }
        for item in score.coverage
    ]


def _coverage_for(concept: Concept) -> ConceptCoverage:
    status, evidence, improvement = COVERAGE_MAP.get(
        concept.id,
        (
            "not_targeted",
            "No implementation evidence has been mapped for this concept.",
            "Add an explicit feature and evidence mapping.",
        ),
    )
    return ConceptCoverage(
        concept=concept,
        status=status,
        score_weight=_score_weight(status),
        evidence=evidence,
        improvement=improvement,
    )


def _score_weight(status: ConceptStatus) -> float:
    if status == "implemented":
        return 1.0
    if status == "partial":
        return 0.5
    return 0.0
