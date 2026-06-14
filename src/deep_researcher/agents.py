"""Specialized agents for multi-hop deep research."""

from __future__ import annotations

import re
from urllib.parse import urlparse

from deep_researcher.config import ResearchConfig
from deep_researcher.llm import ResearchLLM
from deep_researcher.models import ResearchState, SourceAssessment, SourceDocument
from deep_researcher.prompts import get_system_prompt, render_system_prompt
from deep_researcher.retrieval import retrieve_relevant_context
from deep_researcher.search import SOURCE_SEARCH_QUERIES, parallel_tavily_search


class ResearchAgents:
    """Collection of LangGraph-compatible agent node callables."""

    def __init__(self, config: ResearchConfig) -> None:
        self.config = config
        self.llm = ResearchLLM(config)

    def plan_research(self, state: ResearchState) -> ResearchState:
        """Query Planning Agent: decomposes a broad query into sub-questions."""

        query = state["query"]
        logs = _with_logs(state, "Query Planning Agent started: decomposing the research question.")
        prompt = f"""Research question: {query}

Create 4 focused sub-questions that would support a multi-hop investigation.
Cover facts, causes, evidence quality, contradictions, and implications.
Return one sub-question per line."""
        response = self.llm.generate(
            get_system_prompt("query_planner"),
            prompt,
        )
        sub_questions = _parse_lines(response)
        if len(sub_questions) < 3:
            sub_questions = [
                f"What are the most important current facts about {query}?",
                f"What evidence and primary sources best support claims about {query}?",
                f"What contradictions, risks, or unresolved debates exist around {query}?",
                f"What trends, hypotheses, or next-step implications emerge from {query}?",
            ]

        return {
            **state,
            "sub_questions": sub_questions[:5],
            "logs": [
                *logs,
                f"Query Planning Agent completed: generated {min(len(sub_questions), 5)} sub-questions.",
            ],
        }

    def retrieve_context(self, state: ResearchState) -> ResearchState:
        """Contextual Retriever Agent: runs parallel source search and FAISS top-k."""

        query = state["query"]
        logs = _with_logs(
            state,
            "Contextual Retriever Agent started: parallel Tavily lanes are research papers, news, reports, and APIs.",
        )
        source_lanes = ", ".join(SOURCE_SEARCH_QUERIES.keys())
        contextual_prompt = render_system_prompt(
            "contextual_retriever",
            query=query,
            source_lanes=source_lanes,
            top_k=self.config.max_retrieval_docs,
        )
        retrieval_plan = self.llm.generate(
            contextual_prompt,
            "Create a concise retrieval plan for the configured source lanes before search.",
        )
        web_sources = parallel_tavily_search(query, self.config)

        all_sources = _dedupe_sources([*state.get("local_documents", []), *web_sources])
        retrieved_context = retrieve_relevant_context(query, all_sources, self.config)
        if not retrieved_context:
            retrieved_context = all_sources[: self.config.max_retrieval_docs]
        retrieved_context = retrieved_context[: self.config.max_retrieval_docs]

        source_inventory = _format_source_inventory(all_sources)
        tuner_prompt = render_system_prompt(
            "relevant_context_tuner",
            query=query,
            source_inventory=source_inventory,
            faiss_top_context=_format_sources(retrieved_context, max_chars=3_000),
        )
        relevant_context_summary = _limit_words(
            self.llm.generate(
                tuner_prompt,
                "Tune the combined sources into concise relevant context for the Top 3 LLM handoff.",
            ),
            140,
        )

        selector_prompt = render_system_prompt(
            "source_selector",
            query=query,
            top_k=self.config.max_retrieval_docs,
            relevant_context_summary=relevant_context_summary,
            retrieved_context=_format_sources(retrieved_context, max_chars=3_000),
        )
        selector_note = self.llm.generate(
            selector_prompt,
            "Confirm in one short paragraph why these retrieved_context sources should be passed to the LLM agents.",
        )

        return {
            **state,
            "tavily_sources": web_sources,
            "retrieved_context": retrieved_context,
            "relevant_context_summary": relevant_context_summary,
            "logs": [
                *logs,
                (
                    "Contextual Retriever Agent completed: "
                    f"ran {len(SOURCE_SEARCH_QUERIES)} parallel source lanes, "
                    f"collected {len(all_sources)} source documents, and selected top "
                    f"{len(retrieved_context)} FAISS chunks."
                ),
                f"Contextual Retriever prompt plan: {_limit_words(retrieval_plan, 40)}",
                f"Tuning to Relevant Context completed: {_limit_words(relevant_context_summary, 45)}",
                f"FAISS Context Selector note: {_limit_words(selector_note, 40)}",
            ],
        }

    def assess_sources(self, state: ResearchState) -> ResearchState:
        """Source Validator Agent: evaluates the top retrieved results with the LLM."""

        logs = _with_logs(
            state,
            f"Source Validator Agent started: validating top {self.config.validator_top_k} retrieved results.",
        )
        validation_sources = state.get("retrieved_context", [])[: self.config.validator_top_k]
        assessments = [_assess_source(source) for source in validation_sources]
        validation_summary = self.llm.generate(
            get_system_prompt("source_validator"),
            f"""Research question: {state['query']}

Top retrieved results:
{_format_sources(validation_sources, max_chars=4_000)}

Heuristic assessments:
{chr(10).join(f'- {item.source}: {item.credibility}; relevance {item.relevance}; caveat {item.caveats}' for item in assessments)}

Validate these sources in under 120 words. Mention contradictions or provenance risks.""",
        )
        return {
            **state,
            "source_assessments": assessments,
            "source_validation_summary": _limit_words(validation_summary, 120),
            "logs": [
                *logs,
                f"Source Validator Agent completed: validated {len(validation_sources)} top retrieved results.",
            ],
        }

    def analyze_findings(self, state: ResearchState) -> ResearchState:
        """Critical Analysis Agent: synthesizes findings and highlights tensions."""

        logs = _with_logs(
            state,
            f"Critical Analysis Agent started: using top {self.config.max_retrieval_docs} FAISS results.",
        )
        context = _format_sources(state.get("retrieved_context", [])[: self.config.max_retrieval_docs], max_chars=5_000)
        assessments = "\n".join(
            f"- {item.source}: {item.credibility}; {item.caveats}"
            for item in state.get("source_assessments", [])
        )
        synthesis = self.llm.generate(
            get_system_prompt("critical_analysis"),
            f"""Question: {state['query']}

Retrieved context:
{context}

Source assessments:
{assessments}

LLM validation summary:
{state.get('source_validation_summary', '')}

Write within {self.config.critical_analysis_word_limit} words.
Summarize findings, highlight contradictions, and validate source strength.""",
        )
        synthesis = _limit_words(synthesis, self.config.critical_analysis_word_limit)
        return {
            **state,
            "synthesis": synthesis,
            "contradictions": _extract_contradictions(synthesis),
            "logs": [
                *logs,
                f"Critical Analysis Agent completed: produced <= {self.config.critical_analysis_word_limit} words.",
            ],
        }

    def generate_insights(self, state: ResearchState) -> ResearchState:
        """Insight Generation Agent: proposes hypotheses and trends."""

        logs = _with_logs(
            state,
            f"Insight Generation Agent started: capped at {self.config.insight_word_limit} words.",
        )
        context = _format_sources(state.get("retrieved_context", [])[: self.config.max_retrieval_docs], max_chars=4_000)
        response = self.llm.generate(
            get_system_prompt("insight_generation"),
            f"""Question: {state['query']}

Synthesis:
{state.get('synthesis', '')}

Contradictions and caveats:
{chr(10).join(state.get('contradictions', []))}

Evidence context:
{context}

Word limit: {self.config.insight_word_limit}
Return two sections:
Insights:
- ...
Hypotheses:
- ...""",
        )
        response = _limit_words(response, self.config.insight_word_limit)
        insights, hypotheses = _split_insights(response)
        return {
            **state,
            "insights": insights,
            "hypotheses": hypotheses,
            "logs": [
                *logs,
                f"Insight Generation Agent completed: generated {len(insights)} insights and {len(hypotheses)} hypotheses.",
            ],
        }

    def generate_reproducible_snippet(self, state: ResearchState) -> ResearchState:
        """Reproducible Snippet Agent: creates optional notebook-ready code."""

        logs = _with_logs(
            state,
            "Reproducible Snippet Agent started: preparing notebook-ready Python from retrieved sources.",
        )
        if not self.config.generate_code_snippet:
            return {
                **state,
                "reproducible_snippet": "",
                "logs": [*logs, "Reproducible Snippet Agent skipped: disabled in configuration."],
            }

        sources = state.get("retrieved_context", [])[: self.config.max_retrieval_docs]
        source_table = _format_source_records_for_code(sources)
        snippet = self.llm.generate(
            get_system_prompt("reproducible_snippet"),
            f"""Research question: {state['query']}

Critical synthesis:
{state.get('synthesis', '')}

Insights:
{chr(10).join(f'- {item}' for item in state.get('insights', []))}

Source records:
{source_table}

Create a Python snippet that stores these source records, prints source counts by type,
and prints a compact evidence checklist for reproducing the report reasoning.""",
        )
        snippet = _ensure_python_code_block(snippet, sources, state["query"])
        return {
            **state,
            "reproducible_snippet": snippet,
            "logs": [*logs, "Reproducible Snippet Agent completed: generated notebook-ready Python."],
        }

    def build_report(self, state: ResearchState) -> ResearchState:
        """Report Builder Agent: compiles the final structured report."""

        logs = _with_logs(
            state,
            f"Report Builder Agent started: enforcing {self.config.report_min_words}-{self.config.report_max_words} words and source-link rules.",
        )
        sources = state.get("retrieved_context", [])[: self.config.max_retrieval_docs]
        citations = "\n".join(
            _format_source_link(source)
            for source in _ensure_two_sources(sources)
        )
        report = self.llm.generate(
            get_system_prompt("report_builder"),
            f"""Research question: {state['query']}

Sub-questions:
{chr(10).join(f'- {item}' for item in state.get('sub_questions', []))}

Critical synthesis:
{state.get('synthesis', '')}

Contradictions and caveats:
{chr(10).join(f'- {item}' for item in state.get('contradictions', []))}

Insights:
{chr(10).join(f'- {item}' for item in state.get('insights', []))}

Hypotheses:
{chr(10).join(f'- {item}' for item in state.get('hypotheses', []))}

Sources:
{citations}

Build a 200-300 word report. Start with a bold, specific, counterintuitive hook.
End with a ## SOURCES section using exactly this source format:
[Title] - domain.com""",
        )
        report = _enforce_report_rules(report, sources, state["query"], self.config)

        return {
            **state,
            "report": report,
            "logs": [
                *logs,
                "Report Builder Agent completed: compiled the final rules-checked report.",
            ],
        }


def _format_sources(sources: list[SourceDocument], *, max_chars: int) -> str:
    blocks: list[str] = []
    used = 0
    for index, source in enumerate(sources, start=1):
        block = (
            f"[{index}] {source.title}\n"
            f"URL: {source.url or 'local/uploaded'}\n"
            f"Type: {source.source_type}\n"
            f"Content: {source.content.strip()}\n"
        )
        if used + len(block) > max_chars:
            break
        blocks.append(block)
        used += len(block)
    return "\n---\n".join(blocks) if blocks else "No source context available."


def _format_source_inventory(sources: list[SourceDocument], *, max_items: int = 10) -> str:
    if not sources:
        return "No sources collected."

    counts: dict[str, int] = {}
    for source in sources:
        counts[source.source_type] = counts.get(source.source_type, 0) + 1

    count_lines = ", ".join(f"{source_type}: {count}" for source_type, count in sorted(counts.items()))
    source_lines = []
    for index, source in enumerate(sources[:max_items], start=1):
        source_lines.append(
            f"[{index}] {source.title} | type={source.source_type} | url={source.url or 'local/uploaded'}"
        )
    return f"Source counts: {count_lines}\n" + "\n".join(source_lines)


def _format_source_records_for_code(sources: list[SourceDocument]) -> str:
    if not sources:
        return "[]"
    rows = []
    for source in sources:
        rows.append(
            {
                "title": source.title,
                "url": source.url or "",
                "source_type": source.source_type,
                "content_preview": _limit_words(source.content.replace("\n", " "), 40),
            }
        )
    return repr(rows)


def _ensure_python_code_block(
    snippet: str,
    sources: list[SourceDocument],
    query: str,
) -> str:
    if "```python" in snippet and "```" in snippet.replace("```python", "", 1):
        return snippet.strip()

    source_records = _format_source_records_for_code(sources)
    fallback = f'''```python
from collections import Counter

research_question = {query!r}
sources = {source_records}

print("Research question:", research_question)
print("Source count:", len(sources))
print("Sources by type:", dict(Counter(source["source_type"] for source in sources)))

print("\\nEvidence checklist:")
for index, source in enumerate(sources, start=1):
    title = source["title"]
    source_type = source["source_type"]
    preview = source["content_preview"]
    print(f"{{index}}. [{{source_type}}] {{title}}: {{preview}}")
```'''
    return fallback


def _with_logs(state: ResearchState, message: str) -> list[str]:
    return [*state.get("logs", []), message]


def _parse_lines(text: str) -> list[str]:
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip().strip("-* ")
        if not line:
            continue
        if len(line) > 2 and line[0].isdigit() and line[1] in {".", ")"}:
            line = line[2:].strip()
        if line and not line.lower().startswith(("insights:", "hypotheses:")):
            lines.append(line)
    return lines


def _extract_contradictions(text: str) -> list[str]:
    lines = [
        line
        for line in _parse_lines(text)
        if any(cue in line.lower() for cue in ("contradict", "however", "but", "caveat", "risk", "gap"))
    ]
    return lines or ["No direct contradiction was found in the top retrieved context."]


def _split_insights(text: str) -> tuple[list[str], list[str]]:
    insights: list[str] = []
    hypotheses: list[str] = []
    current = insights
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lowered = line.lower().rstrip(":")
        if lowered == "insights":
            current = insights
            continue
        if lowered == "hypotheses":
            current = hypotheses
            continue
        cleaned = line.strip("-* ")
        if cleaned:
            current.append(cleaned)
    if not insights and not hypotheses:
        insights = _parse_lines(text)
    return insights[:6], hypotheses[:6]


def _dedupe_sources(sources: list[SourceDocument]) -> list[SourceDocument]:
    seen: set[str] = set()
    deduped: list[SourceDocument] = []
    for source in sources:
        key = source.url or f"{source.title}:{source.content[:120]}"
        if key in seen or not source.content.strip():
            continue
        seen.add(key)
        deduped.append(source)
    return deduped


def _limit_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]).rstrip(".,;:") + "."


def _word_count(text: str) -> int:
    return len(re.findall(r"\b[\w']+\b", text))


def _sanitize_report_text(text: str) -> str:
    replacements = {
        "—": ",",
        "–": ",",
        " -- ": ", ",
        "--": ",",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    filler_phrases = (
        "game-changer",
        "paradigm shift",
        "in today's landscape",
        "delve into",
        "navigate the complexities",
        "as we move forward",
        "ever-evolving",
    )
    for phrase in filler_phrases:
        text = re.sub(re.escape(phrase), "", text, flags=re.IGNORECASE)

    first_person_claims = (
        "I built",
        "I implemented",
        "we launched",
        "we deployed",
        "we designed",
        "I led",
    )
    for claim in first_person_claims:
        text = re.sub(re.escape(claim), "a team observed", text, flags=re.IGNORECASE)
    return text.strip()


def _enforce_report_rules(
    report: str,
    sources: list[SourceDocument],
    query: str,
    config: ResearchConfig,
) -> str:
    report = _sanitize_report_text(report)
    body = re.split(r"\n##\s+SOURCES\b|\n##\s+Sources\b", report, maxsplit=1)[0].strip()
    body_lines = [line for line in body.splitlines() if line.strip()]
    if not body_lines or not body_lines[0].strip().startswith("**"):
        hook = "**Only three retrieved sources reach the LLM, and that limit is the feature.**"
        body = f"{hook}\n\n{body}" if body else hook

    source_lines = [_format_source_link(source) for source in _ensure_two_sources(sources)]
    sources_section = "## SOURCES\n" + "\n".join(source_lines)
    report = f"{body.strip()}\n\n{sources_section}"
    report = _sanitize_report_text(report)

    if _word_count(report) < config.report_min_words:
        report = _pad_report(report, query, config.report_min_words)
    if _word_count(report) > config.report_max_words:
        report = _trim_report(report, config.report_max_words)
    return _sanitize_report_text(report)


def _pad_report(report: str, query: str, min_words: int) -> str:
    body, sources = _split_sources_section(report)
    additions = [
        f"For {query}, the useful constraint is not more context.",
        "It is stricter evidence routing.",
        "Small source sets make weak provenance visible.",
        "They also make contradictions easier to audit.",
        "That matters when agents summarize fast-moving topics.",
        "A larger crawl can bury the best source.",
        "A smaller ranked set forces review discipline.",
        "The practical risk is confidence without traceability.",
        "The better pattern is narrow retrieval, then explicit validation.",
    ]
    index = 0
    while _word_count(f"{body}\n\n{sources}") < min_words:
        body = f"{body}\n\n{additions[index % len(additions)]}"
        index += 1
    return f"{body.strip()}\n\n{sources.strip()}"


def _trim_report(report: str, max_words: int) -> str:
    body, sources = _split_sources_section(report)
    source_words = _word_count(sources)
    allowed_body_words = max(1, max_words - source_words)
    words = body.split()
    if len(words) > allowed_body_words:
        body = " ".join(words[:allowed_body_words]).rstrip(".,;:") + "."
    return f"{body.strip()}\n\n{sources.strip()}"


def _split_sources_section(report: str) -> tuple[str, str]:
    parts = re.split(r"\n##\s+SOURCES\b", report, maxsplit=1)
    if len(parts) == 2:
        return parts[0].strip(), f"## SOURCES\n{parts[1].strip()}"
    return report.strip(), "## SOURCES"


def _ensure_two_sources(sources: list[SourceDocument]) -> list[SourceDocument]:
    usable = sources[:]
    while len(usable) < 2:
        usable.append(
            SourceDocument(
                title="Workflow configuration",
                content="Configuration source used to document retrieval limits and fallback behavior.",
                source_type="configuration",
                metadata={"domain": "local.config"},
            )
        )
    return usable[: max(2, len(usable))]


def _format_source_link(source: SourceDocument) -> str:
    title = source.title.strip() or "Untitled source"
    return f"[{title}] - {_source_domain(source)}"


def _source_domain(source: SourceDocument) -> str:
    if source.url:
        domain = urlparse(source.url).netloc.lower()
        return domain.replace("www.", "") or "unknown.local"
    if source.metadata.get("domain"):
        return str(source.metadata["domain"])
    if source.source_type == "uploaded_file":
        return "local.upload"
    if source.source_type == "system_notice":
        return "system.notice"
    return "local.source"


def _assess_source(source: SourceDocument) -> SourceAssessment:
    domain = urlparse(source.url).netloc.lower() if source.url else "local document"
    if source.source_type == "system_notice":
        credibility = "configuration notice, not evidence"
        caveats = "Use only to understand missing setup, not to support research claims."
    elif any(domain.endswith(suffix) for suffix in (".gov", ".edu", ".int")):
        credibility = "high baseline credibility"
        caveats = "Still check publication date, methods, and institutional incentives."
    elif any(name in domain for name in ("nature", "science", "arxiv", "pubmed", "who", "worldbank", "oecd")):
        credibility = "potentially strong research or institutional source"
        caveats = "Validate whether the item is peer-reviewed, a preprint, or secondary coverage."
    elif source.url:
        credibility = "open web source"
        caveats = "Corroborate claims with primary or institutional sources."
    else:
        credibility = "user-provided local source"
        caveats = "Credibility depends on the uploaded document provenance."

    relevance = "high" if len(source.content) > 500 else "medium"
    return SourceAssessment(
        source=source.title,
        credibility=credibility,
        relevance=relevance,
        caveats=caveats,
    )
