"""Specialized agents for multi-hop deep research."""

from __future__ import annotations

from urllib.parse import urlparse

from deep_researcher.config import ResearchConfig
from deep_researcher.llm import ResearchLLM
from deep_researcher.models import ResearchState, SourceAssessment, SourceDocument
from deep_researcher.retrieval import retrieve_relevant_context
from deep_researcher.search import tavily_search


class ResearchAgents:
    """Collection of LangGraph-compatible agent node callables."""

    def __init__(self, config: ResearchConfig) -> None:
        self.config = config
        self.llm = ResearchLLM(config)

    def plan_research(self, state: ResearchState) -> ResearchState:
        """Query Planning Agent: decomposes a broad query into sub-questions."""

        query = state["query"]
        prompt = f"""Research question: {query}

Create 4 focused sub-questions that would support a multi-hop investigation.
Cover facts, causes, evidence quality, contradictions, and implications.
Return one sub-question per line."""
        response = self.llm.generate(
            "You are a senior research strategist who plans multi-hop investigations.",
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
            "logs": [*state.get("logs", []), "Query Planning Agent generated research sub-questions."],
        }

    def retrieve_context(self, state: ResearchState) -> ResearchState:
        """Contextual Retriever Agent: searches web and FAISS-indexes all sources."""

        query = state["query"]
        sub_questions = state.get("sub_questions") or [query]
        web_sources: list[SourceDocument] = []
        for sub_question in sub_questions[:4]:
            web_sources.extend(
                tavily_search(
                    sub_question,
                    self.config,
                    max_results=max(1, self.config.max_web_results // 2),
                )
            )

        all_sources = _dedupe_sources([*state.get("local_documents", []), *web_sources])
        retrieved_context = retrieve_relevant_context(query, all_sources, self.config)
        if not retrieved_context:
            retrieved_context = all_sources[: self.config.max_retrieval_docs]

        return {
            **state,
            "retrieved_context": retrieved_context,
            "logs": [
                *state.get("logs", []),
                f"Contextual Retriever Agent collected {len(all_sources)} source documents and selected {len(retrieved_context)} chunks.",
            ],
        }

    def assess_sources(self, state: ResearchState) -> ResearchState:
        """Source Validator Agent: evaluates relevance, credibility, and caveats."""

        assessments = [_assess_source(source) for source in state.get("retrieved_context", [])]
        return {
            **state,
            "source_assessments": assessments,
            "logs": [*state.get("logs", []), "Source Validator Agent assessed credibility and caveats."],
        }

    def analyze_findings(self, state: ResearchState) -> ResearchState:
        """Critical Analysis Agent: synthesizes findings and highlights tensions."""

        context = _format_sources(state.get("retrieved_context", []), max_chars=8_000)
        assessments = "\n".join(
            f"- {item.source}: {item.credibility}; {item.caveats}"
            for item in state.get("source_assessments", [])
        )
        synthesis = self.llm.generate(
            "You are a critical analysis agent. Summarize findings, validate source strength, and avoid unsupported claims.",
            f"""Question: {state['query']}

Retrieved context:
{context}

Source assessments:
{assessments}

Write a concise synthesis with evidence-backed claims and cite source titles inline.""",
        )
        contradictions_text = self.llm.generate(
            "You are a contradiction detection agent. Identify contradictions, uncertainty, and missing evidence.",
            f"""Question: {state['query']}

Retrieved context:
{context}

List contradictions, caveats, or evidence gaps as bullets.""",
        )
        return {
            **state,
            "synthesis": synthesis,
            "contradictions": _parse_lines(contradictions_text) or [contradictions_text],
            "logs": [*state.get("logs", []), "Critical Analysis Agent synthesized findings and evidence gaps."],
        }

    def generate_insights(self, state: ResearchState) -> ResearchState:
        """Insight Generation Agent: proposes hypotheses and trends."""

        context = _format_sources(state.get("retrieved_context", []), max_chars=6_000)
        response = self.llm.generate(
            "You are an insight generation agent. Produce hypotheses, trends, and reasoning chains grounded in evidence.",
            f"""Question: {state['query']}

Synthesis:
{state.get('synthesis', '')}

Contradictions and caveats:
{chr(10).join(state.get('contradictions', []))}

Evidence context:
{context}

Return two sections:
Insights:
- ...
Hypotheses:
- ...""",
        )
        insights, hypotheses = _split_insights(response)
        return {
            **state,
            "insights": insights,
            "hypotheses": hypotheses,
            "logs": [*state.get("logs", []), "Insight Generation Agent proposed trends and hypotheses."],
        }

    def build_report(self, state: ResearchState) -> ResearchState:
        """Report Builder Agent: compiles the final structured report."""

        sources = state.get("retrieved_context", [])
        citations = "\n".join(
            f"- [{index}] {source.citation_label} ({source.source_type})"
            for index, source in enumerate(sources, start=1)
        )
        report = self.llm.generate(
            "You are a report builder agent. Create a polished Markdown deep-research report with citations.",
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

Build a report with: Executive Summary, Research Path, Evidence Review, Contradictions & Caveats, Insights & Hypotheses, Recommended Next Questions, and Sources.""",
        )
        if "## Sources" not in report and "# Sources" not in report:
            report = f"{report.rstrip()}\n\n## Sources\n{citations}"

        return {
            **state,
            "report": report,
            "logs": [*state.get("logs", []), "Report Builder Agent compiled the final report."],
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
