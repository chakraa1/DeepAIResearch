"""LLM abstraction with an offline fallback."""

from __future__ import annotations

import re
from collections import Counter

from langchain_core.messages import HumanMessage, SystemMessage

from deep_researcher.config import ResearchConfig

SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]+")
STOP_WORDS = {
    "about",
    "after",
    "again",
    "also",
    "because",
    "before",
    "between",
    "could",
    "from",
    "have",
    "into",
    "more",
    "most",
    "over",
    "should",
    "than",
    "that",
    "their",
    "there",
    "these",
    "this",
    "through",
    "under",
    "using",
    "were",
    "when",
    "where",
    "which",
    "while",
    "with",
    "would",
}


class ResearchLLM:
    """Small facade over ChatOpenAI plus an extractive local fallback."""

    def __init__(self, config: ResearchConfig) -> None:
        self.config = config
        self._model = None
        if config.openai_api_key:
            from langchain_openai import ChatOpenAI

            self._model = ChatOpenAI(
                model=config.openai_model,
                api_key=config.openai_api_key,
                base_url=config.llm_base_url,
                temperature=0.2,
            )

    @property
    def enabled(self) -> bool:
        return self._model is not None

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate model text or a deterministic extractive response."""

        if self._model is not None:
            response = self._model.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ]
            )
            return str(response.content)

        return local_generate(system_prompt, user_prompt)


def local_generate(system_prompt: str, user_prompt: str) -> str:
    """Create a useful offline response from salient prompt sentences."""

    topic = _extract_topic(user_prompt)
    sentences = _rank_sentences(user_prompt)
    if "contradiction" in system_prompt.lower():
        return _local_contradictions(sentences)
    if "hypotheses" in system_prompt.lower() or "insight" in system_prompt.lower():
        return _local_insights(topic, sentences)
    if "report" in system_prompt.lower():
        return _local_report(topic, sentences)
    return _local_summary(topic, sentences)


def _extract_topic(text: str) -> str:
    for marker in ("Question:", "Research question:", "Topic:"):
        if marker in text:
            line = text.split(marker, 1)[1].splitlines()[0].strip()
            if line:
                return line
    return "the research question"


def _rank_sentences(text: str, limit: int = 8) -> list[str]:
    sentences = [
        sentence.strip()
        for sentence in SENTENCE_RE.split(text.replace("\n", " "))
        if 40 <= len(sentence.strip()) <= 400
    ]
    words = [
        word.lower()
        for word in WORD_RE.findall(text)
        if len(word) > 3 and word.lower() not in STOP_WORDS
    ]
    counts = Counter(words)

    def score(sentence: str) -> tuple[int, int]:
        tokens = WORD_RE.findall(sentence.lower())
        return (sum(counts[token] for token in tokens), -len(sentence))

    ranked = sorted(sentences, key=score, reverse=True)
    deduped: list[str] = []
    seen: set[str] = set()
    for sentence in ranked:
        normalized = sentence.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(sentence)
        if len(deduped) >= limit:
            break
    return deduped


def _local_summary(topic: str, sentences: list[str]) -> str:
    bullets = "\n".join(f"- {sentence}" for sentence in sentences[:5])
    if not bullets:
        bullets = "- No substantive source text was available for grounded synthesis."
    return f"Offline synthesis for {topic}:\n{bullets}"


def _local_contradictions(sentences: list[str]) -> str:
    cue_words = ("however", "but", "although", "conflict", "contradict", "risk")
    conflicts = [
        sentence
        for sentence in sentences
        if any(cue in sentence.lower() for cue in cue_words)
    ]
    if not conflicts:
        return "- No direct contradictions were detected in the retrieved text."
    return "\n".join(f"- Potential tension: {sentence}" for sentence in conflicts[:4])


def _local_insights(topic: str, sentences: list[str]) -> str:
    if not sentences:
        return "- Add more source material to generate evidence-backed insights."
    return "\n".join(
        [
            f"- Hypothesis: {topic} is shaped by the recurring evidence around '{_keywords(sentences)}'.",
            "- Trend watch: prioritize sources that mention recent adoption, regulation, cost, or performance changes.",
            "- Next step: compare claims across source types before treating any single result as definitive.",
        ]
    )


def _local_report(topic: str, sentences: list[str]) -> str:
    evidence = "\n".join(f"- {sentence}" for sentence in sentences[:6])
    if not evidence:
        evidence = "- No source-backed evidence was available."
    return f"""# Deep Research Report: {topic}

## Executive Summary
This offline report was generated from available retrieved context. Configure OPENAI_API_KEY for richer long-form reasoning.

## Key Evidence
{evidence}

## Preliminary Interpretation
The strongest signals are the claims that recur across the retrieved context. Treat this as a first-pass synthesis and validate with additional high-quality sources.

## Recommended Follow-up
- Add or search for primary sources, official reports, and recent technical publications.
- Compare source dates and incentives before making strategic decisions.
- Rerun with Tavily and OpenAI API keys for deeper multi-hop web investigation.
"""


def _keywords(sentences: list[str]) -> str:
    words = [
        word.lower()
        for sentence in sentences
        for word in WORD_RE.findall(sentence)
        if len(word) > 4 and word.lower() not in STOP_WORDS
    ]
    if not words:
        return "available evidence"
    return ", ".join(word for word, _ in Counter(words).most_common(3))
