"""LangGraph orchestration for the multi-agent research assistant."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from deep_researcher.agent.agents import ResearchAgents
from deep_researcher.config import ResearchConfig
from deep_researcher.main.models import ResearchState, SourceDocument


class DeepResearchWorkflow:
    """Coordinates the specialized research agents."""

    def __init__(self, config: ResearchConfig | None = None) -> None:
        self.config = config or ResearchConfig.from_env()
        self.agents = ResearchAgents(self.config)
        self.checkpointer = MemorySaver()
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(ResearchState)
        graph.add_node("query_planner", self.agents.plan_research)
        graph.add_node("contextual_retriever", self.agents.retrieve_context)
        graph.add_node("source_validator", self.agents.assess_sources)
        graph.add_node("critical_analysis", self.agents.analyze_findings)
        graph.add_node("insight_generation", self.agents.generate_insights)
        graph.add_node("reproducible_snippet", self.agents.generate_reproducible_snippet)
        graph.add_node("human_review_gate", self.agents.review_before_report)
        graph.add_node("report_builder", self.agents.build_report)
        graph.add_node("report_reflection", self.agents.reflect_on_report)
        graph.add_node("report_revision", self.agents.revise_report_inline)

        graph.set_entry_point("query_planner")
        graph.add_edge("query_planner", "contextual_retriever")
        graph.add_edge("contextual_retriever", "source_validator")
        graph.add_edge("source_validator", "critical_analysis")
        graph.add_edge("critical_analysis", "insight_generation")
        graph.add_edge("insight_generation", "reproducible_snippet")
        graph.add_edge("reproducible_snippet", "human_review_gate")
        graph.add_edge("human_review_gate", "report_builder")
        graph.add_edge("report_builder", "report_reflection")
        graph.add_edge("report_reflection", "report_revision")
        graph.add_edge("report_revision", END)
        return graph.compile(checkpointer=self.checkpointer)

    def run(
        self,
        query: str,
        *,
        local_documents: list[SourceDocument] | None = None,
        thread_id: str | None = None,
    ) -> ResearchState:
        """Run the full workflow and return the final state."""

        thread_id = thread_id or self.config.checkpoint_thread_id
        initial_state: ResearchState = {
            "query": query,
            "thread_id": thread_id,
            "local_documents": local_documents or [],
            "logs": [],
            "errors": [],
        }
        return self.graph.invoke(initial_state, config=_graph_config(thread_id))

    def stream(
        self,
        query: str,
        *,
        local_documents: list[SourceDocument] | None = None,
        thread_id: str | None = None,
    ) -> Iterator[tuple[str, ResearchState]]:
        """Yield each agent state update as the graph progresses."""

        thread_id = thread_id or self.config.checkpoint_thread_id
        initial_state: ResearchState = {
            "query": query,
            "thread_id": thread_id,
            "local_documents": local_documents or [],
            "logs": [],
            "errors": [],
        }
        for event in self.graph.stream(initial_state, config=_graph_config(thread_id)):
            for node_name, state_update in event.items():
                yield node_name, _coerce_state(state_update)


def _coerce_state(value: Any) -> ResearchState:
    return value if isinstance(value, dict) else {}


def _graph_config(thread_id: str) -> dict[str, dict[str, str]]:
    return {"configurable": {"thread_id": thread_id}}
