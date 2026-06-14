"""Streamlit UI for the Multi-Agent AI Deep Researcher."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from deep_researcher import DeepResearchWorkflow, ResearchConfig
from deep_researcher.models import SourceDocument


AGENT_LABELS = {
    "query_planner": "Query Planning Agent",
    "contextual_retriever": "Contextual Retriever Agent",
    "source_validator": "Source Validator Agent",
    "critical_analysis": "Critical Analysis Agent",
    "insight_generation": "Insight Generation Agent",
    "report_builder": "Report Builder Agent",
}


def main() -> None:
    st.set_page_config(
        page_title="Multi-Agent AI Deep Researcher",
        page_icon="🔎",
        layout="wide",
    )

    st.title("🔎 Multi-Agent AI Deep Researcher")
    st.caption(
        "LangGraph-orchestrated agents for multi-hop, multi-source investigations with Tavily search and FAISS retrieval."
    )

    config = load_config_from_ui()
    render_sidebar(config)

    query = st.text_area(
        "Research question",
        placeholder="Example: What are the latest safety and adoption trends for AI agents in healthcare?",
        height=110,
    )
    uploaded_files = st.file_uploader(
        "Optional local sources",
        type=["txt", "md", "pdf"],
        accept_multiple_files=True,
        help="Upload reports, papers, notes, or PDFs to include in the FAISS retrieval corpus.",
    )

    col_a, col_b = st.columns([1, 3])
    with col_a:
        run_clicked = st.button("Run deep research", type="primary", use_container_width=True)
    with col_b:
        st.info(
            "For best results set `OPENAI_API_KEY` and `TAVILY_API_KEY`. Without keys, the app runs in local demo mode using uploaded files and extractive synthesis.",
            icon="ℹ️",
        )

    if run_clicked:
        if not query.strip():
            st.warning("Enter a research question first.")
            return
        run_research(query.strip(), uploaded_files, config)
        return

    if "last_report" in st.session_state:
        st.divider()
        render_results(st.session_state["last_state"])


def load_config_from_ui() -> ResearchConfig:
    config = ResearchConfig.from_env()
    secrets = _streamlit_secrets()

    openai_key = secrets.get("OPENAI_API_KEY") or config.openai_api_key
    tavily_key = secrets.get("TAVILY_API_KEY") or config.tavily_api_key
    model = secrets.get("OPENAI_MODEL") or config.openai_model

    with st.sidebar:
        st.header("Configuration")
        openai_key = st.text_input(
            "OpenAI API key",
            value=openai_key or "",
            type="password",
            help="Optional. Used for long-context synthesis and report writing.",
        )
        tavily_key = st.text_input(
            "Tavily API key",
            value=tavily_key or "",
            type="password",
            help="Optional. Used by the Contextual Retriever Agent for web search.",
        )
        model = st.text_input("OpenAI model", value=model)
        max_web_results = st.slider("Max web results per investigation", 4, 16, _clamp(config.max_web_results, 4, 16))
        max_retrieval_docs = st.slider(
            "FAISS top-k chunks sent to LLM agents",
            1,
            10,
            _clamp(config.max_retrieval_docs, 1, 10),
            help="Defaults to 3 to control token usage between retrieval and LLM agents.",
        )
        validator_top_k = st.slider("Source validator top-k", 1, 10, _clamp(config.validator_top_k, 1, 10))
        critical_limit = st.slider("Critical analysis word limit", 80, 300, _clamp(config.critical_analysis_word_limit, 80, 300))
        insight_limit = st.slider("Insight generation word limit", 80, 300, _clamp(config.insight_word_limit, 80, 300))
        report_min_words = st.slider("Report minimum words", 100, 300, _clamp(config.report_min_words, 100, 300))
        report_max_words = st.slider("Report maximum words", 200, 400, _clamp(config.report_max_words, 200, 400))
        report_max_words = max(report_max_words, report_min_words)

    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
    if tavily_key:
        os.environ["TAVILY_API_KEY"] = tavily_key

    return ResearchConfig(
        openai_api_key=openai_key or None,
        tavily_api_key=tavily_key or None,
        openai_model=model,
        max_web_results=max_web_results,
        max_retrieval_docs=max_retrieval_docs,
        validator_top_k=validator_top_k,
        critical_analysis_word_limit=critical_limit,
        insight_word_limit=insight_limit,
        report_min_words=report_min_words,
        report_max_words=report_max_words,
    )


def render_sidebar(config: ResearchConfig) -> None:
    with st.sidebar:
        st.divider()
        st.subheader("Agent System")
        st.markdown(
            """
            1. Query Planning Agent
            2. Contextual Retriever Agent
            3. Source Validator Agent
            4. Critical Analysis Agent
            5. Insight Generation Agent
            6. Report Builder Agent
            """
        )
        st.subheader("Runtime Status")
        st.write("LLM:", "enabled" if config.llm_enabled else "local fallback")
        st.write("Web search:", "enabled" if config.web_search_enabled else "local/upload-only")
        st.write("FAISS top-k:", config.max_retrieval_docs)
        st.write("Validator top-k:", config.validator_top_k)
        st.write("Report words:", f"{config.report_min_words}-{config.report_max_words}")


def run_research(query: str, uploaded_files, config: ResearchConfig) -> None:
    local_documents = load_uploaded_documents(uploaded_files or [])
    workflow = DeepResearchWorkflow(config)

    progress = st.progress(0)
    status = st.empty()
    log_container = st.container(border=True)
    final_state = None
    node_count = len(AGENT_LABELS)
    shown_logs = 0

    with st.spinner("Agents are collaborating on the investigation..."):
        for index, (node_name, state) in enumerate(
            workflow.stream(query, local_documents=local_documents),
            start=1,
        ):
            final_state = state
            label = AGENT_LABELS.get(node_name, node_name)
            progress.progress(min(index / node_count, 1.0))
            status.success(f"{label} completed")
            with log_container:
                st.markdown(f"**{label}**")
                logs = state.get("logs", [])
                for log in logs[shown_logs:]:
                    st.write(log)
                shown_logs = len(logs)

    if final_state is None:
        st.error("The workflow did not produce a result.")
        return

    st.session_state["last_state"] = final_state
    st.session_state["last_report"] = final_state.get("report", "")
    render_results(final_state)


def render_results(state: dict) -> None:
    st.header("Research Report")
    report = state.get("report") or "No report generated."
    st.markdown(report)
    st.download_button(
        "Download Markdown report",
        data=report,
        file_name="deep_research_report.md",
        mime="text/markdown",
    )

    with st.expander("Agent trace", expanded=False):
        for log in state.get("logs", []):
            st.write(log)

    with st.expander("Retrieved sources", expanded=False):
        sources = state.get("retrieved_context", [])
        if not sources:
            st.write("No sources were retrieved.")
        for index, source in enumerate(sources, start=1):
            st.markdown(f"**[{index}] {source.title}**")
            if source.url:
                st.caption(source.url)
            st.write(source.content[:900] + ("..." if len(source.content) > 900 else ""))


def load_uploaded_documents(uploaded_files) -> list[SourceDocument]:
    documents: list[SourceDocument] = []
    for uploaded_file in uploaded_files:
        suffix = Path(uploaded_file.name).suffix.lower()
        if suffix == ".pdf":
            text = _read_pdf(uploaded_file)
        else:
            text = uploaded_file.getvalue().decode("utf-8", errors="ignore")

        if text.strip():
            documents.append(
                SourceDocument(
                    title=uploaded_file.name,
                    content=text,
                    source_type="uploaded_file",
                    metadata={"filename": uploaded_file.name},
                )
            )
    return documents


def _read_pdf(uploaded_file) -> str:
    from pypdf import PdfReader

    reader = PdfReader(uploaded_file)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def _streamlit_secrets() -> dict[str, str]:
    try:
        return dict(st.secrets)
    except Exception:
        return {}


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(value, maximum))


if __name__ == "__main__":
    main()
