"""Streamlit UI for CyberSecurityAIAgent."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from deep_researcher import DeepResearchWorkflow, ResearchConfig
from deep_researcher.config import OPENROUTER_BASE_URL
from deep_researcher.config.concepts import coverage_table_rows, score_concept_coverage
from deep_researcher.main.evaluation import run_offline_evaluation
from deep_researcher.main.models import SourceDocument


AGENT_LABELS = {
    "security_planner": "Security Planning Agent",
    "log_monitor": "Log Monitor Agent",
    "threat_intelligence": "Threat Intelligence Agent",
    "vulnerability_scanner": "Vulnerability Scanner Agent",
    "incident_response": "Incident Response Agent",
    "policy_checker": "Policy Checker Agent",
    "evaluation_loop": "Evaluation Loop Agent",
    "human_review_gate": "Human Review Gate",
    "security_report_builder": "Security Report Builder Agent",
    "report_reflection": "Report Reflection Agent",
    "report_revision": "Report Revision Agent",
}


def main() -> None:
    st.set_page_config(
        page_title="CyberSecurityAIAgent",
        page_icon="🛡️",
        layout="wide",
    )

    st.title("🛡️ CyberSecurityAIAgent")
    st.caption(
        "A LangGraph multi-agent cybersecurity system for logs, CVEs, vulnerability scanning, incident response, policy checks, and evaluation."
    )

    try:
        config = load_config_from_ui()
    except Exception:
        render_settings_error()
        return
    render_sidebar(config)

    query = st.text_area(
        "Security assessment request",
        placeholder="Example: Assess suspicious login failures, Docker hardening, and Log4j exposure for an internet banking platform.",
        height=110,
    )
    uploaded_files = st.file_uploader(
        "Optional evidence files",
        type=["txt", "md", "pdf", "log", "py", "js", "java", "json", "yaml", "yml", "conf", "ini", "sql", "toml"],
        accept_multiple_files=True,
        help="Upload logs, source snippets, Docker/database/API configs, SBOMs, reports, or policy documents.",
    )

    col_a, col_b, col_c = st.columns([1, 1, 3])
    with col_a:
        run_clicked = st.button("Run security assessment", type="primary", use_container_width=True)
    with col_b:
        eval_clicked = st.button("Run offline eval suite", use_container_width=True)
    with col_c:
        st.info(
            "For best CVE intelligence set `TAVILY_API_KEY` and an LLM key. Without keys, the app runs with local heuristics, FAISS, and authorized-source guidance.",
            icon="ℹ️",
        )
    if eval_clicked:
        run_evaluation_suite(config)
        return

    if run_clicked:
        if not query.strip():
            st.warning("Enter a security assessment request first.")
            return
        run_security_assessment(query.strip(), uploaded_files, config)
        return

    if "last_report" in st.session_state:
        st.divider()
        render_results(st.session_state["last_state"])


def load_config_from_ui() -> ResearchConfig:
    config = ResearchConfig.from_env()
    secrets = _streamlit_secrets()

    openai_key = secrets.get("OPENAI_API_KEY") or _config_value(config, "openai_api_key", None)
    tavily_key = secrets.get("TAVILY_API_KEY") or _config_value(config, "tavily_api_key", None)
    provider = secrets.get("LLM_PROVIDER") or _config_value(config, "llm_provider", "openai")
    base_url = secrets.get("OPENAI_BASE_URL") or _config_value(config, "openai_base_url", "") or ""
    model = secrets.get("OPENAI_MODEL") or _config_value(config, "openai_model", "gpt-4o-mini")

    with st.sidebar:
        st.header("Configuration")
        provider_label = _provider_label(provider)
        provider_label = st.selectbox(
            "API provider",
            options=["OpenAI", "OpenRouter", "Custom OpenAI-compatible"],
            index=["OpenAI", "OpenRouter", "Custom OpenAI-compatible"].index(provider_label),
            help="Switches the OpenAI-compatible base URL used by all LLM agent calls.",
        )
        provider = _provider_value(provider_label)
        if provider == "openrouter" and not base_url:
            base_url = OPENROUTER_BASE_URL
        if provider == "openai" and base_url == OPENROUTER_BASE_URL:
            base_url = ""
        if provider == "openrouter" and model == "gpt-4o-mini":
            model = "openai/gpt-4o-mini"
        base_url = st.text_input(
            "OpenAI-compatible base URL",
            value=base_url or "",
            help="OpenRouter uses https://openrouter.ai/api/v1. Leave blank for OpenAI.",
        )
        openai_key = st.text_input(
            "LLM API key",
            value=openai_key or "",
            type="password",
            help="Use an OpenAI key for OpenAI, or an OpenRouter key for OpenRouter.",
        )
        tavily_key = st.text_input(
            "Tavily API key",
            value=tavily_key or "",
            type="password",
            help="Optional. Used by the Threat Intelligence Agent for authorized web search lanes.",
        )
        model = st.text_input("LLM model", value=model)
        max_web_results = st.slider("Max web results per assessment", 4, 18, _clamp(_config_value(config, "max_web_results", 8), 4, 18))
        max_retrieval_docs = st.slider(
            "FAISS top-k chunks sent to security agents",
            1,
            10,
            _clamp(_config_value(config, "max_retrieval_docs", 3), 1, 10),
            help="Defaults to 3 to control token usage between retrieval and LLM agents.",
        )
        validator_top_k = st.slider("Source validator top-k", 1, 10, _clamp(_config_value(config, "validator_top_k", 3), 1, 10))
        critical_limit = st.slider("Threat summary word limit", 80, 300, _clamp(_config_value(config, "critical_analysis_word_limit", 200), 80, 300))
        insight_limit = st.slider("Evaluation note word limit", 80, 300, _clamp(_config_value(config, "insight_word_limit", 200), 80, 300))
        report_min_words = st.slider("Report minimum words", 100, 450, _clamp(_config_value(config, "report_min_words", 200), 100, 450))
        report_max_words = st.slider("Report maximum words", 260, 900, _clamp(_config_value(config, "report_max_words", 650), 260, 900))
        report_max_words = max(report_max_words, report_min_words)
        reflection_retry_limit = st.slider(
            "Report reflection retry limit",
            0,
            5,
            _clamp(_config_value(config, "report_reflection_retry_limit", 2), 0, 5),
        )
        generate_code_snippet = st.checkbox(
            "Generate reproducible evaluation snippet",
            value=_config_value(config, "generate_code_snippet", True),
            help="Adds a notebook-ready snippet that reproduces finding counts and evaluation metrics.",
        )
        require_human_review = st.checkbox(
            "Require human review gate",
            value=_config_value(config, "require_human_review", False),
            help="Advanced: uses LangGraph interrupt before Report Builder. Leave off for unattended runs.",
        )
        checkpoint_thread_id = st.text_input(
            "Checkpoint thread ID",
            value=_config_value(config, "checkpoint_thread_id", "cybersecurity-agent-default"),
            help="MemorySaver thread ID for replayable workflow checkpoints.",
        )

    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
    if tavily_key:
        os.environ["TAVILY_API_KEY"] = tavily_key
    os.environ["LLM_PROVIDER"] = provider
    if base_url:
        os.environ["OPENAI_BASE_URL"] = base_url
    else:
        os.environ.pop("OPENAI_BASE_URL", None)

    return _make_research_config(
        {
            "openai_api_key": openai_key or None,
            "tavily_api_key": tavily_key or None,
            "llm_provider": provider,
            "openai_base_url": base_url or None,
            "openai_model": model,
            "max_web_results": max_web_results,
            "max_retrieval_docs": max_retrieval_docs,
            "validator_top_k": validator_top_k,
            "critical_analysis_word_limit": critical_limit,
            "insight_word_limit": insight_limit,
            "report_min_words": report_min_words,
            "report_max_words": report_max_words,
            "generate_code_snippet": generate_code_snippet,
            "report_reflection_retry_limit": reflection_retry_limit,
            "require_human_review": require_human_review,
            "checkpoint_thread_id": checkpoint_thread_id or "cybersecurity-agent-default",
        }
    )


def render_sidebar(config: ResearchConfig) -> None:
    with st.sidebar:
        st.divider()
        st.subheader("Agent System")
        st.markdown(
            """
            1. Security Planning Agent
            2. Log Monitor Agent
            3. Threat Intelligence Agent
            4. Vulnerability Scanner Agent
            5. Incident Response Agent
            6. Policy Checker Agent
            7. Evaluation Loop Agent
            8. Human Review Gate
            9. Security Report Builder Agent
            10. Report Reflection Agent
            11. Report Revision Agent
            """
        )
        st.subheader("Runtime Status")
        st.write("LLM:", "enabled" if _config_value(config, "llm_enabled", False) else "local fallback")
        st.write("Provider:", _provider_label(_config_value(config, "llm_provider", "openai")))
        st.write("Base URL:", _config_value(config, "llm_base_url", None) or "OpenAI default")
        st.write("Threat web search:", "enabled" if _config_value(config, "web_search_enabled", False) else "local/upload-only")
        st.write("FAISS top-k:", _config_value(config, "max_retrieval_docs", 3))
        st.write("Validator top-k:", _config_value(config, "validator_top_k", 3))
        st.write("Report words:", f"{_config_value(config, 'report_min_words', 200)}-{_config_value(config, 'report_max_words', 650)}")
        st.write("Reflection retries:", _config_value(config, "report_reflection_retry_limit", 2))
        st.write("Checkpoint thread:", _config_value(config, "checkpoint_thread_id", "deep-research-default"))
        render_concept_alignment()


def render_concept_alignment() -> None:
    score = score_concept_coverage()
    with st.expander("Hackathon concept alignment", expanded=False):
        st.metric("Concept Score", f"{score.score}/{score.max_score}")
        st.caption(
            f"{score.implemented_count} implemented, {score.partial_count} partial, "
            f"{score.total_count} total concepts."
        )
        st.dataframe(
            coverage_table_rows(),
            hide_index=True,
            use_container_width=True,
        )


def run_security_assessment(query: str, uploaded_files, config: ResearchConfig) -> None:
    local_documents = load_uploaded_documents(uploaded_files or [])
    try:
        workflow = DeepResearchWorkflow(config)
    except Exception:
        render_settings_error()
        return

    progress = st.progress(0)
    status = st.empty()
    log_container = st.container(border=True)
    final_state = None
    node_count = len(AGENT_LABELS)
    shown_logs = 0

    try:
        with st.spinner("Cybersecurity agents are collaborating on the assessment..."):
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
    except AttributeError:
        render_settings_error()
        return
    except Exception:
        st.error(
            "The security assessment could not finish. Please check your API keys, refresh the app, and try again."
        )
        return

    if final_state is None:
        st.error("The workflow did not produce a result.")
        return

    st.session_state["last_state"] = final_state
    st.session_state["last_report"] = final_state.get("report", "")
    render_results(final_state)


def run_evaluation_suite(config: ResearchConfig) -> None:
    st.header("Offline Evaluation Suite")
    with st.spinner("Running deterministic security eval cases through the full workflow..."):
        try:
            rows = run_offline_evaluation(
                ResearchConfig(
                    openai_api_key=None,
                    tavily_api_key=None,
                    max_web_results=_config_value(config, "max_web_results", 8),
                    max_retrieval_docs=_config_value(config, "max_retrieval_docs", 3),
                    validator_top_k=_config_value(config, "validator_top_k", 3),
                    report_reflection_retry_limit=_config_value(config, "report_reflection_retry_limit", 2),
                    checkpoint_thread_id="cybersecurity-agent-eval-ui",
                )
            )
        except Exception as exc:
            st.error(f"Evaluation suite failed: {exc}")
            return
    st.dataframe(rows, hide_index=True, use_container_width=True)
    passed = sum(1 for row in rows if row["passed"])
    st.metric("Eval pass rate", f"{passed}/{len(rows)}")


def render_results(state: dict) -> None:
    st.header("Security Assessment Report")
    report = state.get("report") or "No report generated."
    st.markdown(report)
    st.download_button(
        "Download security report",
        data=report,
        file_name="cybersecurity_ai_agent_report.md",
        mime="text/markdown",
    )

    render_finding_dashboard(state)

    with st.expander("Agent trace", expanded=False):
        for log in state.get("logs", []):
            st.write(log)

    revision_edits = state.get("report_revision_edits", [])
    if revision_edits:
        with st.expander("Report inline edits", expanded=False):
            for edit in revision_edits:
                st.write(f"- {edit}")

    reflection_notes = state.get("report_reflection_notes", [])
    if reflection_notes:
        with st.expander("Report reflection notes", expanded=False):
            st.write(f"Attempts: {state.get('report_reflection_attempts', 0)}")
            for note in reflection_notes:
                st.write(f"- {note}")

    snippet = state.get("reproducible_snippet", "")
    if snippet:
        with st.expander("Reproducible evaluation snippet", expanded=False):
            st.markdown(snippet)
            st.download_button(
                "Download evaluation snippet",
                data=_strip_python_fence(snippet),
                file_name="cybersecurity_eval_snippet.py",
                mime="text/x-python",
            )

    with st.expander("Retrieved threat and policy sources", expanded=False):
        sources = state.get("retrieved_context", [])
        if not sources:
            st.write("No sources were retrieved.")
        for index, source in enumerate(sources, start=1):
            st.markdown(f"**[{index}] {source.title}**")
            if source.url:
                st.caption(source.url)
            st.write(source.content[:900] + ("..." if len(source.content) > 900 else ""))


def render_finding_dashboard(state: dict) -> None:
    findings = state.get("findings", [])
    evaluation = state.get("evaluation_summary")
    risk_score = state.get("risk_score", 0)

    metric_cols = st.columns(4)
    metric_cols[0].metric("Risk score", f"{risk_score:.0f}/100")
    metric_cols[1].metric("Findings", len(findings))
    metric_cols[2].metric("Policy checks", len(state.get("policy_gaps", [])))
    metric_cols[3].metric("Eval score", f"{evaluation.overall_score:.2f}" if evaluation else "n/a")

    if findings:
        rows = [
            {
                "Severity": finding.severity.upper(),
                "Agent": finding.agent,
                "Category": finding.category,
                "Title": finding.title,
                "Assets": ", ".join(finding.affected_assets),
                "Fix": finding.recommended_fix,
            }
            for finding in findings
        ]
        with st.expander("Structured findings", expanded=True):
            st.dataframe(rows, hide_index=True, use_container_width=True)

    policy_gaps = state.get("policy_gaps", [])
    if policy_gaps:
        with st.expander("Policy checker results", expanded=False):
            st.dataframe(
                [
                    {
                        "Framework": gap.framework,
                        "Control": gap.control,
                        "Status": gap.status,
                        "Finding": gap.finding,
                        "Remediation": gap.remediation,
                    }
                    for gap in policy_gaps
                ],
                hide_index=True,
                use_container_width=True,
            )

    incident_steps = state.get("incident_steps", [])
    if incident_steps:
        with st.expander("Incident response steps", expanded=False):
            for step in incident_steps:
                st.write(f"**{step.phase.replace('_', ' ').title()}** ({step.owner}, {step.priority}): {step.action}")

    if evaluation:
        with st.expander("Evaluation loop scorecard", expanded=False):
            st.dataframe(
                [
                    {
                        "Metric": metric.name,
                        "Score": metric.score,
                        "Passed": metric.passed,
                        "Notes": metric.notes,
                    }
                    for metric in evaluation.metrics
                ],
                hide_index=True,
                use_container_width=True,
            )
            st.write("Improvement actions:")
            for action in evaluation.improvement_actions:
                st.write(f"- {action}")


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
                    source_type=_infer_source_type(uploaded_file.name, text),
                    metadata={"filename": uploaded_file.name},
                )
            )
    return documents


def _read_pdf(uploaded_file) -> str:
    from pypdf import PdfReader

    reader = PdfReader(uploaded_file)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def _infer_source_type(filename: str, text: str) -> str:
    name = filename.lower()
    lowered = text[:2_000].lower()
    if name.endswith(".log") or any(marker in lowered for marker in ("failed password", "status=401", "union select")):
        return "log"
    if name == "dockerfile" or name.endswith("dockerfile") or "from " in lowered:
        return "dockerfile"
    if name.endswith((".sql", ".conf", ".ini")) or any(marker in lowered for marker in ("listen_addresses", "mysql_allow_empty_password", "postgres_host_auth_method", "sa_password")):
        return "database_config"
    if name.endswith((".py", ".js", ".java", ".go", ".ts")):
        return "code"
    if name.endswith((".yaml", ".yml", ".json", ".toml")):
        return "configuration"
    return "uploaded_file"


def _streamlit_secrets() -> dict[str, str]:
    try:
        return dict(st.secrets)
    except Exception:
        return {}


def render_settings_error() -> None:
    st.error(
        "The app settings could not be loaded. Please refresh the page, then try again."
    )
    with st.expander("What can I try?"):
        st.write(
            "- Restart Streamlit so it picks up the latest project files.\n"
            "- If you recently pulled changes, reinstall dependencies with `pip install -r requirements.txt`.\n"
            "- Check that your `.env` file does not contain old or misspelled setting names."
        )


def _config_value(config: object, name: str, default):
    try:
        return getattr(config, name)
    except AttributeError:
        return default


def _make_research_config(values: dict) -> ResearchConfig:
    """Build config while tolerating older ResearchConfig shapes."""

    try:
        return ResearchConfig(**values)
    except TypeError:
        if is_dataclass(ResearchConfig):
            allowed = {field.name for field in fields(ResearchConfig)}
            filtered = {key: value for key, value in values.items() if key in allowed}
            return ResearchConfig(**filtered)
        raise


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(value, maximum))


def _provider_label(provider: str | None) -> str:
    provider = (provider or "openai").strip().lower()
    if provider == "openrouter":
        return "OpenRouter"
    if provider == "custom":
        return "Custom OpenAI-compatible"
    return "OpenAI"


def _provider_value(label: str) -> str:
    if label == "OpenRouter":
        return "openrouter"
    if label == "Custom OpenAI-compatible":
        return "custom"
    return "openai"


def _strip_python_fence(snippet: str) -> str:
    stripped = snippet.strip()
    if stripped.startswith("```python"):
        stripped = stripped.removeprefix("```python").strip()
    if stripped.endswith("```"):
        stripped = stripped[: -3].strip()
    return stripped + "\n"


if __name__ == "__main__":
    main()
