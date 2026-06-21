# CyberSecurityAIAgent

CyberSecurityAIAgent is a working multi-agent cybersecurity assistant built with **Python**, **LangGraph/LangChain**, **FAISS**, **Tavily**, and **Streamlit**. It demonstrates how RAG and specialized AI agents can monitor evidence, detect security issues, ground CVE analysis in authorized sources, suggest incident-response actions, and map gaps to ISO 27001, NIST CSF, and SOC 2 controls.

The app is designed as a defensive decision-support demo for security engineers, DevOps teams, and IT teams in regulated financial-services environments.

## What the agent demonstrates

- **Log Monitor Agent** parses uploaded system, application, and network logs for brute force, SQL injection, ransomware, exfiltration, and privilege-escalation signals.
- **Threat Intelligence Agent** searches authorized CVE/security lanes through Tavily and FAISS: NVD, CVE.org, CISA KEV, MITRE ATT&CK, vendor advisories, and compliance guidance.
- **Vulnerability Scanner Agent** scans uploaded code, API/config files, Dockerfiles, and database configs for hard-coded secrets, debug mode, dynamic SQL, broad database listeners, disabled TLS, weak database auth, and privileged containers.
- **Incident Response Agent** creates phase-based response actions for triage, containment, eradication, recovery, communications, and lessons learned.
- **Policy Checker Agent** maps findings to ISO 27001:2022, NIST CSF 2.0, and SOC 2 controls.
- **Evaluation Loop Agent** scores each run for specialist coverage, actionability, severity triage, authorized CVE source use, incident-response completeness, and policy mapping.
- **Report Reflection and Revision Agents** enforce required security report sections and source-authority guardrails.

## Tech stack

| Layer | Technology | Purpose |
|---|---|---|
| Agent orchestration | LangGraph | Stateful multi-agent workflow, streaming node updates, checkpoints, human review gate. |
| LLM framework | LangChain | OpenAI/OpenRouter-compatible chat model integration. |
| Vector database | FAISS | Top-k retrieval over uploaded evidence and Tavily threat-intelligence results. |
| Web search | Tavily | Authorized security and CVE source search lanes. |
| UI | Streamlit | Interactive upload, run, trace, scorecard, and report download experience. |
| Language | Python | Agents, deterministic scanners, evals, and deployment entrypoint. |
| Deployment | Streamlit Community Cloud | Free-tier friendly root `app.py` entrypoint. |

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Optional `.env` values:

```bash
OPENAI_API_KEY=your_openrouter_or_openai_key
TAVILY_API_KEY=your_tavily_key
LLM_PROVIDER=openrouter
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_MODEL=openai/gpt-4o-mini
MAX_WEB_RESULTS=8
MAX_RETRIEVAL_DOCS=3
VALIDATOR_TOP_K=3
REPORT_REFLECTION_RETRY_LIMIT=2
REQUIRE_HUMAN_REVIEW=false
CHECKPOINT_THREAD_ID=cybersecurity-agent-default
```

Without API keys, the app still runs locally using deterministic scanners, hash embeddings for FAISS, authorized-source reference documents, and offline evaluation cases.

## Streamlit Cloud deployment

1. Push the repository branch to GitHub.
2. Create a Streamlit Community Cloud app from the repo.
3. Set the app file to `app.py`.
4. Add secrets for `OPENAI_API_KEY`, `TAVILY_API_KEY`, `LLM_PROVIDER`, `OPENAI_BASE_URL`, and `OPENAI_MODEL` if available.
5. Deploy.

See `docs/free-hosting.md` for more detail.

## Windows target path

The requested destination path is:

```powershell
G:\Outskill\Hackathon\AgentsRepo\CyberSecurityAIAgent
```

This Linux cloud workspace cannot directly write to `G:\`, so clone or copy this repository folder to that path on Windows, then run:

```powershell
cd G:\Outskill\Hackathon\AgentsRepo\CyberSecurityAIAgent
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Evaluation and loop engineering

The project includes a local eval harness inspired by agent-evaluation best practices: define scenario cases, run the full graph, score the resulting state, and feed improvement actions into future runs.

Run tests and evals locally:

```bash
pytest
```

In the UI, click **Run offline eval suite** to execute deterministic cases for:

- brute force plus SQL injection detection,
- Docker and database hardening review.

The eval loop scores:

- specialist-agent coverage,
- finding actionability,
- severity triage,
- authorized CVE source grounding,
- incident-response completeness,
- policy-control mapping.

## Security and financial-services guardrails

- This is a defensive assistant, not an autonomous enforcement system.
- Require human approval before blocking production traffic, rotating privileged credentials, or notifying regulators.
- Do not paste secrets, customer data, payment data, or regulated personal data into public deployments.
- Validate findings with approved scanners, SIEM queries, and change-management processes before production action.
- Preserve audit trails for evidence, agent outputs, human decisions, and remediation closure.

## Project layout

```text
.
├── app.py
├── requirements.txt
├── pyproject.toml
├── docs/
│   └── free-hosting.md
├── src/deep_researcher/
│   ├── agent/
│   │   ├── llm.py
│   │   ├── prompts.py
│   │   └── security_agents.py
│   ├── config/
│   │   ├── concepts.json
│   │   ├── concepts.py
│   │   ├── settings.py
│   │   └── system_prompts.yaml
│   ├── main/
│   │   ├── evaluation.py
│   │   ├── models.py
│   │   ├── streamlit_app.py
│   │   └── workflow.py
│   └── tools/
│       ├── embeddings.py
│       ├── registry.py
│       ├── retrieval.py
│       └── search.py
└── tests/
    └── test_offline_workflow.py
```
