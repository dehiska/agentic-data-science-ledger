# 🧠 Agentic DS Ledger

An end-to-end agentic system that **parses data science files** (notebooks, scripts, docs), tracks experiment history in a structured ledger, and uses a **multi-agent AI pipeline** powered by Claude to suggest the next best improvement — with human-in-the-loop review.

> **Live demo:** deployed on GCP Cloud Run + Supabase

---

## Features

| Feature | Description |
|---------|-------------|
| 📂 **File Parsing** | Upload `.ipynb`, `.py`, `.docx`, or `.json` — AI extracts models, metrics, preprocessing steps |
| 🔍 **Inference Engine** | Confidence-scored extraction with per-field and overall confidence % |
| ✅ **Confirmation UI** | Review, edit, and confirm detected experiments before they enter the ledger |
| 📜 **Ledger** | Full experiment history with project grouping, delete, and status tracking |
| 🏆 **Leaderboard** | Rank experiments by any metric across projects and model families |
| 🌳 **Experiment Tree** | Plotly graph of experiments over time, color-coded by author |
| 🤖 **Multi-Agent Planning** | 5-agent pipeline generates a validated improvement plan |
| 🧑‍⚖️ **LLM Judge** | Validates the plan and scores it 0–100 before showing to user |
| 📄 **Spec Export** | Download a `.part2.agent.revisions.md` spec file from any plan |
| ⚡ **Autoresearch** | AutoML search for the best model configuration |
| 💰 **Cost Dashboard** | Tracks agent plan LLM costs and autoresearch compute costs |
| 🔗 **GitHub Integration** | Tag experiments with repo, branch, and author |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  Frontend  │  Streamlit UI (frontend/app.py)                     │
│            │  Upload · Ledger · Tree · Plan · Costs · Snippet    │
├──────────────────────────────────────────────────────────────────┤
│  Backend   │  FastAPI REST API (backend/api.py)                  │
│            │  /projects · /files/parse · /ledger · /plan · /costs│
├──────────────────────────────────────────────────────────────────┤
│ Middleware │  src/                                                │
│            │  MCP Server · Inference Engine · RAG · 5 Agents · DB│
└──────────────────────────────────────────────────────────────────┘
```

| Phase | Database | Hosting |
|-------|----------|---------|
| 1 — Local | SQLite (auto-created) | `localhost` |
| 2 — Cloud | Supabase (PostgreSQL) | GCP Cloud Run |

---

## Quick Start — Local (Phase 1)

### 1. Clone and install

```bash
git clone https://github.com/dehiska/agentic-data-science-ledger.git
cd agentic-data-science-ledger
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
copy .env.example .env   # Windows
cp .env.example .env     # Mac/Linux
```

Edit `.env`:
```
ANTHROPIC_API_KEY=sk-ant-...   # Required for AI planning
APP_MODE=local
```

### 3. Run

```bash
# Windows one-click:
run_local.bat

# Or manually — two terminals:
# Terminal 1
uvicorn backend.api:app --reload --port 8080
# Terminal 2
streamlit run frontend/app.py --server.port 8501
```

Open **http://localhost:8501** in your browser.

---

## Usage

### Adding Experiments

**Option A — AI Parse:** Upload any `.ipynb`, `.py`, or `.docx` — the AI detects models, metrics, and preprocessing steps with confidence scoring. Review and confirm in the UI.

**Option B — Exact Metrics:** Copy the logging cell from the **📋 Notebook Snippet** tab into the bottom of your notebook. Run it to generate a `*_ledger_entry.json` file, then upload that JSON for 100% accurate metric values.

### Workflow

1. **Create a project** in the sidebar
2. **Upload files** → AI parses → confirm detected experiments
3. **View Ledger** → leaderboard ranks best models by metric
4. **Experiment Tree** → visualize experiment history by author over time
5. **Agent Plan** → set a goal → generate a 5-agent validated plan
6. **Review** → Approve / Revise / Reject → download spec `.md` file
7. **Costs** → see total LLM + compute spend broken down by type

---

## Multi-Agent Pipeline

```
Goal + Ledger State
        │
        ▼
┌─────────────────┐
│ Lead Scientist  │  Proposes next steps, model suggestions, search strategy
└────────┬────────┘
         │
┌────────▼────────┐
│   EDA Agent     │  Preprocessing gaps, feature engineering suggestions
└────────┬────────┘
         │
┌────────▼────────┐
│   DNN Agent     │  Uncertainty methods (only if DNN detected)
└────────┬────────┘
         │
┌────────▼────────┐
│ Cost Estimator  │  GCP instance + time + cost estimate
└────────┬────────┘
         │
┌────────▼────────┐
│   LLM Judge     │  Validates plan, scores 0–100, flags issues
└─────────────────┘
```

Powered by **Claude claude-3-5-haiku** via LangChain.

---

## Project Structure

```
agentic-ds-ledger/
├── src/
│   ├── mcp_server.py              # AST + regex parsing (.ipynb/.py/.docx/.json)
│   ├── inference_engine.py        # Confidence scoring of extracted metadata
│   ├── rag_system.py              # RAG FAISS vector store
│   ├── database.py                # LocalDatabase (SQLite) + CloudDatabase (Supabase)
│   ├── multi_agent_orchestrator.py
│   ├── autoresearch_wrapper.py
│   └── agents/
│       ├── lead_scientist.py
│       ├── eda_agent.py
│       ├── dnn_agent.py
│       ├── cost_estimator.py
│       └── llm_judge.py
│
├── backend/
│   └── api.py                     # FastAPI REST API
│
├── frontend/
│   └── app.py                     # Streamlit UI (6 tabs)
│
├── Dockerfile.backend
├── Dockerfile.frontend
├── cloudbuild.yaml                # GCP Cloud Build CI/CD pipeline
├── supabase_setup.sql             # Supabase schema (run once in SQL editor)
├── rag_manual.json                # DS best-practice RAG knowledge base
├── requirements.txt
└── .env.example
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Recommended | Enables Claude-powered planning. Rule-based fallback works without it. |
| `APP_MODE` | Optional | `local` (default) or `cloud` |
| `BACKEND_URL` | Cloud only | FastAPI URL (set automatically by Cloud Build) |
| `SUPABASE_URL` | Cloud only | Supabase project URL |
| `SUPABASE_KEY` | Cloud only | Supabase service role key |
| `GITHUB_TOKEN` | Optional | GitHub PAT — tags experiments with repo info |

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/projects` | Create project |
| `GET` | `/projects` | List projects |
| `DELETE` | `/projects/{id}` | Delete project |
| `POST` | `/files/parse` | Upload + parse file (multipart) |
| `POST` | `/files/parse-local` | Parse by local path |
| `GET` | `/ledger` | List ledger entries |
| `GET` | `/ledger/{id}` | Get single entry |
| `DELETE` | `/ledger/{id}` | Delete entry |
| `POST` | `/infer` | Run inference engine on metadata |
| `POST` | `/confirm/{id}` | Confirm / store user-corrected experiment |
| `GET` | `/experiments` | List confirmed experiments |
| `POST` | `/plan/generate` | Run multi-agent plan |
| `POST` | `/autoresearch/run` | Run AutoML search |
| `GET` | `/costs` | List cost logs |

Interactive docs: **http://localhost:8080/docs**

---

## Cloud Deployment (Phase 2 — GCP Cloud Run)

### Prerequisites
- GCP project with billing enabled
- Artifact Registry repo created
- Supabase project created

### One-time setup

```bash
# Enable APIs
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com

# Grant Cloud Build service account permissions
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:YOUR_COMPUTE_SA@developer.gserviceaccount.com" \
  --role="roles/run.admin"
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:YOUR_COMPUTE_SA@developer.gserviceaccount.com" \
  --role="roles/artifactregistry.writer"
gcloud iam service-accounts add-iam-policy-binding \
  YOUR_COMPUTE_SA@developer.gserviceaccount.com \
  --member="serviceAccount:YOUR_COMPUTE_SA@developer.gserviceaccount.com" \
  --role="roles/iam.serviceAccountUser"
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:YOUR_COMPUTE_SA@developer.gserviceaccount.com" \
  --role="roles/logging.logWriter"
```

### Supabase schema

Run `supabase_setup.sql` in your Supabase SQL editor. If tables already exist, run the `ALTER TABLE` statements at the bottom of the file to add new columns.

### Cloud Build trigger

Connect your GitHub repo to Cloud Build and set these substitution variables:

| Variable | Value |
|----------|-------|
| `_REGION` | `us-central1` |
| `_REPO` | your Artifact Registry repo name |
| `_APP_MODE` | `cloud` |
| `_SUPABASE_URL` | your Supabase project URL |
| `_SUPABASE_KEY` | your Supabase service role key |
| `_ANTHROPIC_API_KEY` | your Anthropic API key |

Every `git push` to the trigger branch automatically builds + deploys both services.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Frontend | Streamlit |
| Backend | FastAPI + Uvicorn |
| LLM | Anthropic Claude (claude-3-5-haiku) via LangChain |
| RAG | LangChain + FAISS + sentence-transformers |
| File parsing | `nbformat` + Python `ast` + `python-docx` |
| Phase 1 DB | SQLite (auto-migrating) |
| Phase 2 DB | Supabase (PostgreSQL) |
| Charts | Plotly |
| CI/CD | GCP Cloud Build |
| Hosting | GCP Cloud Run |
| Containers | Docker |
| GitHub | PyGithub |
