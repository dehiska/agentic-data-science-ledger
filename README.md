# Agentic DS Ledger

An end-to-end agentic system that **parses Jupyter notebooks**, tracks experiment history in a ledger, and uses a **multi-agent AI workflow** to suggest the next best DS improvement — with a human-in-the-loop approval step.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Frontend  │  Streamlit UI (frontend/app.py)                │
│            │  Upload notebooks · View ledger · HITL review  │
├─────────────────────────────────────────────────────────────┤
│  Backend   │  FastAPI REST API (backend/api.py)             │
│            │  /parse · /ledger · /plan · /autoresearch       │
├─────────────────────────────────────────────────────────────┤
│ Middleware │  src/                                           │
│            │  MCP Server · RAG System · 6 Agents · DB        │
└─────────────────────────────────────────────────────────────┘
```

| Phase | Database | Storage | Hosting |
|-------|----------|---------|---------|
| 1 (local) | SQLite | Local files | `localhost` |
| 2 (cloud) | Supabase | GitHub repos | GCP Compute Engine |

---

## Quick Start (Phase 1 — Local)

### 1. Clone & enter the project
```bash
cd C:\Users\owner\Downloads\agentic-ds-ledger
```

### 2. Activate the virtual environment
```bash
# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Copy and configure `.env`
```bash
copy .env.example .env
# Edit .env — add your OPENAI_API_KEY (optional, rule-based fallback works without it)
```

### 4. Run the app
```bash
# Windows one-click:
run_local.bat

# Or manually:
# Terminal 1 — Backend
uvicorn backend.api:app --reload --port 8000

# Terminal 2 — Frontend
streamlit run frontend/app.py --server.port 8501
```

Open **http://localhost:8501** in your browser.

---

## Usage

1. **Upload a notebook** — drag a `.ipynb` file onto the Upload tab
2. **View the Ledger** — see all parsed experiments with models, metrics, preprocessing
3. **Generate a Plan** — click "Generate Plan" to run the multi-agent pipeline:
   - Lead Scientist proposes next steps
   - EDA Agent flags preprocessing gaps
   - DNN Agent adds uncertainty methods (if DNN detected)
   - Cost Estimator calculates GCP cost
   - LLM Judge validates the plan
4. **Review & Approve** — Approve / Request Revision / Reject (Human-in-the-Loop)
5. **Execute autoresearch** — toggle ON to run AutoML search and log the best model

---

## Project Structure

```
agentic-ds-ledger/
├── venv/                          # Virtual environment
├── data/
│   └── example.ipynb              # Sample notebook to test with
│
├── src/                           # Middleware layer
│   ├── mcp_server.py              # AST parsing + metadata extraction
│   ├── rag_system.py              # RAG FAISS vector store
│   ├── database.py                # SQLite (Phase 1) / Supabase (Phase 2)
│   ├── github_integration.py      # GitHub API wrapper
│   ├── multi_agent_orchestrator.py
│   ├── autoresearch_wrapper.py
│   └── agents/
│       ├── base_agent.py
│       ├── lead_scientist.py      # Proposes DS improvement plan
│       ├── eda_agent.py           # Preprocessing & EDA suggestions
│       ├── dnn_agent.py           # DNN / uncertainty analysis
│       ├── cost_estimator.py      # GCP cost estimation
│       └── llm_judge.py           # Plan validation
│
├── backend/
│   └── api.py                     # FastAPI REST API
│
├── frontend/
│   └── app.py                     # Streamlit UI
│
├── rag_manual.json                # 15 DS best-practice nodes (RAG knowledge base)
├── supabase_setup.sql             # Phase 2 — run in Supabase SQL editor
├── Dockerfile.backend
├── Dockerfile.frontend
├── docker-compose.yml
├── startup_script.sh              # GCP Compute Engine deploy script
├── requirements.txt
└── .env.example
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Optional | Enables LLM-powered planning. Without it, rule-based agents still work. |
| `APP_MODE` | Optional | `local` (default) or `cloud` |
| `BACKEND_URL` | Optional | FastAPI URL for frontend (default: `http://localhost:8000`) |
| `SUPABASE_URL` | Phase 2 | Supabase project URL |
| `SUPABASE_KEY` | Phase 2 | Supabase anon key |
| `GITHUB_TOKEN` | Phase 2 | GitHub PAT with `repo` scope |
| `GCP_PROJECT_ID` | Phase 2 | GCP project ID |

---

## Phase 2 — Cloud Setup

### Supabase
1. Create project at [supabase.com](https://supabase.com)
2. Go to **SQL Editor** → paste contents of `supabase_setup.sql` → Run
3. Copy `SUPABASE_URL` and `SUPABASE_KEY` from **Project Settings → API** into `.env`

### GitHub
1. Create a GitHub repo for notebooks (e.g. `your-username/ds-notebooks`)
2. Generate a PAT at GitHub → Settings → Developer settings → Personal access tokens (scope: `repo`)
3. Add as `GITHUB_TOKEN` in `.env`

### GCP Compute Engine
```bash
# In Google Cloud Shell
gcloud compute instances create agentic-ds-ledger \
  --machine-type=n1-standard-2 \
  --image-family=debian-11 \
  --image-project=debian-cloud \
  --tags=http-server \
  --metadata-from-file startup-script=startup_script.sh
```
Edit `startup_script.sh` first to set your env vars and repo URL.

### Docker Compose (local or any VM)
```bash
cp .env.example .env  # fill in values
docker-compose up --build
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/notebooks/parse` | Upload + parse `.ipynb` |
| `POST` | `/notebooks/parse-local` | Parse by local file path |
| `GET` | `/ledger` | List all experiments |
| `GET` | `/ledger/{id}` | Get single experiment |
| `POST` | `/plan/generate` | Run multi-agent plan |
| `POST` | `/autoresearch/run` | Run AutoML search |
| `GET` | `/costs` | List cost logs |

Interactive docs at **http://localhost:8000/docs** when backend is running.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Frontend | Streamlit |
| Backend | FastAPI + Uvicorn |
| LLM | OpenAI GPT-4o-mini (optional) |
| RAG | LangChain + FAISS + sentence-transformers |
| Notebook parsing | `nbformat` + Python `ast` |
| Phase 1 DB | SQLite |
| Phase 2 DB | Supabase (PostgreSQL) |
| GitHub | PyGithub |
| Cloud | GCP Compute Engine |
| Containers | Docker + Docker Compose |
