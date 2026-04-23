# Spec: Three-Agent Autoresearch Pipeline
*Updated to match the existing Agentic DS Ledger tech stack*

---

## 1. High-Level Workflow

```
Agent A (AutoResearcher)
  → stratified sample → finds best model/params
        │
        ▼
Agent B (Middleman / Orchestrator)
  → validates, logs trace, coordinates
        │
        ▼
Agent C (Executor/Builder)
  → generates notebook, logs result to ledger
        │
        ▼
Streamlit Frontend
  → live trace log tab shows the full run
```

---

## 2. Tech Stack Mapping

| Spec Component | ~~Mistral Suggestion~~ | **What We Already Have** |
|----------------|------------------------|--------------------------|
| Agent A | Fork `autoresearch` | `src/autoresearch_wrapper.py` — enhance with stratified sampling |
| Agent B | New Python + MCP SDK | `src/multi_agent_orchestrator.py` — already coordinates all agents |
| Agent C | New Python + MCP SDK | New `src/agents/executor_agent.py` — added to our existing agents |
| MCP Server (inter-agent) | `modelcontextprotocol/python-sdk` | Our `MultiAgentOrchestrator` + FastAPI (`backend/api.py`) handle coordination — no extra SDK needed |
| MCP Server (file parsing) | — | `src/mcp_server.py` — already parses `.ipynb`/`.py`/`.docx`/`.json` |
| Frontend | New Streamlit / Gradio / React | `frontend/app.py` — add a **Trace Log** tab to the existing Streamlit app |
| Database | SQLite / PostgreSQL / MLflow | `src/database.py` — `LocalDatabase` (SQLite) + `CloudDatabase` (Supabase) — add `traces` table |
| Message Queue | Redis / RabbitMQ | **Not needed** — DB-backed trace log + FastAPI endpoint is sufficient |
| LLM | Any | Anthropic `claude-3-5-haiku` via LangChain (already wired) |
| Hosting | Any VM | GCP Cloud Run — already deployed |
| Containers | New Dockerfiles | `Dockerfile.backend` + `Dockerfile.frontend` — already exist |
| CI/CD | Manual | `cloudbuild.yaml` — auto-deploys on `git push` |
| Experiment tracking | MLflow | Our `ledger` table in SQLite/Supabase — already tracks everything |

---

## 3. What Needs to Be Built

Only **three additions** on top of what exists:

### 3.1 — Enhance `src/autoresearch_wrapper.py` (Agent A)
Add stratified sampling support:

```python
def run_autoresearch(
    self,
    ...
    stratify: bool = True,
    max_sample_rows: int = 10_000,
) -> Dict:
```

- Accepts `X_train`, `y_train` directly (or loads from ledger entry)
- If `stratify=True` and dataset > `max_sample_rows`, uses `sklearn.model_selection.train_test_split` with `stratify=y`
- Logs each iteration to the new `traces` table (see 3.3)
- Stopping condition: N iterations OR metric improvement plateau (configurable)
- Output: same existing dict shape `{status, best_model, simulated, ...}` — no breaking changes

---

### 3.2 — New `src/agents/executor_agent.py` (Agent C)
Receives the best model/params from Agent B (orchestrator) and:

1. Generates a new `.ipynb` notebook using `nbformat` (already in requirements) with the optimised hyperparameters pre-filled
2. Stores the generated notebook path + params in a new ledger entry via `src/mcp_server.py`
3. Logs a trace entry (experiment complete)

```python
class ExecutorAgent(BaseAgent):
    def execute(self, best_model: dict, source_entry_id: int, goal: str) -> dict:
        """Generate notebook + log result to ledger."""
        notebook = self._generate_notebook(best_model)
        entry_id = self._store_in_ledger(notebook, best_model, source_entry_id)
        self._log_trace("executor", f"Generated notebook for {best_model['name']}", entry_id)
        return {"status": "ok", "notebook_path": notebook, "entry_id": entry_id}
```

---

### 3.3 — Add `traces` Table to `src/database.py`

```sql
CREATE TABLE IF NOT EXISTS traces (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT NOT NULL,          -- groups all traces from one pipeline run
    agent       TEXT NOT NULL,          -- 'autoresearcher' | 'orchestrator' | 'executor'
    message     TEXT NOT NULL,
    payload     TEXT DEFAULT NULL,      -- JSON: best_model, metric, params, etc.
    entry_id    INTEGER REFERENCES ledger(id) ON DELETE SET NULL,
    timestamp   DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

New methods on `LocalDatabase` and `CloudDatabase`:
- `log_trace(run_id, agent, message, payload=None, entry_id=None)`
- `get_traces(run_id=None, limit=200)`

---

### 3.4 — Trace Log Tab in `frontend/app.py`

Add a **🔍 Trace Log** tab between Costs and Notebook Snippet:

```
tab_upload | tab_ledger | tab_tree | tab_plan | tab_costs | tab_traces | tab_snippet
```

Displays a live feed of all agent messages for the latest (or selected) pipeline run:

```
🔍 Trace Log

Run ID: [selectbox — latest runs]   [🔄 Refresh]

2026-04-23 22:01:00  autoresearcher   Sampling 10,000 rows (stratified) from 1.3M
2026-04-23 22:01:05  autoresearcher   Iteration 1/50 — XGBClassifier acc=0.841
2026-04-23 22:01:12  autoresearcher   Iteration 7/50 — LightGBM acc=0.879 ✅ new best
2026-04-23 22:01:45  orchestrator     Agent A complete. Best: LightGBM acc=0.879
2026-04-23 22:01:46  orchestrator     Forwarding to Executor...
2026-04-23 22:01:47  executor         Generated notebook: lgbm_optimised.ipynb
2026-04-23 22:01:48  executor         Ledger entry #12 created
```

---

## 4. Updated Agent Roles

### Agent A — AutoResearcher (`src/autoresearch_wrapper.py`)
- **Input:** file path (from ledger) + goal + sampling config
- **Does:** stratified sample → AutoML loop → finds best model
- **Logs:** one trace per iteration to `traces` table
- **Output:** `{best_model, best_metric, params, simulated}`

### Agent B — Orchestrator (`src/multi_agent_orchestrator.py`)
- **Input:** Agent A result + original ledger state
- **Does:** already runs Lead Scientist, EDA, DNN, Cost Estimator, LLM Judge — **now also calls Agent C**
- **Logs:** orchestration decision traces
- **Output:** full plan dict + executor result

### Agent C — Executor (`src/agents/executor_agent.py`) ← **new**
- **Input:** best_model dict from Agent B
- **Does:** generates `.ipynb` with `nbformat`, stores in ledger, logs trace
- **Output:** `{notebook_path, entry_id}`

---

## 5. New API Endpoint

Add to `backend/api.py`:

```python
@app.get("/traces")
def get_traces(run_id: Optional[str] = None, limit: int = 200):
    db, _, _, _ = get_services()
    return {"traces": db.get_traces(run_id=run_id, limit=limit)}
```

---

## 6. Updated Orchestrator Flow

```python
# In MultiAgentOrchestrator.run_pipeline():

# Existing steps 1–7 unchanged ...

# NEW step 8: Executor
if execute_autoresearch and autoresearch_result.get("status") == "success":
    executor = ExecutorAgent(self._get_rag(), self.llm)
    executor_result = executor.execute(
        best_model=autoresearch_result["best_model"],
        source_entry_id=ledger_state.get("entry_id"),
        goal=goal,
    )
    result["executor_result"] = executor_result
```

---

## 7. Database Schema Addition (Supabase)

Add to `supabase_setup.sql` and run in SQL editor:

```sql
CREATE TABLE IF NOT EXISTS traces (
    id          SERIAL PRIMARY KEY,
    run_id      TEXT NOT NULL,
    agent       TEXT NOT NULL,
    message     TEXT NOT NULL,
    payload     JSONB DEFAULT NULL,
    entry_id    INTEGER REFERENCES ledger(id) ON DELETE SET NULL,
    timestamp   TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_traces_run_id   ON traces(run_id);
CREATE INDEX IF NOT EXISTS idx_traces_timestamp ON traces(timestamp DESC);
```

---

## 8. Stopping Condition for Agent A

| Condition | Default |
|-----------|---------|
| Max iterations | 50 |
| Max wall time | 5 minutes |
| Plateau (no improvement for N iterations) | 10 iterations |
| Early stop if metric > threshold | configurable per goal |

The first condition to trigger wins. The best result at stop time is passed to Agent B.

---

## 9. What We Are NOT Building

| Mistral Suggested | Why We Skip It |
|-------------------|----------------|
| `modelcontextprotocol/python-sdk` | Our FastAPI + orchestrator already handle inter-agent comms |
| Redis / RabbitMQ message queue | Overkill — DB-backed trace log is sufficient |
| MLflow | Our ledger already tracks experiments with full metadata |
| Separate Streamlit frontend for traces | Add a tab to the existing app instead |
| Separate Docker container per agent | One backend container already runs all agents |
| Gradio / React | We already have Streamlit |

---

## 10. Implementation Order

1. `src/database.py` — add `traces` table migration + `log_trace()` / `get_traces()`
2. `src/autoresearch_wrapper.py` — add stratified sampling + trace logging per iteration
3. `src/agents/executor_agent.py` — new Agent C
4. `src/multi_agent_orchestrator.py` — wire in Agent C after autoresearch
5. `backend/api.py` — add `GET /traces` endpoint
6. `frontend/app.py` — add Trace Log tab
7. `supabase_setup.sql` — add traces table (+ manual `ALTER TABLE` for existing DB)
8. `cloudbuild.yaml` — nothing to change (auto-deploys on push)

---

*All new code builds directly on top of the existing Agentic DS Ledger v2.1 codebase.*
*No new infrastructure, no new frameworks, no new hosting.*
