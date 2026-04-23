# Agentic DS Ledger — Three-Agent Autoresearch Pipeline
### Spec v1.1 · April 2026
**Repo:** https://github.com/dehiska/agentic-data-science-ledger

---

## Overview

This spec describes the next evolution of the Agentic DS Ledger: a fully automated three-agent pipeline that takes a dataset, searches for the best model configuration, and delivers a ready-to-run optimised notebook — all logged in real time to the existing ledger and traceable from the Streamlit dashboard.

The pipeline runs entirely on the existing infrastructure (GCP Cloud Run + Supabase + Streamlit) with no new frameworks or services required.

---

## Goals

- **Automate the full experiment loop** — from raw dataset to optimised model to ledger entry — with no manual steps
- **Stratified sampling** — Agent A works on a representative subset so the loop is fast even on million-row datasets
- **Full observability** — every agent action is logged to a trace feed visible in the dashboard in real time
- **Zero new infrastructure** — build on top of what already exists

---

## How It Works

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│   User clicks "Run Pipeline" in Agent Plan tab                     │
│                                                                     │
│        ┌──────────────────────────────────────────────┐            │
│        │  Agent A — AutoResearcher                    │            │
│        │  src/autoresearch_wrapper.py                 │            │
│        │                                              │            │
│        │  1. Stratified-sample the dataset            │            │
│        │  2. Loop: try model variants                 │            │
│        │  3. Keep best metric result                  │            │
│        │  4. Log each iteration → traces table        │            │
│        └──────────────────┬───────────────────────────┘            │
│                           │  best_model + params                   │
│        ┌──────────────────▼───────────────────────────┐            │
│        │  Agent B — Orchestrator                      │            │
│        │  src/multi_agent_orchestrator.py             │            │
│        │                                              │            │
│        │  1. Receives Agent A result                  │            │
│        │  2. Runs Lead Scientist, EDA, DNN,           │            │
│        │     Cost Estimator, LLM Judge (existing)     │            │
│        │  3. Validates plan, scores 0–100             │            │
│        │  4. Forwards to Agent C                      │            │
│        └──────────────────┬───────────────────────────┘            │
│                           │  validated plan + best params          │
│        ┌──────────────────▼───────────────────────────┐            │
│        │  Agent C — Executor                          │            │
│        │  src/agents/executor_agent.py  ← NEW         │            │
│        │                                              │            │
│        │  1. Generates optimised .ipynb notebook      │            │
│        │  2. Creates a new ledger entry               │            │
│        │  3. Logs completion trace                    │            │
│        └──────────────────────────────────────────────┘            │
│                                                                     │
│   Streamlit Trace Log tab shows the full run live                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Agents

### Agent A — AutoResearcher
**File:** `src/autoresearch_wrapper.py` (existing, enhanced)

**Responsibility:** Search for the best model and hyperparameters on a stratified sample of the dataset.

**Input:**
```json
{
  "file_path": "notebooks/experiment.ipynb",
  "goal": "Maximise F1 on imbalanced classes",
  "max_sample_rows": 10000,
  "stratify": true,
  "max_iterations": 50,
  "plateau_stop": 10
}
```

**New behaviour:**
- If the dataset has more than `max_sample_rows` rows, use `sklearn.model_selection.train_test_split` with `stratify=y` to create a representative subset before searching
- Log one trace entry per iteration (model tried, metric achieved, whether it is a new best)
- Stop on the first of: max iterations reached, wall time exceeded (5 min), or metric plateau for N consecutive iterations

**Output (unchanged shape):**
```json
{
  "status": "success",
  "best_model": {
    "name": "LightGBMClassifier",
    "family": "Boosting",
    "params": { "n_estimators": 300, "learning_rate": 0.05 },
    "metrics": { "f1": 0.879, "accuracy": 0.912 }
  },
  "simulated": false,
  "iterations_run": 23
}
```

---

### Agent B — Orchestrator
**File:** `src/multi_agent_orchestrator.py` (existing, extended)

**Responsibility:** Receive Agent A's result, run the existing five-agent validation pipeline, then hand off to Agent C.

**What already exists (unchanged):**
- Lead Scientist → proposes next steps
- EDA Agent → preprocessing suggestions
- DNN Agent → uncertainty methods
- Cost Estimator → resource estimate
- LLM Judge → scores the plan 0–100

**New addition:**
- After autoresearch completes with `status == "success"`, call `ExecutorAgent.execute()`
- Log an orchestration trace entry at handoff

---

### Agent C — Executor
**File:** `src/agents/executor_agent.py` ← **new file**

**Responsibility:** Take the validated best model and generate a ready-to-run optimised notebook, then register it in the ledger.

**Steps:**
1. Load the source notebook (identified by `file_path` from the ledger entry) using `nbformat`
2. Inject a new code cell at the end with the optimised hyperparameters pre-filled
3. Add a logging cell (same format as the Notebook Snippet tab) so the result auto-logs when run
4. Save as `{original_name}_optimised_{run_id}.ipynb`
5. Call `MCPServer.store_in_db()` to create a new ledger entry for the generated notebook
6. Log a completion trace

**Output:**
```json
{
  "status": "ok",
  "notebook_path": "experiment_optimised_a1b2c3.ipynb",
  "entry_id": 14,
  "injected_params": { "n_estimators": 300, "learning_rate": 0.05 }
}
```

---

## New Database Table — `traces`

Added to `src/database.py` and `supabase_setup.sql`.

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
```

| Column | Purpose |
|--------|---------|
| `run_id` | Groups all traces from a single pipeline run (UUID generated at start) |
| `agent` | `autoresearcher` · `orchestrator` · `executor` |
| `message` | Human-readable description of what happened |
| `payload` | JSON: model tried, metric, params, entry_id, etc. |
| `entry_id` | Links the trace to a specific ledger entry (optional) |

**New methods on `LocalDatabase` and `CloudDatabase`:**
- `log_trace(run_id, agent, message, payload=None, entry_id=None)`
- `get_traces(run_id=None, limit=200)`

---

## New API Endpoint

Added to `backend/api.py`:

```
GET /traces?run_id=<uuid>&limit=200
```

Returns the trace log for a given run (or all recent traces if no `run_id`).

---

## New Frontend Tab — Trace Log

Added to `frontend/app.py` between Costs and Notebook Snippet:

```
📂 Upload Files  |  📜 Ledger  |  🌳 Experiment Tree  |  🤖 Agent Plan  |  💰 Costs  |  🔍 Trace Log  |  📋 Notebook Snippet
```

**UI:**
```
🔍 Trace Log

Latest Run ▼   [🔄 Refresh]

  22:01:00  🔬 autoresearcher   Sampling 10,000 rows (stratified) from 1.3M dataset
  22:01:05  🔬 autoresearcher   Iteration  1/50 — RandomForest    acc=0.841
  22:01:09  🔬 autoresearcher   Iteration  4/50 — XGBClassifier   acc=0.863  ✅ new best
  22:01:18  🔬 autoresearcher   Iteration  7/50 — LightGBM        acc=0.879  ✅ new best
  22:01:44  🔬 autoresearcher   Plateau reached after 10 iterations. Stopping.
  22:01:45  🧠 orchestrator     Agent A complete → LightGBM acc=0.879. Running plan validation...
  22:01:52  🧠 orchestrator     LLM Judge score: 94/100 APPROVED. Forwarding to Executor.
  22:01:53  ⚙️  executor         Generating optimised notebook from experiment.ipynb...
  22:01:54  ⚙️  executor         Notebook saved: experiment_optimised_a1b2c3.ipynb
  22:01:54  ⚙️  executor         Ledger entry #14 created.
```

Each row is one `traces` table entry. The tab auto-refreshes with a `st.rerun()` button and shows the run selector dropdown.

---

## Known Gaps & Decisions

| Gap | Risk | Decision |
|-----|------|----------|
| **Data loading** | How is the dataset loaded from the notebook? | Use `nbformat` to extract data-loading cells from the source notebook and reuse the same logic in `autoresearch_wrapper.py`. Agent A re-executes those cells to get `X_train` / `y_train` before sampling. |
| **Non-tabular data** | What if the notebook uses images or text? | **Skip** — the pipeline only runs on tabular datasets. Agent A checks for a DataFrame shape; if not found, it exits gracefully with `status: "skipped"` and logs a trace. |
| **Model serialization** | How is `best_model` passed from Agent A to Agent B? | Use JSON only — no file serialization. Hyperparameters, metric values, and model name are all JSON-serializable. If the user later wants to download the trained model object, a **Download Model** button is added to the Trace Log tab that serializes on demand (`joblib` for scikit-learn, `torch.save` for PyTorch). Otherwise the pipeline stays lightweight and stateless. |
| **Parallelization** | Slow for large hyperparameter spaces. | **Future improvement** — use `joblib.Parallel` or Ray Tune for parallel trials. Not in v1. |

---

## Stopping Conditions for Agent A

| Condition | Default | Configurable |
|-----------|---------|--------------|
| Max iterations | 50 | Yes |
| Max wall time | 5 minutes | Yes |
| Metric plateau (no improvement for N iterations) | 10 | Yes |
| Metric threshold reached (e.g. acc > 0.95) | None | Yes |

The first condition to trigger wins. The best result found up to that point is passed to Agent B.

---

## Files Changed

| File | Change |
|------|--------|
| `src/database.py` | Add `traces` table migration, `log_trace()`, `get_traces()` |
| `src/autoresearch_wrapper.py` | Add stratified sampling + per-iteration trace logging |
| `src/agents/executor_agent.py` | **New** — Agent C |
| `src/agents/__init__.py` | Export `ExecutorAgent` |
| `src/multi_agent_orchestrator.py` | Wire in `ExecutorAgent` after autoresearch |
| `backend/api.py` | Add `GET /traces` endpoint |
| `frontend/app.py` | Add `tab_traces` — Trace Log tab |
| `supabase_setup.sql` | Add `traces` table + index |

**Nothing else changes.** No new Docker images, no new Cloud Run services, no new dependencies beyond what is already in `requirements.txt`.

---

## Implementation Order

1. `src/database.py` — traces table + methods
2. `src/autoresearch_wrapper.py` — stratified sampling + trace logging
3. `src/agents/executor_agent.py` — Agent C
4. `src/multi_agent_orchestrator.py` — wire Agent C in
5. `backend/api.py` — `/traces` endpoint
6. `frontend/app.py` — Trace Log tab
7. `supabase_setup.sql` — schema update
8. Push → Cloud Build auto-deploys both services

---

## What We Are Not Building

| Idea | Reason Skipped |
|------|----------------|
| `modelcontextprotocol/python-sdk` | Our FastAPI + orchestrator already handle inter-agent communication |
| Redis / RabbitMQ | The `traces` DB table + polling is sufficient at this scale |
| MLflow | Our `ledger` table already tracks every experiment with full metadata |
| Separate Docker container per agent | One backend container runs all agents — no benefit to splitting |
| React frontend | Streamlit already serves the UI and is already deployed |
| Gradio | Same — already have Streamlit |

---

*Built on top of Agentic DS Ledger v2.1*
*Repo: https://github.com/dehiska/agentic-data-science-ledger*
