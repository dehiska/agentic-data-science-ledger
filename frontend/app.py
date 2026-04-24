"""
Agentic DS Ledger — Streamlit Frontend

Supports Phase 1 (direct imports + SQLite) and Phase 2 (FastAPI + Supabase).

Run:
    streamlit run frontend/app.py
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Config ─────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Agentic DS Ledger",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_MODE = os.getenv("APP_MODE", "local")
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8080")
SUPPORTED_TYPES = [".ipynb", ".py", ".docx", ".json", ".csv"]

FAMILY_OPTIONS = [
    "Ensemble", "Linear", "Deep Neural Network", "Boosting",
    "Tree", "Bayesian", "Clustering", "Other",
]

# ── Service layer ──────────────────────────────────────────────────────────────

@st.cache_resource
def get_local_services():
    from dotenv import load_dotenv
    load_dotenv()
    from src.database import get_db
    from src.mcp_server import MCPServer
    from src.multi_agent_orchestrator import MultiAgentOrchestrator
    db = get_db()
    mcp = MCPServer(db=db)
    orchestrator = MultiAgentOrchestrator(
        db=db,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        github_token=os.getenv("GITHUB_TOKEN"),
    )
    return db, mcp, orchestrator


def api(method: str, path: str, silent: bool = False, **kwargs):
    import httpx
    try:
        r = httpx.request(method, f"{BACKEND_URL}{path}", timeout=60.0, **kwargs)
        if r.status_code == 409:
            # Duplicate — return structured error, don't toast
            return {"_error": 409, "detail": r.json().get("detail", "Already exists.")}
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        if not silent:
            st.error(f"API error: {e}")
        return None
    except Exception as e:
        if not silent:
            st.error(f"API error: {e}")
        return None


def get_db_direct():
    db, _, _ = get_local_services()
    return db


def get_mcp_direct():
    _, mcp, _ = get_local_services()
    return mcp


def get_orch_direct():
    _, _, orch = get_local_services()
    return orch


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_projects():
    if APP_MODE == "cloud":
        r = api("GET", "/projects")
        return r.get("projects", []) if r else []
    return get_db_direct().get_projects()


def create_project_action(name: str, description: str):
    if APP_MODE == "cloud":
        return api("POST", "/projects", silent=True, json={"name": name, "description": description})
    import sqlite3
    db = get_db_direct()
    try:
        pid = db.create_project(name, description)
        return {"id": pid, "name": name}
    except sqlite3.IntegrityError:
        return {"_error": 409, "detail": f"Project '{name}' already exists."}


def delete_project_action(project_id: int):
    if APP_MODE == "cloud":
        api("DELETE", f"/projects/{project_id}")
    else:
        get_db_direct().delete_project(project_id)


def parse_uploaded_file(uploaded, project_id):
    ext = Path(uploaded.name).suffix.lower()
    content = uploaded.getvalue()

    if APP_MODE == "cloud":
        params = {}
        if project_id:
            params["project_id"] = project_id
        gh_repo = st.session_state.get("gh_repo", "").strip()
        gh_branch = st.session_state.get("gh_branch", "main").strip()
        gh_user = st.session_state.get("gh_user", "").strip()
        if gh_repo:
            params["github_repo"] = gh_repo
            params["github_branch"] = gh_branch or "main"
        if gh_user:
            params["team_member"] = gh_user
        return api("POST", "/files/parse",
                   files={"file": (uploaded.name, content, "application/octet-stream")},
                   params=params)

    # Phase 1: direct
    from src.inference_engine import InferenceEngine
    mcp = get_mcp_direct()
    db = get_db_direct()
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False, mode="wb") as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    try:
        meta = mcp.parse_file(tmp_path)
        meta["file_path"] = uploaded.name
        entry_id = mcp.store_in_db(meta, project_id=project_id)
        infer_result = InferenceEngine().infer(meta)
        db.update_ledger_entry(entry_id, {
            "auto_extracted": infer_result,
            "confidence_score": infer_result["overall_confidence"],
        })
        return {
            "status": "ok", "entry_id": entry_id,
            "file_type": ext.lstrip("."),
            "models_found": len(meta.get("models", [])),
            "metrics_found": len(meta.get("metrics", [])),
            "preprocessing_found": len(meta.get("preprocessing", [])),
            "confidence": infer_result["overall_confidence"],
            "word_count": meta.get("word_count"),
            "doc_keywords": meta.get("doc_keywords"),
            "metadata": meta,
            "experiment": infer_result,
        }
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def confirm_entry_action(entry_id: int, user_corrected: dict):
    if APP_MODE == "cloud":
        return api("POST", f"/confirm/{entry_id}", json={"user_corrected": user_corrected})
    db = get_db_direct()
    db.confirm_entry(entry_id, user_corrected)
    return {"status": "confirmed", "entry_id": entry_id}


def delete_ledger_entry_action(entry_id: int):
    if APP_MODE == "cloud":
        api("DELETE", f"/ledger/{entry_id}")
    else:
        get_db_direct().delete_ledger_entry(entry_id)


def build_leaderboard(entries: list) -> list:
    """Extract rows with numeric metric values for the leaderboard."""
    rows = []
    for e in entries:
        uc = e.get("user_corrected") or {}
        if isinstance(uc, str):
            try:
                import json as _json
                uc = _json.loads(uc)
            except Exception:
                uc = {}
        metrics = uc.get("metrics") or e.get("metrics") or []
        models = uc.get("models") or e.get("models") or []
        model_name = models[0]["name"] if models else "—"
        model_family = models[0].get("family", "Other") if models else "Other"
        author = uc.get("team_member") or e.get("team_member") or "—"
        for m in metrics:
            val = m.get("value")
            if isinstance(val, (int, float)):
                rows.append({
                    "Entry": f"#{e['id']}",
                    "File": e.get("file_path", "?"),
                    "Model": model_name,
                    "Family": model_family or "Other",
                    "Metric": m["name"],
                    "Value": round(val, 4),
                    "Author": author,
                    "Status": e.get("status", "Pending"),
                })
    return rows


def load_ledger(project_id=None):
    if APP_MODE == "cloud":
        params = {}
        if project_id:
            params["project_id"] = project_id
        r = api("GET", "/ledger", params=params)
        return r.get("entries", []) if r else []
    return get_db_direct().get_ledger_entries(project_id=project_id)


# ── Header ─────────────────────────────────────────────────────────────────────

st.title("🧠 Agentic DS Ledger")
st.caption(f"Mode: **{APP_MODE.upper()}**")

# ── Sidebar: Project management ────────────────────────────────────────────────

with st.sidebar:
    st.header("📁 Projects")

    # Create new project
    with st.expander("➕ New Project", expanded=False):
        new_name = st.text_input("Project name", key="new_proj_name")
        new_desc = st.text_area("Description (optional)", key="new_proj_desc", height=68)
        if st.button("Create Project", type="primary", key="btn_create_proj"):
            if not new_name.strip():
                st.warning("Enter a project name.")
            else:
                result = create_project_action(new_name.strip(), new_desc.strip())
                if result and result.get("_error") == 409:
                    st.warning(f'A project named "**{new_name}**" already exists. Choose a different name.')
                elif result:
                    st.success(f"Created: **{new_name}**")
                    st.rerun()
                else:
                    st.error("Failed to create project. Please try again.")

    st.divider()

    # Project selector
    projects = load_projects()
    project_options = {f"{p['name']} ({p.get('file_count', 0)} files)": p["id"] for p in projects}
    project_options = {"All projects": None, **project_options}

    selected_label = st.selectbox("View project", list(project_options.keys()), key="proj_select")
    selected_project_id = project_options[selected_label]

    # Delete project button (not for "All projects")
    if selected_project_id is not None:
        if st.button("🗑️ Delete this project", key="btn_del_proj"):
            delete_project_action(selected_project_id)
            st.success("Project deleted.")
            st.rerun()

    st.divider()

    # GitHub settings
    with st.expander("🔗 GitHub Username (required for Leaderboard)", expanded=True):
        st.caption("Optional — used to tag experiments with repo info and your username.")
        repo_name = st.text_input("GitHub Repo", placeholder="username/repo-name", key="gh_repo")
        branch = st.text_input("Branch", value="main", key="gh_branch")
        team_member = st.text_input("Your GitHub Username", key="gh_user")
        if st.button("💾 Save GitHub Settings", key="btn_save_gh"):
            if repo_name.strip():
                st.session_state["gh_saved"] = True
                st.success(f"✅ Connected: `{repo_name}` ({branch})")
            else:
                st.warning("Enter a GitHub repo first.")
        if st.session_state.get("gh_saved") and repo_name.strip():
            st.caption(f"🟢 Connected to `{repo_name}`")

    st.divider()

# ── Tabs ───────────────────────────────────────────────────────────────────────

tab_upload, tab_leaderboard, tab_ledger, tab_tree, tab_plan, tab_swarm, tab_costs, tab_traces, tab_snippet = st.tabs([
    "📂 Upload Files",
    "🏆 Leaderboard",
    "📜 Ledger",
    "🌳 Experiment Tree",
    "🤖 Agent Plan",
    "🐝 Swarm AutoML",
    "💰 Costs",
    "🔍 Trace Log",
    "📋 Notebook Snippet",
])

# ── Tab 1: Upload ──────────────────────────────────────────────────────────────

with tab_upload:
    st.header("Upload Files")

    st.info(
        "**Two ways to add experiments:**\n\n"
        "**Option A — AI Parse:** Upload any `.ipynb`, `.py`, or `.docx` file and the AI will "
        "automatically detect models, metrics, and preprocessing steps (best effort — confidence scored).\n\n"
        "**Option B — Exact Metrics:** Add the logging cell from the **📋 Notebook Snippet** tab to "
        "the end of your notebook, run it to save a `*_ledger_entry.json`, then upload that JSON here "
        "for 100% accurate metric values and full feature importances.",
        icon="💡",
    )

    # Project picker for upload
    upload_proj_options = {p["name"]: p["id"] for p in projects}
    upload_proj_options = {"No project (unassigned)": None, **upload_proj_options}

    _up_col1, _up_col2 = st.columns([2, 1])
    with _up_col1:
        upload_proj_label = st.selectbox("Assign to project", list(upload_proj_options.keys()), key="upload_proj")
    with _up_col2:
        _upload_author = st.text_input(
            "Your GitHub Username ✱",
            value=st.session_state.get("gh_user", ""),
            placeholder="e.g. dehiska",
            key="upload_author_field",
            help="Required — appears in the Leaderboard and Experiment Tree.",
        )
        if _upload_author.strip():
            st.session_state["gh_user"] = _upload_author.strip()
        elif not st.session_state.get("gh_user"):
            st.warning("⚠️ Enter your GitHub username so your name appears in the Leaderboard.")

    upload_proj_id = upload_proj_options[upload_proj_label]

    st.caption("Supported: `.ipynb` notebooks · `.py` scripts · `.docx` documents · `.json` ledger entries · `.csv` datasets")
    uploaded_files = st.file_uploader(
        "Upload files",
        type=["ipynb", "py", "docx", "doc", "json", "csv"],
        accept_multiple_files=True,
        key="file_uploader",
    )

    # Session state: list of dicts {entry_id, experiment, file_name}
    if "pending_confirmations" not in st.session_state:
        st.session_state["pending_confirmations"] = []

    if uploaded_files and st.button("📋 Parse All Files", type="primary"):
        for uploaded in uploaded_files:
            with st.spinner(f"Parsing {uploaded.name}..."):
                result = parse_uploaded_file(uploaded, upload_proj_id)

            if result and result.get("status") == "ok":
                ftype = result.get("file_type", "?")
                icon = {"ipynb": "📓", "py": "🐍", "docx": "📄"}.get(ftype, "📁")
                conf = result.get("confidence", 0)
                st.success(
                    f"{icon} **{uploaded.name}** — entry #{result.get('entry_id')} "
                    f"| confidence {conf:.0%}"
                )

                c1, c2, c3 = st.columns(3)
                c1.metric("Models", result["models_found"])
                c2.metric("Metrics", result["metrics_found"])
                c3.metric("Preprocessing", result["preprocessing_found"])

                if result.get("word_count"):
                    st.caption(f"Word count: {result['word_count']}")
                if result.get("doc_keywords"):
                    st.caption(f"DS keywords found: {', '.join(result['doc_keywords'][:10])}")

                with st.expander("Full metadata"):
                    st.json(result.get("metadata", {}))

                # Queue for confirmation
                if result.get("experiment"):
                    st.session_state["pending_confirmations"].append({
                        "entry_id": result["entry_id"],
                        "experiment": result["experiment"],
                        "file_name": uploaded.name,
                    })
            else:
                st.error(f"Failed to parse {uploaded.name}")

    # ── Confirmation UI ────────────────────────────────────────────────────────
    if st.session_state.get("pending_confirmations"):
        st.divider()
        st.subheader("📋 Review Detected Experiments")
        st.caption("Confirm or edit what the parser found. Confirmed entries are used by the agent pipeline.")

        confirmed_indices = []

        for idx, pending in enumerate(st.session_state["pending_confirmations"]):
            exp = pending["experiment"]
            entry_id = pending["entry_id"]
            fname = pending["file_name"]
            conf = exp.get("overall_confidence", 0)

            with st.expander(
                f"📊 {fname}  —  confidence {conf:.0%}  |  entry #{entry_id}",
                expanded=True,
            ):
                # Models
                st.markdown("**🤖 Models**")
                edited_models = []
                for mi, m in enumerate(exp.get("models", [])):
                    mc1, mc2, mc3 = st.columns([3, 3, 1])
                    name_val = mc1.text_input(
                        "Name", value=m["name"],
                        key=f"conf_mname_{entry_id}_{mi}",
                    )
                    family_val = mc2.selectbox(
                        "Family",
                        FAMILY_OPTIONS,
                        index=FAMILY_OPTIONS.index(m["family"]) if m["family"] in FAMILY_OPTIONS else len(FAMILY_OPTIONS) - 1,
                        key=f"conf_mfam_{entry_id}_{mi}",
                    )
                    mc3.caption(f"{m.get('confidence', 0):.0%}")
                    edited_models.append({**m, "name": name_val, "family": family_val})

                if not exp.get("models"):
                    st.caption("No models detected.")

                # Metrics — checkbox + value input per metric
                st.markdown("**📊 Metrics**")
                edited_metrics = []
                for mi, m in enumerate(exp.get("metrics", [])):
                    mc1, mc2, mc3 = st.columns([3, 2, 1])
                    include = mc1.checkbox(
                        m["name"], value=True, key=f"conf_met_inc_{entry_id}_{mi}"
                    )
                    existing_val = m.get("value")
                    if isinstance(existing_val, (int, float)):
                        val = mc2.number_input(
                            "Value", value=float(existing_val),
                            format="%.4f", key=f"conf_met_val_{entry_id}_{mi}",
                            label_visibility="collapsed",
                        )
                    elif existing_val is None:
                        val = mc2.number_input(
                            "Value", value=0.0,
                            format="%.4f", key=f"conf_met_val_{entry_id}_{mi}",
                            label_visibility="collapsed",
                        )
                    else:
                        val = existing_val  # complex value (e.g. classification_report dict)
                        mc2.caption("(complex value)")
                    mc3.caption(f"{m.get('confidence', 0):.0%}")
                    if include:
                        edited_metrics.append({**m, "value": val})
                if not exp.get("metrics"):
                    st.caption("No metrics detected.")

                # Preprocessing
                st.markdown("**⚙️ Preprocessing**")
                all_prep_names = [p["name"] for p in exp.get("preprocessing", [])]
                selected_prep = st.multiselect(
                    "Preprocessing steps",
                    options=all_prep_names,
                    default=all_prep_names,
                    key=f"conf_prep_{entry_id}",
                )

                # Task type
                task_type_options = ["classification", "regression", "clustering", "unknown"]
                current_task = exp.get("task_type", "unknown")
                task_idx = task_type_options.index(current_task) if current_task in task_type_options else 3
                edited_task = st.selectbox(
                    "Task type",
                    task_type_options,
                    index=task_idx,
                    key=f"conf_task_{entry_id}",
                )

                cb1, cb2 = st.columns(2)
                if cb1.button("✅ Confirm & Save", key=f"confirm_btn_{entry_id}", type="primary"):
                    user_corrected = {
                        "models": edited_models,
                        "metrics": edited_metrics,
                        "preprocessing": [
                            p for p in exp.get("preprocessing", []) if p["name"] in selected_prep
                        ],
                        "hyperparameters": exp.get("hyperparameters", {}),
                        "task_type": edited_task,
                        "overall_confidence": conf,
                    }
                    confirm_entry_action(entry_id, user_corrected)
                    st.success(f"Entry #{entry_id} confirmed!")
                    confirmed_indices.append(idx)

                if cb2.button("⏭ Skip", key=f"skip_btn_{entry_id}"):
                    st.info("Skipped — entry saved as unconfirmed.")
                    confirmed_indices.append(idx)

        # Remove handled confirmations (reverse order to preserve indices)
        for idx in sorted(confirmed_indices, reverse=True):
            st.session_state["pending_confirmations"].pop(idx)

# ── Tab 2: Leaderboard ─────────────────────────────────────────────────────────

with tab_leaderboard:
    import pandas as pd
    import plotly.express as px

    st.header("🏆 Team Leaderboard")
    st.caption("All confirmed experiments ranked by metric — across every project and team member.")

    lb_col1, lb_col2 = st.columns([4, 1])
    with lb_col2:
        if st.button("🔄 Refresh", key="refresh_lb"):
            st.rerun()

    all_lb_entries = load_ledger(project_id=None)

    def _build_full_leaderboard(entries: list) -> list:
        rows = []
        for e in entries:
            # Prefer user-confirmed data; fall back to raw parse
            uc = e.get("user_corrected") or {}
            if isinstance(uc, str):
                try:
                    import json as _j; uc = _j.loads(uc)
                except Exception:
                    uc = {}

            metrics      = uc.get("metrics") or e.get("metrics") or []
            models       = uc.get("models")  or e.get("models")  or []
            model_name   = models[0]["name"]   if models else "—"
            model_family = models[0].get("family", "Other") if models else "Other"

            # Author: team_member column takes priority
            author = (e.get("team_member") or
                      uc.get("team_member") or
                      st.session_state.get("gh_user") or "—")

            # AutoML used? — swarm CSV entries, or flaml flag in model dict
            flaml_used = any(
                m.get("flaml") is True for m in models
            ) if models else False
            is_automl = (
                e.get("file_type") == "csv"   # came from Swarm
                or flaml_used
                or e.get("status") == "Executed"   # came from autoresearch
            )
            automl_label = "✅ FLAML" if flaml_used else ("🔬 AutoML" if is_automl else "—")

            for m in metrics:
                val = m.get("value")
                if isinstance(val, (int, float)):
                    rows.append({
                        "Entry":    f"#{e['id']}",
                        "Dataset":  e.get("file_path", "?"),
                        "Model":    model_name,
                        "Family":   model_family or "Other",
                        "Metric":   m["name"],
                        "Score":    round(float(val), 4),
                        "Author":   author,
                        "AutoML":   automl_label,
                        "Project":  e.get("project_name", "—"),
                        "Status":   e.get("status", "Pending"),
                        "_approved": bool(e.get("approved")),
                    })
        return rows

    lb_rows = _build_full_leaderboard(all_lb_entries)

    if not lb_rows:
        st.info(
            "No scored experiments yet.\n\n"
            "Upload a file in **📂 Upload Files** and confirm metrics, "
            "or run the **🐝 Swarm AutoML** pipeline — results appear here automatically."
        )
    else:
        lb_df = pd.DataFrame(lb_rows)

        # ── Filter controls ────────────────────────────────────────────────────
        fc1, fc2, fc3, fc4 = st.columns([2, 2, 2, 1])
        metric_names   = sorted(lb_df["Metric"].unique())
        sel_metric     = fc1.selectbox("Metric", metric_names, key="lb2_metric")
        higher_better  = fc2.checkbox("Higher = better", value=True, key="lb2_higher")
        approved_only  = fc3.checkbox("Confirmed entries only", value=False, key="lb2_approved")
        group_family   = fc4.checkbox("Best/family", value=False, key="lb2_family")

        filtered = lb_df[lb_df["Metric"] == sel_metric].copy()
        if approved_only:
            filtered = filtered[filtered["_approved"]]
        filtered = filtered.sort_values("Score", ascending=not higher_better).reset_index(drop=True)

        if group_family:
            agg_fn = "idxmax" if higher_better else "idxmin"
            idx = filtered.groupby("Family")["Score"].agg(agg_fn)
            filtered = filtered.loc[idx.values].sort_values(
                "Score", ascending=not higher_better
            ).reset_index(drop=True)

        filtered.insert(0, "Rank", range(1, len(filtered) + 1))

        # ── Podium metrics ─────────────────────────────────────────────────────
        if len(filtered) >= 1:
            p_cols = st.columns(min(3, len(filtered)))
            medals = ["🥇", "🥈", "🥉"]
            for i, col in enumerate(p_cols):
                row = filtered.iloc[i]
                col.metric(
                    f"{medals[i]} {row['Author']}",
                    f"{row['Score']:.4f}",
                    delta=row["Model"],
                )

        st.divider()

        # ── Full table ─────────────────────────────────────────────────────────
        display_cols = ["Rank", "Dataset", "Model", "Family",
                        "Score", "Author", "AutoML", "Project", "Status", "Entry"]
        st.dataframe(
            filtered[display_cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Score":  st.column_config.NumberColumn(format="%.4f"),
                "AutoML": st.column_config.TextColumn(width="small"),
                "Rank":   st.column_config.NumberColumn(width="small"),
            },
        )

        # ── Score distribution by author ───────────────────────────────────────
        if filtered["Author"].nunique() > 1:
            with st.expander("📊 Score by team member", expanded=False):
                fig = px.box(
                    filtered, x="Author", y="Score", color="Author",
                    points="all",
                    title=f"{sel_metric} distribution by team member",
                )
                fig.update_layout(height=340, margin=dict(t=40, b=20), showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        # ── AutoML vs manual breakdown ─────────────────────────────────────────
        with st.expander("🤖 AutoML vs Manual breakdown", expanded=False):
            automl_counts = filtered["AutoML"].value_counts().reset_index()
            automl_counts.columns = ["Type", "Count"]
            fig2 = px.pie(automl_counts, names="Type", values="Count",
                          title="AutoML vs Manual entries",
                          color_discrete_sequence=["#4C9BE8", "#EF553B", "#636EFA"])
            fig2.update_layout(height=300, margin=dict(t=40, b=10))
            st.plotly_chart(fig2, use_container_width=True)


# ── Tab 3: Ledger ──────────────────────────────────────────────────────────────

with tab_ledger:
    proj_label = selected_label if selected_project_id else "All Projects"
    st.header(f"📜 Ledger — {proj_label}")

    if st.button("🔄 Refresh", key="refresh_ledger"):
        st.rerun()

    entries = load_ledger(project_id=selected_project_id)

    if not entries:
        st.info("No files in this project yet. Upload files in the Upload tab.")
    else:
        st.caption(f"{len(entries)} file(s)")

        # ── Leaderboard ────────────────────────────────────────────────────────
        lb_rows = build_leaderboard(entries)
        if lb_rows:
            with st.expander("🏆 Leaderboard", expanded=True):
                import pandas as pd
                lb_df = pd.DataFrame(lb_rows)
                metric_names = sorted(lb_df["Metric"].unique())
                lc1, lc2, lc3 = st.columns([2, 1, 1])
                rank_metric = lc1.selectbox("Rank by metric", metric_names, key="lb_metric")
                higher_better = lc2.checkbox("Higher is better", value=True, key="lb_higher")
                group_by_family = lc3.checkbox("Best by family", value=False, key="lb_family")

                filtered = lb_df[lb_df["Metric"] == rank_metric].copy()
                filtered = filtered.sort_values("Value", ascending=not higher_better).reset_index(drop=True)

                if group_by_family:
                    # Keep the best row per model family
                    idx = (
                        filtered.groupby("Family")["Value"]
                        .agg("idxmax" if higher_better else "idxmin")
                    )
                    filtered = filtered.loc[idx.values].sort_values(
                        "Value", ascending=not higher_better
                    ).reset_index(drop=True)

                filtered.insert(0, "Rank", range(1, len(filtered) + 1))
                display_cols = ["Rank", "File", "Model", "Family", "Value", "Author", "Status", "Entry"]
                st.dataframe(
                    filtered[display_cols],
                    use_container_width=True,
                    hide_index=True,
                )
            st.divider()

        for entry in entries:
            ftype = entry.get("file_type", "ipynb")
            icon = {"ipynb": "📓", "py": "🐍", "docx": "📄"}.get(ftype, "📁")
            models = entry.get("models") or []
            metrics = entry.get("metrics") or []
            preprocessing = entry.get("preprocessing") or []
            proj_name = entry.get("project_name", "—")

            label = (
                f"{icon} {entry.get('file_path', '?')}  |  "
                f"Project: {proj_name}  |  "
                f"{len(models)} model(s)  |  {entry.get('timestamp', '')[:16]}"
            )
            with st.expander(label, expanded=False):
                col_info, col_content = st.columns([1, 2])

                with col_info:
                    st.write(f"**Type:** `{ftype}`")
                    st.write(f"**Project:** {proj_name}")
                    st.write(f"**Status:** {entry.get('status', '—')}")
                    st.write(f"**Logged:** {entry.get('timestamp', '—')[:19]}")
                    if st.button("🗑️ Delete entry", key=f"del_entry_{entry.get('id')}"):
                        delete_ledger_entry_action(entry["id"])
                        st.success("Entry deleted.")
                        st.rerun()

                with col_content:
                    if models:
                        st.subheader("🤖 Models")
                        for m in models:
                            st.write(f"- **{m.get('name', '?')}** ({m.get('family', '?')})")
                            if m.get("params"):
                                st.code(str(m["params"]), language="python")

                    if metrics:
                        st.subheader("📊 Metrics")
                        for met in metrics:
                            val = met.get("value")
                            val_str = f" = **{val:.4f}**" if isinstance(val, (int, float)) else ""
                            st.write(f"- `{met.get('name', met.get('function', '?'))}`{val_str}"
                                     + (f" — line {met['line_number']}" if met.get("line_number") else ""))

                    if preprocessing:
                        st.subheader("⚙️ Preprocessing")
                        for p in preprocessing:
                            st.write(f"- **{p.get('name', '?')}** ({p.get('type', '?')})")

                    # For docx: show raw text preview
                    if ftype == "docx" and entry.get("raw_text"):
                        st.subheader("📄 Document Preview")
                        st.text(entry["raw_text"][:800] + ("..." if len(entry.get("raw_text","")) > 800 else ""))

# ── Tab 4: Experiment Tree ────────────────────────────────────────────────────

_AUTHOR_COLORS = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
    "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
]


with tab_tree:
    import plotly.graph_objects as go
    from datetime import datetime as _dt

    st.header("🌳 Experiment Tree")
    st.caption("Each node is a confirmed experiment. Color = author. Y-axis = selected metric value.")

    tree_entries = load_ledger(project_id=selected_project_id)

    if not tree_entries:
        st.info("No entries yet. Upload and confirm experiments first.")
    else:
        # Collect all numeric metric names across entries
        tree_lb_rows = build_leaderboard(tree_entries)
        all_tree_metrics = sorted({r["Metric"] for r in tree_lb_rows}) if tree_lb_rows else []

        if not all_tree_metrics:
            st.info("No confirmed entries with numeric metric values yet. Confirm entries in the Upload tab first.")
        else:
            tree_metric = st.selectbox("Metric for Y-axis", all_tree_metrics, key="tree_metric")

            # Build node data
            node_data = []
            for e in sorted(tree_entries, key=lambda x: x.get("timestamp", "")):
                uc = e.get("user_corrected") or {}
                if isinstance(uc, str):
                    try:
                        import json as _json2
                        uc = _json2.loads(uc)
                    except Exception:
                        uc = {}
                metrics_list = uc.get("metrics") or e.get("metrics") or []
                models_list = uc.get("models") or e.get("models") or []
                model_name = models_list[0]["name"] if models_list else "—"
                author = uc.get("team_member") or e.get("team_member") or "Unknown"

                # Find value for selected metric
                metric_val = None
                for m in metrics_list:
                    if m.get("name") == tree_metric and isinstance(m.get("value"), (int, float)):
                        metric_val = m["value"]
                        break

                try:
                    ts = _dt.fromisoformat(e.get("timestamp", "").replace("Z", ""))
                except Exception:
                    ts = _dt(2000, 1, 1)

                node_data.append({
                    "id": e["id"],
                    "file": e.get("file_path", "?"),
                    "model": model_name,
                    "author": author,
                    "ts": ts,
                    "value": metric_val,
                    "status": e.get("status", "Pending"),
                    "confidence": e.get("confidence_score") or 0,
                    "project_id": e.get("project_id"),
                })

            # Author → color map
            unique_authors = sorted({n["author"] for n in node_data})
            color_map = {a: _AUTHOR_COLORS[i % len(_AUTHOR_COLORS)] for i, a in enumerate(unique_authors)}

            fig = go.Figure()

            # Draw edges (project-grouped, ordered by timestamp)
            from itertools import groupby
            project_groups = {}
            for nd in node_data:
                pid = nd["project_id"] or 0
                project_groups.setdefault(pid, []).append(nd)

            for pid, group in project_groups.items():
                group_sorted = sorted(group, key=lambda x: x["ts"])
                for i in range(len(group_sorted) - 1):
                    a, b = group_sorted[i], group_sorted[i + 1]
                    ya = a["value"] if a["value"] is not None else 0
                    yb = b["value"] if b["value"] is not None else 0
                    fig.add_trace(go.Scatter(
                        x=[a["ts"], b["ts"]], y=[ya, yb],
                        mode="lines",
                        line=dict(color="#444", width=1, dash="dot"),
                        showlegend=False,
                        hoverinfo="skip",
                    ))

            # Draw nodes per author (for legend grouping)
            for author in unique_authors:
                author_nodes = [n for n in node_data if n["author"] == author]
                xs = [n["ts"] for n in author_nodes]
                ys = [n["value"] if n["value"] is not None else 0 for n in author_nodes]
                texts = []
                for n in author_nodes:
                    _vstr = f"{n['value']:.4f}" if n["value"] is not None else "—"
                    texts.append(
                        f"#{n['id']} {n['model']}<br>{tree_metric}: {_vstr}<br>"
                        f"Author: {n['author']}<br>Status: {n['status']}<br>"
                        f"Confidence: {n['confidence']:.0%}"
                    )
                symbols = [
                    "circle" if n["value"] is not None else "circle-open"
                    for n in author_nodes
                ]
                fig.add_trace(go.Scatter(
                    x=xs, y=ys,
                    mode="markers+text",
                    marker=dict(size=16, color=color_map[author], symbol=symbols),
                    text=[f"#{n['id']}" for n in author_nodes],
                    textposition="top center",
                    hovertext=texts,
                    hoverinfo="text",
                    name=author,
                ))

            fig.update_layout(
                xaxis_title="Time",
                yaxis_title=tree_metric,
                legend_title="Author",
                height=480,
                margin=dict(l=40, r=20, t=30, b=40),
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Open circles = no confirmed value for this metric. Connect lines = same project, ordered by time.")


# ── Tab 5: Agent Plan ──────────────────────────────────────────────────────────

def _build_spec_md(selected_file: str, goal: str, plan: dict, validation: dict, autoresearch: dict) -> str:
    from datetime import datetime as _dt
    ts = _dt.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    base = Path(selected_file).stem
    lines = [
        f"# Experiment Spec: {base}.part2.agent.revisions",
        f"",
        f"**Generated:** {ts}  ",
        f"**Source File:** `{selected_file}`  ",
        f"**Goal:** {goal}  ",
        f"**LLM Judge Score:** {validation.get('score', '?')}/100 ({validation.get('verdict', '?')})",
        f"",
        f"---",
        f"",
        f"## 📋 Recommended Next Steps",
        f"",
    ]
    for i, step in enumerate(plan.get("next_steps", []), 1):
        lines.append(f"{i}. {step}")
    lines += ["", "## 🤖 Suggested Models", ""]
    for m in plan.get("suggested_models", []):
        lines.append(f"- `{m}`")
    lines += ["", f"## 🔍 Search Strategy", "", plan.get("suggested_search_strategy", "—"), ""]
    lines += ["", "## 💡 Rationale", "", plan.get("rationale", "—"), ""]
    if plan.get("eda_suggestions"):
        lines += ["", "## 🔬 EDA Suggestions", ""]
        for sug in plan["eda_suggestions"]:
            lines.append(f"- **{sug.get('category')}** ({sug.get('priority','?')}): {sug.get('action')}")
            if sug.get("code_hint"):
                lines += [f"  ```python", f"  {sug['code_hint']}", f"  ```"]
    if autoresearch and autoresearch.get("status") == "success":
        best = autoresearch.get("best_model", {})
        lines += ["", "## ⚡ Autoresearch Best Result", ""]
        lines.append(f"**Model:** {best.get('name')} ({best.get('family')})")
        for k, v in best.get("metrics", {}).items():
            lines.append(f"- {k.upper()}: {v:.4f}")
        if autoresearch.get("simulated"):
            lines.append(f"\n> ⚠️ Simulated result — install autoresearch package for real AutoML.")
    cost = plan.get("resource_estimate", {})
    if cost:
        lines += ["", "## 💰 Cost Estimate", ""]
        lines.append(f"- **Instance:** {cost.get('gcp_instance','—')}")
        lines.append(f"- **Hours:** {cost.get('time_hours','—')}")
        lines.append(f"- **Est. Cost:** ${cost.get('total_cost', 0):.3f}")
    lines += ["", "---", f"*Generated by Agentic DS Ledger — {ts}*"]
    return "\n".join(lines)


with tab_plan:
    # Header row with LLM Judge label
    h_col, judge_col = st.columns([4, 1])
    with h_col:
        st.header("🤖 Multi-Agent Planning")
    with judge_col:
        st.markdown(
            "<div style='text-align:right; padding-top:18px;'>"
            "<span style='background:#1e3a5f; color:#7eb8f7; padding:4px 10px; "
            "border-radius:6px; font-size:0.75rem;'>🧑‍⚖️ Validated by LLM Judge</span>"
            "</div>",
            unsafe_allow_html=True,
        )

    goal = st.text_area(
        "🎯 Goal",
        value=st.session_state.get("goal_input", "Improve accuracy on the test set"),
        height=70,
        key="goal_input",
        help="Describe what you want to optimize. E.g. 'Maximize accuracy for multiclass classification'",
    )
    execute_ar = st.toggle("⚡ Execute autoresearch", value=False, key="exec_ar")

    all_entries = load_ledger(project_id=selected_project_id)
    file_options = [e.get("file_path", "?") for e in all_entries]

    if not file_options:
        st.warning("No files in ledger. Upload files first.")
    else:
        selected_file = st.selectbox("Select file for planning", file_options, key="plan_file_select")

        if st.button("🚀 Generate Plan", type="primary", key="btn_generate_plan"):
            with st.spinner("Running multi-agent pipeline..."):
                if APP_MODE == "cloud":
                    result = api("POST", "/plan/generate", json={
                        "file_path": selected_file, "goal": goal,
                        "project_id": selected_project_id,
                        "repo_name": st.session_state.get("gh_repo") or None,
                        "branch": st.session_state.get("gh_branch", "main"),
                        "team_member": st.session_state.get("gh_user") or None,
                        "execute_autoresearch": execute_ar,
                    })
                else:
                    orch = get_orch_direct()
                    result = orch.run_pipeline(
                        file_path=selected_file, goal=goal,
                        execute_autoresearch=execute_ar,
                    )
                    # Log agent-plan cost locally (cloud mode logs it server-side)
                    if result:
                        try:
                            cost_est = result.get("plan", {}).get("resource_estimate", {})
                            total = float(cost_est.get("total_cost") or 0)
                            if total > 0:
                                get_db_direct().log_cost(
                                    plan_id=None,
                                    gcp_instance=cost_est.get("gcp_instance", "claude-3-5-haiku"),
                                    time_hours=float(cost_est.get("time_hours") or 0),
                                    cost=total,
                                    cost_type="agent_plan",
                                    description=(goal or "")[:120],
                                )
                        except Exception:
                            pass
            if result:
                st.session_state["plan_result"] = result
                st.session_state["plan_file"] = selected_file
                st.session_state["plan_goal"] = goal
                st.session_state["plan_review"] = None  # reset review state

        # Render plan from session state so it persists across button clicks
        result = st.session_state.get("plan_result")
        _sel_file = st.session_state.get("plan_file", selected_file)
        _sel_goal = st.session_state.get("plan_goal", goal)

        if result:
            plan = result.get("plan", {})
            validation = result.get("validation", {})
            autoresearch = result.get("autoresearch_result")

            verdict = validation.get("verdict", "UNKNOWN")
            score = validation.get("score", 0)
            if "REJECTED" in verdict:
                st.error(f"❌ {verdict} — Score: {score}/100")
                st.caption("The LLM Judge found issues with this plan. Review the validation issues below, update your goal, and regenerate.")
            elif "WARNINGS" in verdict:
                st.warning(f"⚠️ {verdict} — Score: {score}/100")
            else:
                st.success(f"✅ {verdict} — Score: {score}/100")

            col_plan, col_meta = st.columns([3, 2])

            with col_plan:
                st.subheader("📋 Next Steps")
                for i, step in enumerate(plan.get("next_steps", []), 1):
                    st.write(f"{i}. {step}")

                if plan.get("eda_suggestions"):
                    st.subheader("🔬 EDA Suggestions")
                    for sug in plan["eda_suggestions"]:
                        icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(sug.get("priority", "medium"), "⚪")
                        st.write(f"{icon} **{sug.get('category')}**: {sug.get('action')}")
                        if sug.get("code_hint"):
                            st.code(sug["code_hint"], language="python")

                if plan.get("dnn_suggestions"):
                    st.subheader("🧠 DNN Suggestions")
                    for sug in plan["dnn_suggestions"]:
                        st.write(f"- **{sug.get('category')}**: {sug.get('action')}")
                        if sug.get("code_hint"):
                            st.code(sug["code_hint"], language="python")

                if plan.get("uncertainty_methods"):
                    st.subheader("❓ Uncertainty Methods")
                    for m in plan["uncertainty_methods"]:
                        st.write(f"- {m}")

            with col_meta:
                st.subheader("🤖 Suggested Models")
                for m in plan.get("suggested_models", []):
                    st.write(f"- `{m}`")

                st.subheader("🔍 Search Strategy")
                st.write(plan.get("suggested_search_strategy", "—"))

                st.subheader("💡 Rationale")
                st.write(plan.get("rationale", "—"))

                cost = plan.get("resource_estimate", {})
                if cost:
                    st.subheader("💰 Cost Estimate")
                    st.metric("Instance", cost.get("gcp_instance", "—"))
                    st.metric("Hours", cost.get("time_hours", "—"))
                    st.metric("Est. Cost", f"${cost.get('total_cost', 0):.3f}")

            if validation.get("issues") or validation.get("warnings"):
                with st.expander("⚠️ Validation Issues"):
                    for i in validation.get("issues", []):
                        st.write(f"- [{i.get('severity','?').upper()}] {i.get('message','?')}")
                    for w in validation.get("warnings", []):
                        st.write(f"- [WARN] {w.get('message','?')}")

            if autoresearch:
                st.subheader("⚡ Autoresearch Result")
                if autoresearch.get("status") == "success":
                    best = autoresearch.get("best_model", {})
                    st.success(f"Best: **{best.get('name')}** ({best.get('family')})")
                    m = best.get("metrics", {})
                    if m:
                        cols = st.columns(len(m))
                        for col, (k, v) in zip(cols, m.items()):
                            col.metric(k.upper(), f"{v:.4f}")
                    if autoresearch.get("simulated"):
                        st.caption("Simulated — install autoresearch for real AutoML")
                    if autoresearch.get("run_id"):
                        st.caption(f"Run ID: `{autoresearch['run_id']}` — see **🔍 Trace Log** tab for full iteration history")
                else:
                    st.error(f"Failed: {autoresearch.get('error', '?')}")

            if result.get("executor_result"):
                er = result["executor_result"]
                st.subheader("⚙️ Executor Result")
                st.success(
                    f"Optimised notebook **`{er.get('notebook_path')}`** generated"
                    + (f" → Ledger entry **#{er.get('entry_id')}**" if er.get("entry_id") else "")
                )

            st.divider()

            # ── Spec writer ───────────────────────────────────────────────────
            spec_md = _build_spec_md(_sel_file, _sel_goal, plan, validation, autoresearch or {})
            spec_filename = f"{Path(_sel_file).stem}.part2.agent.revisions.md"
            st.download_button(
                "📄 Download Spec (.md)",
                data=spec_md,
                file_name=spec_filename,
                mime="text/markdown",
                key="dl_spec",
            )

            st.subheader("🖊️ Review")
            review_state = st.session_state.get("plan_review")

            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("✅ Approve", key="approve_plan"):
                    st.session_state["plan_review"] = "approved"
            with c2:
                if st.button("🔄 Revise", key="revise_plan"):
                    st.session_state["plan_review"] = "revise"
            with c3:
                if st.button("❌ Reject", key="reject_plan"):
                    st.session_state["plan_review"] = "rejected"

            review_state = st.session_state.get("plan_review")
            if review_state == "approved":
                st.success("✅ Plan approved! Follow the next steps above to improve your model.")
            elif review_state == "revise":
                st.warning("🔄 Update your goal above and click **Generate Plan** again to get a revised plan.")
            elif review_state == "rejected":
                st.error("❌ Plan rejected. Update your goal or upload a different file, then regenerate.")

            with st.expander("🔎 Full JSON"):
                st.json(result)

# ── Tab 6: Swarm AutoML ────────────────────────────────────────────────────────

with tab_swarm:
    st.header("🐝 Swarm AutoML")
    st.caption(
        "Divide-and-conquer AutoML: the dataset is split into stratified chunks, "
        "one **SwarmWorkerAgent** per chunk runs FLAML in parallel, results are "
        "aggregated by model family, and the global champion is stored in the ledger."
    )

    if APP_MODE == "cloud":
        st.info(
            "**How to use (Cloud):**\n\n"
            "1. Upload your training CSV — it will be stored in Google Cloud Storage\n"
            "2. Optionally upload your EDA notebook — auto-detects target column & task type\n"
            "3. Tune the swarm settings, then click **🚀 Launch Swarm**\n\n"
            "Results stream into the **🔍 Trace Log** tab in real time.",
            icon="💡",
        )
    else:
        st.info(
            "**How to use (Local):**\n\n"
            "1. Enter the path to your training CSV\n"
            "2. Optionally point to an EDA notebook/script\n"
            "3. Tune the swarm settings, then click **🚀 Launch Swarm**\n\n"
            "Results stream into the **🔍 Trace Log** tab in real time.",
            icon="💡",
        )

    # ── CSV upload (cloud) / existing datasets picker ──────────────────────────
    _resolved_csv_path = st.session_state.get("swarm_csv_path", "")

    if APP_MODE == "cloud":
        st.subheader("📤 Dataset")
        up_col, pick_col = st.columns([2, 2])

        with up_col:
            csv_upload = st.file_uploader(
                "Upload training CSV",
                type=["csv"],
                key="swarm_csv_uploader",
                help="Uploaded to Google Cloud Storage — shared with all team members.",
            )
            if csv_upload and st.button("⬆️ Upload to GCS", key="btn_upload_csv"):
                with st.spinner(f"Uploading {csv_upload.name} to GCS…"):
                    up_result = api("POST", "/swarm/upload-csv",
                                    files={"file": (csv_upload.name,
                                                    csv_upload.getvalue(),
                                                    "text/csv")})
                if up_result and up_result.get("status") == "ok":
                    st.session_state["swarm_csv_path"] = up_result["csv_path"]
                    _resolved_csv_path = up_result["csv_path"]
                    st.success(
                        f"Uploaded ({up_result['size_mb']} MB) → "
                        f"`{up_result['csv_path']}`"
                    )
                else:
                    st.error("Upload failed.")

        with pick_col:
            datasets_resp = api("GET", "/swarm/datasets", silent=True)
            datasets = (datasets_resp or {}).get("datasets", [])
            if datasets:
                ds_options = {f"{d['name']} ({d['size_mb']} MB)": d["gcs_uri"]
                              for d in datasets}
                ds_options = {"— select existing —": ""} | ds_options
                picked = st.selectbox("Or pick existing dataset",
                                      list(ds_options.keys()),
                                      key="swarm_ds_pick")
                if ds_options[picked]:
                    _resolved_csv_path = ds_options[picked]
                    st.session_state["swarm_csv_path"] = _resolved_csv_path

        if _resolved_csv_path:
            st.caption(f"Active dataset: `{_resolved_csv_path}`")
        st.divider()

    # ── Config form ────────────────────────────────────────────────────────────
    with st.form("swarm_form"):
        s_col1, s_col2 = st.columns(2)

        with s_col1:
            if APP_MODE == "cloud":
                # Show the resolved path (read-only) so user sees what will run
                csv_path_input = st.text_input(
                    "📊 Dataset path / GCS URI",
                    value=_resolved_csv_path,
                    help="Auto-filled from the upload above. Edit only if needed.",
                )
            else:
                csv_path_input = st.text_input(
                    "📊 Training CSV path",
                    placeholder=r"C:\Users\...\train.csv",
                    help="Full local path to the CSV dataset file.",
                )
            eda_file_input = st.text_input(
                "🔬 EDA file (optional)",
                placeholder="notebooks/eda.ipynb",
                help="Any .ipynb / .py / .docx / .json file — "
                     "used to auto-detect target column and task type.",
            )
            target_col_input = st.text_input(
                "🎯 Target column",
                placeholder="auto-detect from EDA file",
                help="Leave blank to auto-detect from EDA file or column names.",
            )
            task_type_input = st.selectbox(
                "📐 Task type",
                ["auto-detect", "classification", "regression"],
                help="Auto-detect reads from EDA file or target column cardinality.",
            )

        with s_col2:
            time_budget = st.slider(
                "⏱ FLAML time budget per agent (seconds)",
                min_value=15, max_value=600, value=60, step=15,
                help="Each worker gets this many seconds of FLAML search. "
                     "Longer = better models, more wall-clock time.",
            )
            max_rows = st.number_input(
                "📏 Max rows per agent",
                min_value=1_000, max_value=500_000,
                value=150_000, step=10_000,
                help="n_agents = ceil(total_rows / this value). "
                     "Each agent trains on one stratified chunk.",
            )
            max_parallel = st.slider(
                "⚡ Max parallel agents",
                min_value=1, max_value=8, value=4,
                help="Capped by CPU count. More parallel = faster but more RAM.",
            )
            swarm_proj_label = st.selectbox(
                "📁 Assign to project",
                ["No project (unassigned)"] + [p["name"] for p in load_projects()],
                key="swarm_proj",
            )

        _swarm_author_col, _swarm_goal_col = st.columns([1, 2])
        with _swarm_author_col:
            _swarm_author = st.text_input(
                "Your GitHub Username ✱",
                value=st.session_state.get("gh_user", ""),
                placeholder="e.g. dehiska",
                key="swarm_author_field",
                help="Tags the swarm ledger entry with your name in the Leaderboard.",
            )
            if _swarm_author.strip():
                st.session_state["gh_user"] = _swarm_author.strip()
        with _swarm_goal_col:
            swarm_goal = st.text_input(
                "🎯 Goal",
                value="Maximize F1 on the test set",
                help="Stored in the trace log and ledger entry.",
            )
        launch_btn = st.form_submit_button("🚀 Launch Swarm", type="primary")

    # ── Result display helper ──────────────────────────────────────────────────
    def _display_swarm_result(result: dict):
        if result.get("status") != "success":
            st.error(f"Swarm failed: {result.get('error', '?')}")
            return

        best = result.get("best_model", {})
        metrics = best.get("metrics", {})
        val_score = metrics.get("val_score", 0.0)
        flaml_used = best.get("flaml", False)

        st.success(
            f"🏆 Champion: **{best.get('name')}** "
            f"({best.get('family', '?')})  |  "
            f"val_score = **{val_score:.4f}**  |  "
            f"{'🧪 FLAML' if flaml_used else '⚙️ sklearn fallback'}"
        )

        # Summary metrics row
        m_cols = st.columns(4)
        m_cols[0].metric("Agents launched", result.get("n_agents", "?"))
        m_cols[1].metric("Agents succeeded", result.get("n_succeeded", "?"))
        m_cols[2].metric("Rows trained on", f"{result.get('n_rows', 0):,}")
        m_cols[3].metric("Ledger entry", f"#{result.get('entry_id', '—')}")

        # Per-agent leaderboard
        worker_results = result.get("all_worker_results", [])
        if worker_results:
            import pandas as pd
            rows = []
            for r in worker_results:
                bm = r.get("best_model", {})
                rows.append({
                    "Agent":   r.get("agent_id", "?"),
                    "Model":   bm.get("name", "?"),
                    "Family":  bm.get("family", "?"),
                    "Score":   round(bm.get("metrics", {}).get("val_score", 0), 4),
                    "Rows":    f"{bm.get('n_rows_trained', 0):,}",
                    "FLAML":   "✅" if bm.get("flaml") else "⚙️",
                })
            df_lb = (
                pd.DataFrame(rows)
                .sort_values("Score", ascending=False)
                .reset_index(drop=True)
            )
            df_lb.insert(0, "Rank", range(1, len(df_lb) + 1))
            with st.expander("📊 Per-agent leaderboard", expanded=True):
                st.dataframe(df_lb, use_container_width=True, hide_index=True)

        # Champion hyperparameters
        params = best.get("params", {})
        if params:
            with st.expander("⚙️ Champion hyperparameters"):
                st.json(params)

        # Family tree visualisation
        worker_results = result.get("all_worker_results", [])
        if worker_results:
            try:
                import plotly.graph_objects as go
                import pandas as pd

                by_family: dict = {}
                for r in worker_results:
                    bm = r["best_model"]
                    fam = bm.get("family", "Unknown")
                    s   = bm["metrics"].get("val_score", 0)
                    if fam not in by_family or s > by_family[fam]["score"]:
                        by_family[fam] = {"name": bm["name"], "score": s}

                families = list(by_family.keys())
                scores   = [by_family[f]["score"] for f in families]
                names    = [by_family[f]["name"]  for f in families]
                champion_fam = best.get("family", families[0] if families else "")

                colors = [
                    "#FFD700" if f == champion_fam else "#4C9BE8"
                    for f in families
                ]
                fig = go.Figure(go.Bar(
                    x=families, y=scores,
                    text=[f"{n}<br>{s:.4f}" for n, s in zip(names, scores)],
                    textposition="outside",
                    marker_color=colors,
                    hovertemplate="%{x}<br>%{text}<extra></extra>",
                ))
                fig.update_layout(
                    title="Swarm: best score per model family (🥇 = champion)",
                    yaxis_title="Val Score",
                    height=340,
                    margin=dict(t=50, b=30),
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                pass  # plotly might not be available

        run_id = result.get("run_id")
        if run_id:
            st.caption(
                f"Run ID: `{run_id}` — full iteration trace in **🔍 Trace Log** tab"
            )

        # ── Evaluation panel ──────────────────────────────────────────────────
        eval_data = result.get("eval")
        if eval_data is None and APP_MODE == "cloud":
            # Fetch eval from backend if not embedded in result
            try:
                eval_data = api("POST", "/eval/swarm", json={**result, "run_deepeval": True})
            except Exception:
                eval_data = None

        if eval_data:
            _STATUS_ICON = {"PASS": "✅", "PARTIAL": "⚠️", "FAIL": "❌", "NOT_EVALUATED": "—"}
            overall = eval_data.get("overall_status", "?")
            overall_icon = _STATUS_ICON.get(overall, "?")

            with st.expander(f"🔬 Agent Evaluation — {overall_icon} {overall}", expanded=True):
                st.caption(
                    "**Tier 1** — custom Python metrics (instant, no API cost).  "
                    "**Tier 2** — DeepEval LLM-based metrics via Claude Haiku (requires ANTHROPIC_API_KEY on backend)."
                )
                custom = eval_data.get("custom_metrics", {})
                if custom:
                    st.subheader("Tier 1 — Custom Metrics")
                    c_rows = []
                    for mname, mval in custom.items():
                        if isinstance(mval, dict):
                            c_rows.append({
                                "Metric":  mname.replace("_", " ").title(),
                                "Status":  f"{_STATUS_ICON.get(mval.get('status','?'), '?')} {mval.get('status','?')}",
                                "Details": mval.get("note", ""),
                            })
                    if c_rows:
                        import pandas as _epd
                        st.dataframe(_epd.DataFrame(c_rows), use_container_width=True, hide_index=True)

                per_agent = eval_data.get("per_agent", {})
                if per_agent:
                    st.subheader("Per-Agent Breakdown")
                    pa_rows = []
                    for agent_name, agent_data in per_agent.items():
                        pa_rows.append({
                            "Agent":     agent_name,
                            "Status":    f"{_STATUS_ICON.get(agent_data.get('status','?'),'?')} {agent_data.get('status','?')}",
                            "Model":     agent_data.get("model") or agent_data.get("champion_model", "—"),
                            "Score":     agent_data.get("val_score") or agent_data.get("champion_score", "—"),
                            "FLAML":     "✅" if agent_data.get("flaml") else "—",
                        })
                    import pandas as _epd2
                    st.dataframe(_epd2.DataFrame(pa_rows), use_container_width=True, hide_index=True)

                de = eval_data.get("deepeval_metrics")
                if de and not de.get("error"):
                    st.subheader("Tier 2 — DeepEval (LLM-based)")
                    de_cols = st.columns(3)
                    for i, key in enumerate(("faithfulness", "answer_relevancy", "context_relevancy")):
                        val = de.get(key)
                        label = key.replace("_", " ").title()
                        icon = "✅" if (val is not None and val >= 0.5) else ("❌" if val is not None else "—")
                        de_cols[i].metric(label, f"{val:.4f}" if val is not None else "N/A", delta=icon)
                    reasons = de.get("reasons", {})
                    if reasons:
                        with st.expander("Reasons"):
                            st.json(reasons)
                elif de and de.get("error"):
                    st.info(f"DeepEval: {de['error']}")

    # ── Launch & render ────────────────────────────────────────────────────────
    if launch_btn:
        if not csv_path_input.strip():
            st.warning("Enter the path to your training CSV.")
        else:
            _target = target_col_input.strip() or None
            _task   = None if task_type_input == "auto-detect" else task_type_input
            _eda    = eda_file_input.strip() or None

            # Resolve project_id
            _swarm_proj_id = None
            if swarm_proj_label != "No project (unassigned)":
                for p in load_projects():
                    if p["name"] == swarm_proj_label:
                        _swarm_proj_id = p["id"]
                        break

            with st.spinner(
                f"🐝 Swarm running — this may take "
                f"{max_parallel * time_budget // 60 + 1}+ minutes…"
            ):
                if APP_MODE == "cloud":
                    result = api("POST", "/swarm/run", json={
                        "csv_path": csv_path_input.strip(),
                        "eda_file": _eda,
                        "target_col": _target,
                        "task_type": _task,
                        "time_budget_per_agent": time_budget,
                        "max_rows_per_agent": int(max_rows),
                        "max_parallel_agents": max_parallel,
                        "project_id": _swarm_proj_id,
                        "team_member": st.session_state.get("gh_user") or None,
                        "goal": swarm_goal,
                    })
                else:
                    from src.swarm_orchestrator import SwarmOrchestrator
                    orch = SwarmOrchestrator(db=get_db_direct())
                    result = orch.run_swarm(
                        csv_path=csv_path_input.strip(),
                        eda_file=_eda,
                        target_col=_target,
                        task_type=_task,
                        time_budget_per_agent=time_budget,
                        max_rows_per_agent=int(max_rows),
                        max_parallel_agents=max_parallel,
                        project_id=_swarm_proj_id,
                        team_member=st.session_state.get("gh_user") or None,
                        goal=swarm_goal,
                    )

            if result:
                st.session_state["swarm_result"] = result

    # Persist result across re-runs
    swarm_result = st.session_state.get("swarm_result")
    if swarm_result:
        st.divider()
        _display_swarm_result(swarm_result)
        with st.expander("🔎 Full JSON"):
            st.json(swarm_result)


# ── Tab 7: Costs ───────────────────────────────────────────────────────────────

with tab_costs:
    st.header("💰 Cost Dashboard")
    st.caption("Tracks LLM agent plan costs and autoresearch compute costs.")

    if APP_MODE == "cloud":
        data = api("GET", "/costs")
        costs = data.get("costs", []) if data else []
    else:
        costs = get_db_direct().get_costs()

    if not costs:
        st.info(
            "No costs logged yet.\n\n"
            "Costs are recorded automatically whenever you **Generate a Plan** (Agent Plan tab) "
            "or run **Autoresearch**. Go generate a plan and come back here!"
        )
    else:
        import pandas as pd
        import plotly.express as px

        df = pd.DataFrame(costs)
        # Ensure cost_type column exists (older rows may not have it)
        if "cost_type" not in df.columns:
            df["cost_type"] = "autoresearch"
        if "description" not in df.columns:
            df["description"] = ""

        total = df["cost"].sum()
        plan_total = df[df["cost_type"] == "agent_plan"]["cost"].sum()
        ar_total = df[df["cost_type"] == "autoresearch"]["cost"].sum()

        m1, m2, m3 = st.columns(3)
        m1.metric("💰 Total Cost", f"${total:.4f}")
        m2.metric("🤖 Agent Plans", f"${plan_total:.4f}")
        m3.metric("⚡ Autoresearch", f"${ar_total:.4f}")

        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(
                df, x="timestamp", y="cost",
                color="cost_type",
                color_discrete_map={"agent_plan": "#636EFA", "autoresearch": "#EF553B"},
                labels={"cost_type": "Type", "cost": "Cost ($)", "timestamp": "Time"},
                title="Cost Over Time",
            )
            fig.update_layout(height=320, margin=dict(t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            pie_df = df.groupby("cost_type")["cost"].sum().reset_index()
            pie_df["cost_type"] = pie_df["cost_type"].replace({
                "agent_plan": "Agent Plan", "autoresearch": "Autoresearch"
            })
            fig2 = px.pie(
                pie_df, names="cost_type", values="cost",
                title="By Cost Type",
                color_discrete_sequence=["#636EFA", "#EF553B"],
            )
            fig2.update_layout(height=320, margin=dict(t=40, b=20))
            st.plotly_chart(fig2, use_container_width=True)

        # Clean up display columns
        display_df = df[["id", "timestamp", "cost_type", "description", "gcp_instance", "time_hours", "cost"]].copy()
        display_df.columns = ["ID", "Timestamp", "Type", "Description", "Instance", "Hours", "Cost ($)"]
        display_df["Type"] = display_df["Type"].replace({"agent_plan": "🤖 Agent Plan", "autoresearch": "⚡ Autoresearch"})
        st.dataframe(display_df, use_container_width=True, hide_index=True)

# ── Tab 8: Trace Log ──────────────────────────────────────────────────────────

with tab_traces:
    st.header("🔍 Trace Log")
    st.caption("Live feed of every agent action from the autoresearch pipeline.")

    if APP_MODE == "cloud":
        data = api("GET", "/traces")
        all_traces = data.get("traces", []) if data else []
        run_ids = data.get("run_ids", []) if data else []
    else:
        all_traces = get_db_direct().get_traces(limit=500)
        run_ids = get_db_direct().get_trace_run_ids()

    hdr_col, btn_col = st.columns([4, 1])
    with btn_col:
        if st.button("🔄 Refresh", key="refresh_traces"):
            st.rerun()

    if not all_traces:
        st.info(
            "No traces yet.\n\n"
            "Go to the **🤖 Agent Plan** tab, toggle **⚡ Execute autoresearch** ON, "
            "then click **🚀 Generate Plan** to start the pipeline."
        )
    else:
        with hdr_col:
            run_options = ["All runs"] + run_ids
            selected_run = st.selectbox("Run", run_options, key="trace_run_select")

        filtered = (
            all_traces if selected_run == "All runs"
            else [t for t in all_traces if t.get("run_id") == selected_run]
        )
        # When viewing all runs, show newest first; for a specific run show oldest first
        if selected_run != "All runs":
            filtered = sorted(filtered, key=lambda x: x.get("timestamp", ""))

        AGENT_ICONS = {
            "autoresearcher": "🔬",
            "orchestrator": "🧠",
            "executor": "⚙️",
        }

        st.caption(f"{len(filtered)} trace entries")
        for t in filtered:
            icon = AGENT_ICONS.get(t.get("agent", ""), "•")
            ts = str(t.get("timestamp", ""))[:19]
            agent = t.get("agent", "?")
            msg = t.get("message", "")
            # Highlight "new best" entries
            if "new best" in msg:
                st.markdown(
                    f"`{ts}` &nbsp; {icon} **{agent}** &nbsp; 🟢 {msg}",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"`{ts}` &nbsp; {icon} **{agent}** &nbsp; {msg}",
                    unsafe_allow_html=True,
                )

        # Download button for generated notebooks
        executor_traces = [
            t for t in filtered
            if t.get("agent") == "executor" and isinstance(t.get("payload"), dict)
            and t["payload"].get("notebook")
        ]
        if executor_traces:
            st.divider()
            st.subheader("📥 Generated Notebooks")
            for t in executor_traces:
                nb_path = t["payload"]["notebook"]
                if Path(nb_path).exists():
                    with open(nb_path, "rb") as f:
                        st.download_button(
                            f"📥 Download `{nb_path}`",
                            data=f.read(),
                            file_name=nb_path,
                            mime="application/json",
                            key=f"dl_nb_{t.get('id', nb_path)}",
                        )
                else:
                    st.caption(f"`{nb_path}` — not available for download in cloud mode")


# ── Tab 9: Notebook Snippet ────────────────────────────────────────────────────

with tab_snippet:
    st.header("📋 Notebook Logging Cell")
    st.markdown(
        "Copy this cell into the **bottom of any notebook** after training. "
        "It saves a `*_ledger_entry.json` file next to the notebook. "
        "Then upload that JSON here — the ledger will ingest it directly with full metrics, "
        "model params, and preprocessing steps."
    )

    _SNIPPET = '''\
# ═══════════════════════════════════════════════════════════════
# AGENTIC DS LEDGER — Logging Cell
# Run this after training to save a ledger entry JSON.
# Then upload the JSON file in the DS Ledger dashboard.
# ═══════════════════════════════════════════════════════════════
import json, sys, subprocess
from datetime import datetime
from pathlib import Path

# ── 1. Model info ────────────────────────────────────────────────
model_info = {
    "name": "LightGBMClassifier",          # ← change to your model name
    "family": "Gradient Boosting",
    "params": params,                       # ← your params dict
    "best_iteration": last_model.best_iteration,
    "feature_importance": {
        "gain":  last_model.feature_importance(importance_type="gain").tolist(),
        "split": last_model.feature_importance(importance_type="split").tolist(),
        "features": feature_cols,
    },
}

# ── 2. Preprocessing steps ───────────────────────────────────────
preprocessing_steps = [
    {
        "name": "Drop invalid labels",
        "type": "Data Cleaning",
        "description": "Removed rows where cancel == -1",
        "code": "train = train[train[\\'cancel\\'] != -1].copy()",
    },
    {
        "name": "Categorical Encoding",
        "type": "Feature Engineering",
        "description": "Cast categorical columns to \\'category\\' dtype for LightGBM",
        "code": "for col in cat_cols: X[col] = X[col].astype(\\'category\\')",
    },
    {
        "name": "Class Weighting",
        "type": "Imbalance Handling",
        "description": "Applied sample weights to handle class imbalance",
        "code": "sample_weights[y == c] = w",
    },
    {
        "name": "StratifiedKFold",
        "type": "Cross Validation",
        "description": "5-fold stratified cross-validation",
        "code": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
    },
]

# ── 3. Metrics ───────────────────────────────────────────────────
oof_class = oof_preds.argmax(axis=1)
metrics = [
    {
        "name": "OOF Accuracy",
        "value": float(f"{accuracy_score(y, oof_class):.4f}"),
        "function": "accuracy_score",
    },
    {
        "name": "Mean CV Accuracy",
        "value": float(f"{np.mean(fold_accs):.4f}"),
        "std":   float(f"{np.std(fold_accs):.4f}"),
        "function": "np.mean(fold_accs)",
    },
    {
        "name": "Classification Report",
        "value": classification_report(
            y, oof_class,
            target_names=["Not Cancel (0)", "May Cancel (1)", "Cancel (2)"],
            output_dict=True,
        ),
        "function": "classification_report",
    },
]

# ── 4. Environment ───────────────────────────────────────────────
environment = {
    "python_version": sys.version,
    "pip_freeze": subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"]
    ).decode().splitlines(),
}

# ── 5. Assemble & save ───────────────────────────────────────────
notebook_name = Path(__file__).stem if "__file__" in dir() else "notebook"
ledger_entry = {
    "file_path":    f"{notebook_name}.ipynb",
    "timestamp":    datetime.now().isoformat(),
    "team_member":  "your_name_here",       # ← change this
    "models":       [model_info],
    "metrics":      metrics,
    "preprocessing": preprocessing_steps,
    "environment":  environment,
}

output_path = f"{notebook_name}_ledger_entry.json"
with open(output_path, "w") as f:
    json.dump(ledger_entry, f, indent=2, default=str)

print(f"✅ Ledger entry saved → {output_path}")
print(f"   Models: {len(ledger_entry[\\'models\\'])}")
print(f"   Metrics: {len(ledger_entry[\\'metrics\\'])}")
print(f"   Preprocessing: {len(ledger_entry[\\'preprocessing\\'])}")
'''

    st.code(_SNIPPET, language="python")

    st.divider()
    st.subheader("How to use it")
    st.markdown("""
1. **Copy** the cell above into the bottom of your notebook (after training completes)
2. **Edit** the 3 marked lines: model name, `params` variable, `team_member`
3. **Run** the cell — it saves `<notebook_name>_ledger_entry.json` next to your notebook
4. **Upload** that `.json` file in the **Upload Files** tab
5. The ledger will show full model params, OOF accuracy, per-class metrics, and all preprocessing steps
""")

    st.info(
        "💡 If the JSON file is saved **next to** the `.ipynb` with the name "
        "`<notebook>_ledger_entry.json`, uploading the notebook itself will also "
        "auto-detect and use the JSON instead of AST-parsing.",
        icon="💡",
    )
