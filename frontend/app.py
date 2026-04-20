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
SUPPORTED_TYPES = [".ipynb", ".py", ".docx"]

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
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        github_token=os.getenv("GITHUB_TOKEN"),
    )
    return db, mcp, orchestrator


def api(method: str, path: str, **kwargs):
    import httpx
    try:
        r = httpx.request(method, f"{BACKEND_URL}{path}", timeout=60.0, **kwargs)
        r.raise_for_status()
        return r.json()
    except Exception as e:
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
        return api("POST", "/projects", json={"name": name, "description": description})
    import sqlite3
    db = get_db_direct()
    try:
        pid = db.create_project(name, description)
        return {"id": pid, "name": name}
    except sqlite3.IntegrityError:
        return None  # duplicate name — caller shows warning


def delete_project_action(project_id: int):
    if APP_MODE == "cloud":
        api("DELETE", f"/projects/{project_id}")
    else:
        get_db_direct().delete_project(project_id)


def parse_uploaded_file(uploaded, project_id):
    ext = Path(uploaded.name).suffix.lower()
    content = uploaded.getvalue()

    if APP_MODE == "cloud":
        return api("POST", "/files/parse",
                   files={"file": (uploaded.name, content, "application/octet-stream")},
                   params={"project_id": project_id} if project_id else {})

    # Phase 1: direct
    mcp = get_mcp_direct()
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False, mode="wb") as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    try:
        meta = mcp.parse_file(tmp_path)
        meta["file_path"] = uploaded.name
        entry_id = mcp.store_in_db(meta, project_id=project_id)
        return {
            "status": "ok", "entry_id": entry_id,
            "file_type": ext.lstrip("."),
            "models_found": len(meta.get("models", [])),
            "metrics_found": len(meta.get("metrics", [])),
            "preprocessing_found": len(meta.get("preprocessing", [])),
            "word_count": meta.get("word_count"),
            "doc_keywords": meta.get("doc_keywords"),
            "metadata": meta,
        }
    finally:
        Path(tmp_path).unlink(missing_ok=True)


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
        new_desc = st.text_area("Description (optional)", key="new_proj_desc", height=60)
        if st.button("Create Project", type="primary", key="btn_create_proj"):
            if not new_name.strip():
                st.warning("Enter a project name.")
            else:
                result = create_project_action(new_name.strip(), new_desc.strip())
                if result:
                    st.success(f"Created: **{new_name}**")
                    st.rerun()
                else:
                    st.warning(f'A project named "{new_name}" already exists.')

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

    # GitHub / Phase 2 settings
    with st.expander("GitHub (Phase 2)", expanded=False):
        repo_name = st.text_input("GitHub Repo", placeholder="user/repo", key="gh_repo")
        branch = st.text_input("Branch", value="main", key="gh_branch")
        team_member = st.text_input("Your GitHub Username", key="gh_user")

    st.divider()
    goal = st.text_area("🎯 Goal", value="Improve F1 score and add uncertainty quantification", height=70, key="goal_input")
    execute_ar = st.toggle("⚡ Execute autoresearch", value=False, key="exec_ar")

# ── Tabs ───────────────────────────────────────────────────────────────────────

tab_upload, tab_ledger, tab_plan, tab_costs = st.tabs([
    "📂 Upload Files",
    "📜 Ledger",
    "🤖 Agent Plan",
    "💰 Costs",
])

# ── Tab 1: Upload ──────────────────────────────────────────────────────────────

with tab_upload:
    st.header("Upload Files")

    # Project picker for upload
    upload_proj_options = {p["name"]: p["id"] for p in projects}
    upload_proj_options = {"No project (unassigned)": None, **upload_proj_options}
    upload_proj_label = st.selectbox("Assign to project", list(upload_proj_options.keys()), key="upload_proj")
    upload_proj_id = upload_proj_options[upload_proj_label]

    st.caption("Supported: `.ipynb` notebooks · `.py` scripts · `.docx` documents")
    uploaded_files = st.file_uploader(
        "Upload files",
        type=["ipynb", "py", "docx", "doc"],
        accept_multiple_files=True,
        key="file_uploader",
    )

    if uploaded_files and st.button("📋 Parse All Files", type="primary"):
        for uploaded in uploaded_files:
            with st.spinner(f"Parsing {uploaded.name}..."):
                result = parse_uploaded_file(uploaded, upload_proj_id)

            if result and result.get("status") == "ok":
                ftype = result.get("file_type", "?")
                icon = {"ipynb": "📓", "py": "🐍", "docx": "📄"}.get(ftype, "📁")
                st.success(f"{icon} **{uploaded.name}** — entry #{result.get('entry_id')}")

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
            else:
                st.error(f"Failed to parse {uploaded.name}")

# ── Tab 2: Ledger ──────────────────────────────────────────────────────────────

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
                            st.write(f"- `{met.get('function', met.get('name', '?'))}`"
                                     + (f" — line {met['line_number']}" if met.get("line_number") else ""))

                    if preprocessing:
                        st.subheader("⚙️ Preprocessing")
                        for p in preprocessing:
                            st.write(f"- **{p.get('name', '?')}** ({p.get('type', '?')})")

                    # For docx: show raw text preview
                    if ftype == "docx" and entry.get("raw_text"):
                        st.subheader("📄 Document Preview")
                        st.text(entry["raw_text"][:800] + ("..." if len(entry.get("raw_text","")) > 800 else ""))

# ── Tab 3: Agent Plan ──────────────────────────────────────────────────────────

with tab_plan:
    st.header("🤖 Multi-Agent Planning")

    all_entries = load_ledger(project_id=selected_project_id)
    file_options = [e.get("file_path", "?") for e in all_entries]

    if not file_options:
        st.warning("No files in ledger. Upload files first.")
    else:
        selected_file = st.selectbox("Select file for planning", file_options, key="plan_file_select")
        st.info(f"Goal: **{goal}** | autoresearch: **{'ON' if execute_ar else 'OFF'}**")

        if st.button("🚀 Generate Plan", type="primary", key="btn_generate_plan"):
            with st.spinner("Running multi-agent pipeline..."):
                if APP_MODE == "cloud":
                    result = api("POST", "/plan/generate", json={
                        "file_path": selected_file, "goal": goal,
                        "project_id": selected_project_id,
                        "repo_name": repo_name or None, "branch": branch,
                        "team_member": team_member or None,
                        "execute_autoresearch": execute_ar,
                    })
                else:
                    orch = get_orch_direct()
                    result = orch.run_pipeline(
                        file_path=selected_file, goal=goal,
                        execute_autoresearch=execute_ar,
                    )

            if result:
                plan = result.get("plan", {})
                validation = result.get("validation", {})
                autoresearch = result.get("autoresearch_result")

                verdict = validation.get("verdict", "UNKNOWN")
                score = validation.get("score", 0)
                if "REJECTED" in verdict:
                    st.error(f"❌ {verdict} — Score: {score}/100")
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
                    st.subheader("⚡ autoresearch Result")
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
                    else:
                        st.error(f"Failed: {autoresearch.get('error', '?')}")

                st.divider()
                st.subheader("🖊️ Review")
                c1, c2, c3 = st.columns(3)
                with c1:
                    if st.button("✅ Approve", key="approve_plan"):
                        st.success("Plan approved!")
                with c2:
                    if st.button("🔄 Revise", key="revise_plan"):
                        st.warning("Update your goal and regenerate.")
                with c3:
                    if st.button("❌ Reject", key="reject_plan"):
                        st.error("Plan rejected.")

                with st.expander("🔎 Full JSON"):
                    st.json(result)

# ── Tab 4: Costs ───────────────────────────────────────────────────────────────

with tab_costs:
    st.header("💰 Cost Dashboard")

    if APP_MODE == "cloud":
        data = api("GET", "/costs")
        costs = data.get("costs", []) if data else []
    else:
        costs = get_db_direct().get_costs()

    if not costs:
        st.info("No costs logged yet. Costs are logged when autoresearch runs.")
    else:
        import pandas as pd
        import plotly.express as px

        df = pd.DataFrame(costs)
        st.metric("Total Estimated Cost", f"${df['cost'].sum():.3f}")

        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(df, x="timestamp", y="cost", color="gcp_instance", title="Cost Over Time")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = px.pie(
                df.groupby("gcp_instance")["cost"].sum().reset_index(),
                names="gcp_instance", values="cost", title="By Instance Type",
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.dataframe(df, use_container_width=True)
