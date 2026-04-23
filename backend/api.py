"""
FastAPI Backend — REST API for the Agentic DS Ledger.

Run with:
    uvicorn backend.api:app --reload --port 8080

Endpoints:
  POST /projects                     — Create a new project
  GET  /projects                     — List all projects
  GET  /projects/{id}                — Get single project
  DELETE /projects/{id}              — Delete a project
  POST /files/parse                  — Upload + parse .ipynb / .py / .docx
  POST /files/parse-local            — Parse by local path
  GET  /ledger                       — List ledger entries (filter by project_id)
  GET  /ledger/{entry_id}            — Get single entry
  POST /plan/generate                — Generate a multi-agent plan
  POST /autoresearch/run             — Run (or simulate) autoresearch
  GET  /costs                        — List cost logs
  GET  /health                       — Health check
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, File, HTTPException, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from dotenv import load_dotenv
load_dotenv()

from src.database import get_db
from src.inference_engine import InferenceEngine
from src.mcp_server import MCPServer, generate_experiment_hash
from src.multi_agent_orchestrator import MultiAgentOrchestrator
from src.rag_system import RAGSystem

# ── App setup ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Agentic DS Ledger API",
    description="Backend for the Agentic DS Ledger.",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPPORTED_TYPES = {".ipynb", ".py", ".docx", ".doc", ".json"}

# ── Singletons ─────────────────────────────────────────────────────────────────

_db = None
_rag = None
_orchestrator = None
_mcp = None


def get_services():
    global _db, _rag, _orchestrator, _mcp
    if _db is None:
        _db = get_db()
    if _rag is None:
        try:
            _rag = RAGSystem()
        except Exception:
            _rag = None
    if _orchestrator is None:
        _orchestrator = MultiAgentOrchestrator(
            db=_db, rag=_rag,
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            github_token=os.getenv("GITHUB_TOKEN"),
        )
    if _mcp is None:
        _mcp = MCPServer(db=_db)
    return _db, _rag, _orchestrator, _mcp


# ── Request models ─────────────────────────────────────────────────────────────

class CreateProjectRequest(BaseModel):
    name: str
    description: str = ""


class ParseLocalRequest(BaseModel):
    file_path: str
    project_id: Optional[int] = None
    repo_name: Optional[str] = None
    branch: str = "main"
    team_member: Optional[str] = None


class PlanRequest(BaseModel):
    file_path: str
    goal: str = "Improve model performance"
    project_id: Optional[int] = None
    repo_name: Optional[str] = None
    branch: str = "main"
    team_member: Optional[str] = None
    execute_autoresearch: bool = False


class AutoresearchRequest(BaseModel):
    file_path: str
    goal: str = "Improve model accuracy"
    project_id: Optional[int] = None
    repo_name: Optional[str] = None
    branch: str = "main"
    team_member: Optional[str] = None


class InferRequest(BaseModel):
    metadata: dict


class ConfirmRequest(BaseModel):
    user_corrected: dict


# ── Health ─────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "mode": os.getenv("APP_MODE", "local")}


# ── Projects ───────────────────────────────────────────────────────────────────

@app.post("/projects", status_code=201)
def create_project(req: CreateProjectRequest):
    db, _, _, _ = get_services()
    try:
        project_id = db.create_project(req.name, req.description)
    except Exception as e:
        err = str(e)
        if any(k in err for k in ("UNIQUE", "unique", "duplicate key", "23505", "already exists")):
            raise HTTPException(409, f"Project '{req.name}' already exists.")
        raise HTTPException(400, err)
    return {"id": project_id, "name": req.name, "description": req.description}


@app.get("/projects")
def list_projects():
    db, _, _, _ = get_services()
    return {"projects": db.get_projects()}


@app.get("/projects/{project_id}")
def get_project(project_id: int):
    db, _, _, _ = get_services()
    project = db.get_project(project_id)
    if not project:
        raise HTTPException(404, f"Project {project_id} not found.")
    entries = db.get_ledger_entries(project_id=project_id)
    return {**project, "files": entries}


@app.delete("/projects/{project_id}")
def delete_project(project_id: int):
    db, _, _, _ = get_services()
    if not db.get_project(project_id):
        raise HTTPException(404, f"Project {project_id} not found.")
    db.delete_project(project_id)
    return {"status": "deleted", "id": project_id}


# ── File parsing ───────────────────────────────────────────────────────────────

@app.post("/files/parse")
async def parse_file_upload(
    file: UploadFile = File(...),
    project_id: Optional[int] = Query(None),
    team_member: Optional[str] = Query(None),
    github_repo: Optional[str] = Query(None),
    github_branch: Optional[str] = Query("main"),
):
    """Upload .ipynb / .py / .docx → parse → store in ledger under a project."""
    db, _, _, mcp = get_services()

    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_TYPES:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Allowed: {', '.join(SUPPORTED_TYPES)}")

    content = await file.read()
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False, mode="wb") as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        metadata = mcp.parse_file(tmp_path)
        metadata["file_path"] = file.filename
        # Check for duplicate experiment before inserting
        exp_hash = generate_experiment_hash(metadata)
        duplicate = db.find_by_hash(exp_hash) if exp_hash else None
        entry_id = mcp.store_in_db(
            metadata, project_id=project_id,
            github_repo=github_repo or "",
            github_branch=github_branch or "main",
            team_member=team_member,
        )
        # Run inference layer and attach to entry
        infer_result = InferenceEngine().infer(metadata)
        db.update_ledger_entry(entry_id, {
            "auto_extracted": infer_result,
            "confidence_score": infer_result["overall_confidence"],
        })
        return {
            "status": "ok",
            "file": file.filename,
            "file_type": ext.lstrip("."),
            "project_id": project_id,
            "entry_id": entry_id,
            "models_found": len(metadata.get("models", [])),
            "metrics_found": len(metadata.get("metrics", [])),
            "preprocessing_found": len(metadata.get("preprocessing", [])),
            "confidence": infer_result["overall_confidence"],
            "experiment_hash": exp_hash,
            "duplicate_of": duplicate,
            "word_count": metadata.get("word_count"),
            "doc_keywords": metadata.get("doc_keywords"),
            "metadata": metadata,
            "experiment": infer_result,
        }
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/files/parse-local")
def parse_file_local(req: ParseLocalRequest):
    db, _, _, mcp = get_services()
    if not Path(req.file_path).exists():
        raise HTTPException(404, f"File not found: {req.file_path}")
    metadata = mcp.parse_file(req.file_path, req.repo_name, req.branch, req.team_member)
    entry_id = mcp.store_in_db(metadata, project_id=req.project_id)
    infer_result = InferenceEngine().infer(metadata)
    db.update_ledger_entry(entry_id, {
        "auto_extracted": infer_result,
        "confidence_score": infer_result["overall_confidence"],
    })
    return {
        "status": "ok",
        "entry_id": entry_id,
        "models_found": len(metadata.get("models", [])),
        "confidence": infer_result["overall_confidence"],
        "metadata": metadata,
        "experiment": infer_result,
    }


# Keep old endpoint working
@app.post("/notebooks/parse")
async def parse_notebook_upload(file: UploadFile = File(...), team_member: Optional[str] = None):
    return await parse_file_upload(file=file, project_id=None, team_member=team_member)

@app.post("/notebooks/parse-local")
def parse_notebook_local(req: ParseLocalRequest):
    return parse_file_local(req)


# ── Inference & Confirmation ───────────────────────────────────────────────────

@app.post("/infer")
def infer_experiment(req: InferRequest):
    """Stateless: run inference engine on parse metadata, return structured experiment."""
    result = InferenceEngine().infer(req.metadata)
    return {"status": "ok", "experiment": result}


@app.post("/confirm/{entry_id}")
def confirm_entry(entry_id: int, req: ConfirmRequest):
    """Store user-corrected experiment and mark entry as approved."""
    db, _, _, _ = get_services()
    for e in db.get_ledger_entries():
        if e.get("id") == entry_id:
            db.confirm_entry(entry_id, req.user_corrected)
            return {"status": "confirmed", "entry_id": entry_id}
    raise HTTPException(404, f"Entry {entry_id} not found.")


@app.get("/experiments")
def get_experiments(
    project_id: Optional[int] = None,
    approved_only: bool = True,
):
    """Return ledger entries with confirmed experiment data."""
    db, _, _, _ = get_services()
    exps = db.get_experiments(project_id=project_id, approved_only=approved_only)
    return {"experiments": exps, "count": len(exps)}


# ── Ledger ─────────────────────────────────────────────────────────────────────

@app.get("/ledger")
def list_ledger(
    file_path: Optional[str] = None,
    project_id: Optional[int] = None,
    github_repo: Optional[str] = None,
):
    db, _, _, _ = get_services()
    try:
        entries = db.get_ledger_entries(file_path=file_path, project_id=project_id)
    except TypeError:
        entries = db.get_ledger_entries(file_path=file_path)
    return {"entries": entries, "count": len(entries)}


@app.get("/ledger/{entry_id}")
def get_ledger_entry(entry_id: int):
    db, _, _, _ = get_services()
    for e in db.get_ledger_entries():
        if e.get("id") == entry_id:
            return e
    raise HTTPException(404, f"Entry {entry_id} not found.")


@app.delete("/ledger/{entry_id}")
def delete_ledger_entry(entry_id: int):
    db, _, _, _ = get_services()
    db.delete_ledger_entry(entry_id)
    return {"status": "deleted", "entry_id": entry_id}


# ── Planning ───────────────────────────────────────────────────────────────────

@app.post("/plan/generate")
def generate_plan(req: PlanRequest):
    db, _, orchestrator, _ = get_services()
    result = orchestrator.run_pipeline(
        file_path=req.file_path, goal=req.goal,
        repo_name=req.repo_name, branch=req.branch,
        team_member=req.team_member, execute_autoresearch=req.execute_autoresearch,
    )
    # Log agent-plan LLM cost from the resource estimate
    try:
        cost_est = result.get("plan", {}).get("resource_estimate", {})
        total = float(cost_est.get("total_cost") or 0)
        if total > 0:
            db.log_cost(
                plan_id=None,
                gcp_instance=cost_est.get("gcp_instance", "claude-3-5-haiku"),
                time_hours=float(cost_est.get("time_hours") or 0),
                cost=total,
                cost_type="agent_plan",
                description=(req.goal or "")[:120],
            )
    except Exception:
        pass
    return result


@app.post("/autoresearch/run")
def run_autoresearch(req: AutoresearchRequest):
    from src.autoresearch_wrapper import AutoResearchWrapper
    db, _, _, _ = get_services()
    wrapper = AutoResearchWrapper(db=db, github_token=os.getenv("GITHUB_TOKEN"))
    return wrapper.run_autoresearch(
        repo_name=req.repo_name or "", file_path=req.file_path,
        branch=req.branch, goal=req.goal, team_member=req.team_member, execute=True,
    )


# ── Costs ──────────────────────────────────────────────────────────────────────

@app.get("/costs")
def get_costs():
    db, _, _, _ = get_services()
    return {"costs": db.get_costs()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.api:app", host="0.0.0.0", port=8080, reload=True)
