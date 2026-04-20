"""
ledger.py — High-level Ledger facade.

Works in Google Colab, Jupyter, VS Code notebooks, and plain scripts.

Usage:
    from agentic_ds_ledger import Ledger

    ledger = Ledger("MyProject")
    summary = ledger.parse("model.ipynb")
    ledger.show()

    # Google Colab with Drive persistence:
    ledger = Ledger("MyProject", db_path="/content/drive/MyDrive/myproject.db")

    # Agent plan (requires OPENAI_API_KEY):
    result = ledger.run_agent_plan("Improve F1 score")
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional


class Ledger:
    """
    Single entry point for parsing and tracking DS experiments.

    - `project_name` is created if it doesn't exist (idempotent).
    - `db_path` defaults to "ledger.db" in CWD. Pass an absolute path
      when persisting to Google Drive.
    """

    def __init__(
        self,
        project_name: str,
        db_path: str = "ledger.db",
        openai_api_key: Optional[str] = None,
    ):
        from .database import LocalDatabase
        from .mcp_server import MCPServer

        self.project_name = project_name
        self._db = LocalDatabase(db_path)
        self._mcp = MCPServer(db=self._db)
        self._openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self._project_id = self._get_or_create_project(project_name)

    # ── Core API ───────────────────────────────────────────────────────────────

    def parse(self, file_path: str) -> Dict:
        """
        Parse a .ipynb, .py, or .docx file and store the ledger entry.

        Returns a summary dict with entry_id, models/metrics/preprocessing
        found counts, and the full lists.
        """
        resolved = str(Path(file_path).resolve())
        metadata = self._mcp.parse_file(resolved)
        metadata["file_path"] = file_path  # keep display name clean
        entry_id = self._mcp.store_in_db(metadata, project_id=self._project_id)

        summary = {
            "entry_id": entry_id,
            "file": file_path,
            "models": metadata.get("models", []),
            "metrics": metadata.get("metrics", []),
            "preprocessing": metadata.get("preprocessing", []),
            "models_found": len(metadata.get("models", [])),
            "metrics_found": len(metadata.get("metrics", [])),
            "preprocessing_found": len(metadata.get("preprocessing", [])),
        }

        # Render parse summary inline if in IPython
        try:
            from .display import render_parse_result
            render_parse_result(summary)
        except Exception:
            pass

        return summary

    def show(self, n: int = 20) -> None:
        """Render the project ledger as an HTML table (Colab/Jupyter) or plain text."""
        from .display import render_ledger_table
        render_ledger_table(self.entries(n=n), project_name=self.project_name)

    def entries(self, n: int = 20) -> List[Dict]:
        """Return the n most recent ledger entries for this project."""
        return self._db.get_ledger_entries(project_id=self._project_id)[:n]

    def run_agent_plan(
        self,
        goal: str = "Improve model performance",
        file_path: Optional[str] = None,
    ) -> Dict:
        """
        Run the multi-agent planning pipeline against this project.

        Requires OPENAI_API_KEY (set via env or Ledger(..., openai_api_key=...)).
        `file_path` defaults to the most recently parsed file.
        """
        # Fix internal src.X imports when running from installed package
        _root = str(Path(__file__).parent.parent)
        import sys
        if _root not in sys.path:
            sys.path.insert(0, _root)

        from .multi_agent_orchestrator import MultiAgentOrchestrator

        if file_path is None:
            recent = self.entries(n=1)
            if not recent:
                raise ValueError("No files parsed yet. Call ledger.parse() first.")
            file_path = recent[0]["file_path"]

        orch = MultiAgentOrchestrator(
            db=self._db,
            openai_api_key=self._openai_api_key,
        )
        return orch.run_pipeline(file_path=file_path, goal=goal)

    # ── Project management ─────────────────────────────────────────────────────

    def delete(self) -> None:
        """Delete this project and all its ledger entries."""
        self._db.delete_project(self._project_id)

    @property
    def project_id(self) -> int:
        return self._project_id

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _get_or_create_project(self, name: str) -> int:
        for p in self._db.get_projects():
            if p["name"] == name:
                return p["id"]
        try:
            return self._db.create_project(name)
        except sqlite3.IntegrityError:
            # Race condition (rare): re-fetch
            for p in self._db.get_projects():
                if p["name"] == name:
                    return p["id"]
            raise

    def __repr__(self) -> str:
        count = len(self._db.get_ledger_entries(project_id=self._project_id))
        return f"<Ledger project={self.project_name!r} entries={count}>"
