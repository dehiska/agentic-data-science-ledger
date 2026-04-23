"""
Database layer — supports both Phase 1 (SQLite) and Phase 2 (Supabase).

Schema:
  projects  — one row per project (name, description)
  ledger    — one row per file uploaded, FK to projects
  costs     — one row per autoresearch run cost log
"""

import json
import os
import sqlite3
from typing import Dict, List, Optional


# ── Phase 1: Local SQLite ──────────────────────────────────────────────────────

class LocalDatabase:
    def __init__(self, db_path: str = "ledger.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_tables()

    def _create_tables(self):
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS projects (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT NOT NULL UNIQUE,
            description TEXT DEFAULT '',
            created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS ledger (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id    INTEGER REFERENCES projects(id) ON DELETE SET NULL,
            file_path     TEXT NOT NULL,
            file_type     TEXT DEFAULT 'ipynb',
            environment   TEXT,
            models        TEXT,
            metrics       TEXT,
            preprocessing TEXT,
            raw_text      TEXT,
            status        TEXT DEFAULT 'Pending',
            timestamp     DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS costs (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            plan_id      INTEGER REFERENCES ledger(id),
            gcp_instance TEXT,
            time_hours   REAL,
            cost         REAL,
            timestamp    DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS traces (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id       TEXT NOT NULL,
            agent        TEXT NOT NULL,
            message      TEXT NOT NULL,
            payload      TEXT DEFAULT NULL,
            entry_id     INTEGER REFERENCES ledger(id) ON DELETE SET NULL,
            timestamp    DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        """)
        self.conn.commit()
        self._migrate()

    def _migrate(self):
        """Add columns that were introduced after initial schema creation."""
        existing = {row[1] for row in self.conn.execute("PRAGMA table_info(ledger)")}
        migrations = [
            ("project_id",      "INTEGER REFERENCES projects(id) ON DELETE SET NULL"),
            ("file_type",       "TEXT DEFAULT 'ipynb'"),
            ("raw_text",        "TEXT DEFAULT ''"),
            ("status",          "TEXT DEFAULT 'Pending'"),
            # v2: inference + confirmation
            ("auto_extracted",  "TEXT DEFAULT NULL"),
            ("user_corrected",  "TEXT DEFAULT NULL"),
            ("confidence_score","REAL DEFAULT NULL"),
            ("approved",        "INTEGER DEFAULT 0"),
            # v2.1: deduplication
            ("experiment_hash", "TEXT DEFAULT NULL"),
        ]
        for col, definition in migrations:
            if col not in existing:
                self.conn.execute(f"ALTER TABLE ledger ADD COLUMN {col} {definition}")

        # Costs table migrations
        costs_existing = {row[1] for row in self.conn.execute("PRAGMA table_info(costs)")}
        costs_migrations = [
            ("cost_type",    "TEXT DEFAULT 'autoresearch'"),
            ("description",  "TEXT DEFAULT ''"),
        ]
        for col, definition in costs_migrations:
            if col not in costs_existing:
                self.conn.execute(f"ALTER TABLE costs ADD COLUMN {col} {definition}")

        self.conn.commit()

    # ── Projects ───────────────────────────────────────────────────────────────

    def create_project(self, name: str, description: str = "") -> int:
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO projects (name, description) VALUES (?, ?)",
            (name, description),
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_projects(self) -> List[Dict]:
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT p.*, COUNT(l.id) as file_count
            FROM projects p
            LEFT JOIN ledger l ON l.project_id = p.id
            GROUP BY p.id
            ORDER BY p.created_at DESC
        """)
        columns = [col[0] for col in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def get_project(self, project_id: int) -> Optional[Dict]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
        columns = [col[0] for col in cursor.description]
        row = cursor.fetchone()
        return dict(zip(columns, row)) if row else None

    def delete_project(self, project_id: int):
        self.conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
        self.conn.commit()

    # ── Ledger ─────────────────────────────────────────────────────────────────

    def insert_ledger_entry(
        self,
        metadata: Dict,
        project_id: Optional[int] = None,
        auto_extracted: Optional[Dict] = None,
        confidence_score: Optional[float] = None,
        experiment_hash: Optional[str] = None,
        **kwargs,
    ) -> int:
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO ledger
              (project_id, file_path, file_type, environment, models, metrics,
               preprocessing, raw_text, auto_extracted, confidence_score, experiment_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                project_id,
                metadata["file_path"],
                metadata.get("file_type", "ipynb"),
                json.dumps(metadata.get("environment", {})),
                json.dumps(metadata.get("models", [])),
                json.dumps(metadata.get("metrics", [])),
                json.dumps(metadata.get("preprocessing", [])),
                metadata.get("raw_text", ""),
                json.dumps(auto_extracted) if auto_extracted is not None else None,
                confidence_score,
                experiment_hash,
            ),
        )
        self.conn.commit()
        return cursor.lastrowid

    def find_by_hash(self, experiment_hash: str) -> Optional[Dict]:
        """Return the first entry with a matching experiment_hash, or None."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT id, file_path, timestamp FROM ledger WHERE experiment_hash = ? LIMIT 1",
            (experiment_hash,),
        )
        row = cursor.fetchone()
        if row:
            return {"id": row[0], "file_path": row[1], "timestamp": row[2]}
        return None

    def get_ledger_entries(
        self,
        file_path: Optional[str] = None,
        project_id: Optional[int] = None,
    ) -> List[Dict]:
        cursor = self.conn.cursor()
        where, params = [], []
        if file_path:
            where.append("l.file_path = ?")
            params.append(file_path)
        if project_id is not None:
            where.append("l.project_id = ?")
            params.append(project_id)
        clause = ("WHERE " + " AND ".join(where)) if where else ""
        cursor.execute(f"""
            SELECT l.*, p.name as project_name
            FROM ledger l
            LEFT JOIN projects p ON p.id = l.project_id
            {clause}
            ORDER BY l.timestamp DESC
        """, params)
        columns = [col[0] for col in cursor.description]
        rows = []
        for row in cursor.fetchall():
            entry = dict(zip(columns, row))
            for field in ("environment", "models", "metrics", "preprocessing"):
                try:
                    entry[field] = json.loads(entry[field] or "[]")
                except Exception:
                    entry[field] = []
            rows.append(entry)
        return rows

    def update_ledger_entry(self, entry_id: int, updates: Dict):
        set_parts, values = [], []
        for k, v in updates.items():
            set_parts.append(f"{k} = ?")
            values.append(json.dumps(v) if isinstance(v, (dict, list)) else v)
        values.append(entry_id)
        self.conn.execute(
            f"UPDATE ledger SET {', '.join(set_parts)} WHERE id = ?", values
        )
        self.conn.commit()

    def delete_ledger_entry(self, entry_id: int) -> None:
        self.conn.execute("DELETE FROM ledger WHERE id = ?", (entry_id,))
        self.conn.commit()

    def confirm_entry(self, entry_id: int, user_corrected: Dict) -> None:
        """Store user-edited experiment and mark entry as approved."""
        self.conn.execute(
            "UPDATE ledger SET user_corrected=?, approved=1, status='Confirmed' WHERE id=?",
            (json.dumps(user_corrected), entry_id),
        )
        self.conn.commit()

    def get_experiments(
        self,
        project_id: Optional[int] = None,
        approved_only: bool = True,
    ) -> List[Dict]:
        """Return ledger entries with auto_extracted/user_corrected parsed."""
        cursor = self.conn.cursor()
        where, params = [], []
        if approved_only:
            where.append("l.approved = 1")
        if project_id is not None:
            where.append("l.project_id = ?")
            params.append(project_id)
        clause = ("WHERE " + " AND ".join(where)) if where else ""
        cursor.execute(f"""
            SELECT l.*, p.name as project_name
            FROM ledger l
            LEFT JOIN projects p ON p.id = l.project_id
            {clause}
            ORDER BY l.timestamp DESC
        """, params)
        columns = [col[0] for col in cursor.description]
        rows = []
        for row in cursor.fetchall():
            entry = dict(zip(columns, row))
            for field in ("environment", "models", "metrics", "preprocessing"):
                try:
                    entry[field] = json.loads(entry[field] or "[]")
                except Exception:
                    entry[field] = []
            for field in ("auto_extracted", "user_corrected"):
                try:
                    entry[field] = json.loads(entry[field]) if entry[field] else None
                except Exception:
                    entry[field] = None
            rows.append(entry)
        return rows

    # ── Costs ──────────────────────────────────────────────────────────────────

    def log_cost(
        self,
        plan_id: Optional[int],
        gcp_instance: str,
        time_hours: float,
        cost: float,
        cost_type: str = "autoresearch",
        description: str = "",
    ):
        self.conn.execute(
            "INSERT INTO costs (plan_id, gcp_instance, time_hours, cost, cost_type, description) VALUES (?, ?, ?, ?, ?, ?)",
            (plan_id, gcp_instance, time_hours, cost, cost_type, description),
        )
        self.conn.commit()

    def get_costs(self) -> List[Dict]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM costs ORDER BY timestamp DESC")
        columns = [col[0] for col in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    # ── Traces ─────────────────────────────────────────────────────────────────

    def log_trace(
        self,
        run_id: str,
        agent: str,
        message: str,
        payload: Optional[Dict] = None,
        entry_id: Optional[int] = None,
    ):
        self.conn.execute(
            "INSERT INTO traces (run_id, agent, message, payload, entry_id) VALUES (?, ?, ?, ?, ?)",
            (run_id, agent, message, json.dumps(payload) if payload else None, entry_id),
        )
        self.conn.commit()

    def get_traces(self, run_id: Optional[str] = None, limit: int = 200) -> List[Dict]:
        cursor = self.conn.cursor()
        if run_id:
            cursor.execute(
                "SELECT * FROM traces WHERE run_id = ? ORDER BY timestamp ASC LIMIT ?",
                (run_id, limit),
            )
        else:
            cursor.execute(
                "SELECT * FROM traces ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            )
        columns = [col[0] for col in cursor.description]
        rows = []
        for row in cursor.fetchall():
            entry = dict(zip(columns, row))
            if entry.get("payload"):
                try:
                    entry["payload"] = json.loads(entry["payload"])
                except Exception:
                    pass
            rows.append(entry)
        return rows

    def get_trace_run_ids(self) -> List[str]:
        """Return distinct run_ids ordered by most recent first."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT DISTINCT run_id, MAX(timestamp) as ts FROM traces "
            "GROUP BY run_id ORDER BY ts DESC LIMIT 50"
        )
        return [row[0] for row in cursor.fetchall()]

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ── Phase 2: Supabase ──────────────────────────────────────────────────────────

class CloudDatabase:
    def __init__(self):
        from supabase import create_client
        self.db = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

    # ── Projects ───────────────────────────────────────────────────────────────

    def create_project(self, name: str, description: str = "") -> int:
        r = self.db.table("projects").insert({"name": name, "description": description}).execute()
        return r.data[0]["id"]

    def get_projects(self) -> List[Dict]:
        return self.db.table("projects").select("*, ledger(count)").order("created_at", desc=True).execute().data

    def get_project(self, project_id: int) -> Optional[Dict]:
        r = self.db.table("projects").select("*").eq("id", project_id).execute()
        return r.data[0] if r.data else None

    def delete_project(self, project_id: int):
        self.db.table("projects").delete().eq("id", project_id).execute()

    # ── Ledger ─────────────────────────────────────────────────────────────────

    def insert_ledger_entry(
        self,
        metadata: Dict,
        project_id: Optional[int] = None,
        github_repo: str = "",
        github_branch: str = "main",
        github_commit: str = "",
        team_member: Optional[str] = None,
        auto_extracted: Optional[Dict] = None,
        confidence_score: Optional[float] = None,
        experiment_hash: Optional[str] = None,
        **kwargs,
    ) -> int:
        data = {
            "project_id": project_id,
            "file_path": metadata["file_path"],
            "file_type": metadata.get("file_type", "ipynb"),
            "github_repo": github_repo,
            "github_branch": github_branch,
            "github_commit": github_commit,
            "environment": metadata.get("environment", {}),
            "models": metadata.get("models", []),
            "metrics": metadata.get("metrics", []),
            "preprocessing": metadata.get("preprocessing", []),
            "raw_text": metadata.get("raw_text", ""),
            "team_member": team_member,
            "auto_extracted": auto_extracted,
            "confidence_score": confidence_score,
            "experiment_hash": experiment_hash,
        }
        return self.db.table("ledger").insert(data).execute().data[0]["id"]

    def find_by_hash(self, experiment_hash: str) -> Optional[Dict]:
        """Return the first entry with a matching experiment_hash, or None."""
        r = (
            self.db.table("ledger")
            .select("id, file_path, timestamp")
            .eq("experiment_hash", experiment_hash)
            .limit(1)
            .execute()
        )
        return r.data[0] if r.data else None

    def get_ledger_entries(
        self,
        file_path: Optional[str] = None,
        project_id: Optional[int] = None,
        github_repo: Optional[str] = None,
    ) -> List[Dict]:
        q = self.db.table("ledger").select("*, projects(name)").order("timestamp", desc=True)
        if file_path:
            q = q.eq("file_path", file_path)
        if project_id is not None:
            q = q.eq("project_id", project_id)
        if github_repo:
            q = q.eq("github_repo", github_repo)
        rows = q.execute().data
        # Flatten joined project name to match LocalDatabase shape
        for row in rows:
            proj = row.pop("projects", None) or {}
            row["project_name"] = proj.get("name", "—") if isinstance(proj, dict) else "—"
        return rows

    def update_ledger_entry(self, entry_id: int, updates: Dict):
        self.db.table("ledger").update(updates).eq("id", entry_id).execute()

    def delete_ledger_entry(self, entry_id: int) -> None:
        self.db.table("ledger").delete().eq("id", entry_id).execute()

    def confirm_entry(self, entry_id: int, user_corrected: Dict) -> None:
        """Store user-edited experiment and mark entry as approved."""
        self.db.table("ledger").update({
            "user_corrected": user_corrected,
            "approved": 1,
            "status": "Confirmed",
        }).eq("id", entry_id).execute()

    def get_experiments(
        self,
        project_id: Optional[int] = None,
        approved_only: bool = True,
    ) -> List[Dict]:
        q = self.db.table("ledger").select("*, projects(name)").order("timestamp", desc=True)
        if approved_only:
            q = q.eq("approved", 1)
        if project_id is not None:
            q = q.eq("project_id", project_id)
        rows = q.execute().data
        for row in rows:
            proj = row.pop("projects", None) or {}
            row["project_name"] = proj.get("name", "—") if isinstance(proj, dict) else "—"
        return rows

    def log_cost(
        self,
        plan_id: Optional[int],
        gcp_instance: str,
        time_hours: float,
        cost: float,
        cost_type: str = "autoresearch",
        description: str = "",
    ):
        self.db.table("costs").insert({
            "plan_id": plan_id,
            "gcp_instance": gcp_instance,
            "time_hours": time_hours,
            "cost": cost,
            "cost_type": cost_type,
            "description": description,
        }).execute()

    def get_costs(self) -> List[Dict]:
        return self.db.table("costs").select("*").order("timestamp", desc=True).execute().data

    # ── Traces ─────────────────────────────────────────────────────────────────

    def log_trace(
        self,
        run_id: str,
        agent: str,
        message: str,
        payload: Optional[Dict] = None,
        entry_id: Optional[int] = None,
    ):
        self.db.table("traces").insert({
            "run_id": run_id,
            "agent": agent,
            "message": message,
            "payload": payload,
            "entry_id": entry_id,
        }).execute()

    def get_traces(self, run_id: Optional[str] = None, limit: int = 200) -> List[Dict]:
        q = (
            self.db.table("traces")
            .select("*")
            .order("timestamp", desc=False)
            .limit(limit)
        )
        if run_id:
            q = q.eq("run_id", run_id)
        return q.execute().data

    def get_trace_run_ids(self) -> List[str]:
        r = (
            self.db.table("traces")
            .select("run_id, timestamp")
            .order("timestamp", desc=True)
            .limit(500)
            .execute()
        )
        seen, result = set(), []
        for row in r.data:
            if row["run_id"] not in seen:
                seen.add(row["run_id"])
                result.append(row["run_id"])
        return result[:50]


# ── Factory ────────────────────────────────────────────────────────────────────

def get_db(mode: Optional[str] = None):
    mode = mode or os.getenv("APP_MODE", "local")
    if mode == "cloud":
        return CloudDatabase()
    return LocalDatabase()
