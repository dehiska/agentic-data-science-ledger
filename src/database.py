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
        """)
        self.conn.commit()
        self._migrate()

    def _migrate(self):
        """Add columns that were introduced after initial schema creation."""
        existing = {row[1] for row in self.conn.execute("PRAGMA table_info(ledger)")}
        migrations = [
            ("project_id", "INTEGER REFERENCES projects(id) ON DELETE SET NULL"),
            ("file_type",  "TEXT DEFAULT 'ipynb'"),
            ("raw_text",   "TEXT DEFAULT ''"),
            ("status",     "TEXT DEFAULT 'Pending'"),
        ]
        for col, definition in migrations:
            if col not in existing:
                self.conn.execute(f"ALTER TABLE ledger ADD COLUMN {col} {definition}")
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

    def insert_ledger_entry(self, metadata: Dict, project_id: Optional[int] = None, **kwargs) -> int:
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO ledger
              (project_id, file_path, file_type, environment, models, metrics, preprocessing, raw_text)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
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
            ),
        )
        self.conn.commit()
        return cursor.lastrowid

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

    # ── Costs ──────────────────────────────────────────────────────────────────

    def log_cost(self, plan_id: int, gcp_instance: str, time_hours: float, cost: float):
        self.conn.execute(
            "INSERT INTO costs (plan_id, gcp_instance, time_hours, cost) VALUES (?, ?, ?, ?)",
            (plan_id, gcp_instance, time_hours, cost),
        )
        self.conn.commit()

    def get_costs(self) -> List[Dict]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM costs ORDER BY timestamp DESC")
        columns = [col[0] for col in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

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
        }
        return self.db.table("ledger").insert(data).execute().data[0]["id"]

    def get_ledger_entries(
        self,
        file_path: Optional[str] = None,
        project_id: Optional[int] = None,
        github_repo: Optional[str] = None,
    ) -> List[Dict]:
        q = self.db.table("ledger").select("*").order("timestamp", desc=True)
        if file_path:
            q = q.eq("file_path", file_path)
        if project_id is not None:
            q = q.eq("project_id", project_id)
        if github_repo:
            q = q.eq("github_repo", github_repo)
        return q.execute().data

    def update_ledger_entry(self, entry_id: int, updates: Dict):
        self.db.table("ledger").update(updates).eq("id", entry_id).execute()

    def log_cost(self, plan_id: int, gcp_instance: str, time_hours: float, cost: float):
        self.db.table("costs").insert(
            {"plan_id": plan_id, "gcp_instance": gcp_instance, "time_hours": time_hours, "cost": cost}
        ).execute()

    def get_costs(self) -> List[Dict]:
        return self.db.table("costs").select("*").order("timestamp", desc=True).execute().data


# ── Factory ────────────────────────────────────────────────────────────────────

def get_db(mode: Optional[str] = None):
    mode = mode or os.getenv("APP_MODE", "local")
    if mode == "cloud":
        return CloudDatabase()
    return LocalDatabase()
