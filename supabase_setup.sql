-- Agentic DS Ledger — Supabase Schema Setup
-- Run this in the Supabase SQL editor: https://app.supabase.com/project/<your-project>/sql
-- Phase 2 only. Phase 1 uses SQLite (automatic, no setup needed).

-- ── Projects table ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS projects (
    id              SERIAL PRIMARY KEY,
    name            TEXT NOT NULL UNIQUE,
    description     TEXT DEFAULT '',
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_projects_name ON projects(name);

-- ── Ledger table ──────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS ledger (
    id              SERIAL PRIMARY KEY,
    project_id      INTEGER REFERENCES projects(id) ON DELETE SET NULL,
    file_path       TEXT NOT NULL,
    file_type       TEXT NOT NULL DEFAULT 'ipynb',
    github_repo     TEXT NOT NULL DEFAULT '',
    github_branch   TEXT NOT NULL DEFAULT 'main',
    github_commit   TEXT,
    environment     JSONB DEFAULT '{}',
    models          JSONB DEFAULT '[]',
    metrics         JSONB DEFAULT '[]',
    preprocessing   JSONB DEFAULT '[]',
    raw_text        TEXT DEFAULT '',
    timestamp       TIMESTAMPTZ DEFAULT NOW(),
    team_member     TEXT,
    status          TEXT DEFAULT 'Pending',
    supabase_id     UUID DEFAULT gen_random_uuid(),
    -- v2: inference + confirmation
    auto_extracted  JSONB DEFAULT NULL,
    user_corrected  JSONB DEFAULT NULL,
    confidence_score REAL DEFAULT NULL,
    approved        INTEGER DEFAULT 0,
    -- v2.1: deduplication
    experiment_hash TEXT DEFAULT NULL
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_ledger_project_id   ON ledger(project_id);
CREATE INDEX IF NOT EXISTS idx_ledger_file_path    ON ledger(file_path);
CREATE INDEX IF NOT EXISTS idx_ledger_file_type    ON ledger(file_type);
CREATE INDEX IF NOT EXISTS idx_ledger_github_repo  ON ledger(github_repo);
CREATE INDEX IF NOT EXISTS idx_ledger_team_member      ON ledger(team_member);
CREATE INDEX IF NOT EXISTS idx_ledger_timestamp        ON ledger(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_ledger_experiment_hash  ON ledger(experiment_hash);
CREATE INDEX IF NOT EXISTS idx_ledger_approved         ON ledger(approved);

-- ── Team members table ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS team_members (
    id              SERIAL PRIMARY KEY,
    github_username TEXT UNIQUE NOT NULL,
    name            TEXT,
    email           TEXT,
    last_active     TIMESTAMPTZ DEFAULT NOW()
);

-- ── Costs table ───────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS costs (
    id              SERIAL PRIMARY KEY,
    plan_id         INTEGER REFERENCES ledger(id) ON DELETE SET NULL,
    gcp_instance    TEXT,
    time_hours      REAL,
    cost            REAL,
    timestamp       TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_costs_plan_id   ON costs(plan_id);
CREATE INDEX IF NOT EXISTS idx_costs_timestamp ON costs(timestamp DESC);

-- ── Row Level Security (optional, recommended for multi-user) ─────────────────
-- Uncomment to restrict reads to authenticated users only:
-- ALTER TABLE projects       ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE ledger         ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE team_members   ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE costs          ENABLE ROW LEVEL SECURITY;
-- CREATE POLICY "Public read" ON projects FOR SELECT USING (true);
-- CREATE POLICY "Public read" ON ledger FOR SELECT USING (true);
-- CREATE POLICY "Authenticated insert" ON ledger FOR INSERT WITH CHECK (auth.role() = 'authenticated');

-- ── Sample verification query ─────────────────────────────────────────────────
-- Run after setup to confirm tables exist:
SELECT table_name FROM information_schema.tables
WHERE table_schema = 'public'
  AND table_name IN ('projects', 'ledger', 'team_members', 'costs');
