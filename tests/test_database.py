"""Tests for src/database.py — LocalDatabase (Phase 1)."""

import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.database import LocalDatabase, get_db


@pytest.fixture
def db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    database = LocalDatabase(path)
    yield database
    database.close()
    os.unlink(path)


def _sample_metadata(file_path="notebooks/test.ipynb", file_type="ipynb"):
    return {
        "file_path": file_path,
        "file_type": file_type,
        "models": [{"name": "RandomForestClassifier", "family": "Ensemble", "params": {"n_estimators": "100"}}],
        "metrics": [{"name": "f1", "function": "f1_score", "line_number": 42}],
        "preprocessing": [{"name": "StandardScaler", "type": "Scaling"}],
        "environment": {"python_version": "3.11.0"},
        "raw_text": "sample raw text",
    }


# ── Project CRUD ───────────────────────────────────────────────────────────────

def test_create_and_list_projects(db):
    pid = db.create_project("TestProject", "A test project")
    assert isinstance(pid, int)
    projects = db.get_projects()
    assert len(projects) == 1
    assert projects[0]["name"] == "TestProject"
    assert projects[0]["description"] == "A test project"


def test_get_project(db):
    pid = db.create_project("MyProject")
    p = db.get_project(pid)
    assert p is not None
    assert p["name"] == "MyProject"


def test_get_project_not_found(db):
    assert db.get_project(9999) is None


def test_delete_project(db):
    pid = db.create_project("ToDelete")
    db.delete_project(pid)
    assert db.get_project(pid) is None


def test_project_name_unique(db):
    db.create_project("UniqueProject")
    import sqlite3
    with pytest.raises(sqlite3.IntegrityError):
        db.create_project("UniqueProject")


def test_project_file_count(db):
    pid = db.create_project("CountProject")
    db.insert_ledger_entry(_sample_metadata("a.ipynb"), project_id=pid)
    db.insert_ledger_entry(_sample_metadata("b.py", "py"), project_id=pid)
    projects = db.get_projects()
    assert projects[0]["file_count"] == 2


# ── Ledger CRUD ────────────────────────────────────────────────────────────────

def test_insert_and_retrieve(db):
    meta = _sample_metadata()
    entry_id = db.insert_ledger_entry(meta)
    assert isinstance(entry_id, int)
    assert entry_id >= 1

    entries = db.get_ledger_entries()
    assert len(entries) == 1
    assert entries[0]["file_path"] == "notebooks/test.ipynb"


def test_file_type_stored(db):
    db.insert_ledger_entry(_sample_metadata("train.py", "py"))
    entries = db.get_ledger_entries()
    assert entries[0]["file_type"] == "py"


def test_models_are_deserialized(db):
    db.insert_ledger_entry(_sample_metadata())
    entries = db.get_ledger_entries()
    models = entries[0]["models"]
    assert isinstance(models, list)
    assert models[0]["name"] == "RandomForestClassifier"


def test_filter_by_file_path(db):
    db.insert_ledger_entry(_sample_metadata("a.ipynb"))
    db.insert_ledger_entry(_sample_metadata("b.ipynb"))

    result = db.get_ledger_entries(file_path="a.ipynb")
    assert len(result) == 1
    assert result[0]["file_path"] == "a.ipynb"


def test_filter_by_project_id(db):
    pid1 = db.create_project("ProjectA")
    pid2 = db.create_project("ProjectB")
    db.insert_ledger_entry(_sample_metadata("a.ipynb"), project_id=pid1)
    db.insert_ledger_entry(_sample_metadata("b.ipynb"), project_id=pid2)

    result = db.get_ledger_entries(project_id=pid1)
    assert len(result) == 1
    assert result[0]["file_path"] == "a.ipynb"
    assert result[0]["project_name"] == "ProjectA"


def test_update_entry(db):
    entry_id = db.insert_ledger_entry(_sample_metadata())
    db.update_ledger_entry(entry_id, {"status": "Executed"})

    entries = db.get_ledger_entries()
    assert entries[0]["status"] == "Executed"


def test_log_and_get_costs(db):
    entry_id = db.insert_ledger_entry(_sample_metadata())
    db.log_cost(entry_id, "n1-standard-4", 2.0, 0.38)

    costs = db.get_costs()
    assert len(costs) == 1
    assert costs[0]["gcp_instance"] == "n1-standard-4"
    assert abs(costs[0]["cost"] - 0.38) < 1e-9


def test_get_db_returns_local():
    db = get_db("local")
    assert isinstance(db, LocalDatabase)
    db.close()
