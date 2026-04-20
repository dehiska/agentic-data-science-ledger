"""Tests for src/mcp_server.py — MCPServer file parsing (.ipynb, .py, .docx)."""

import os
import sys
import tempfile
import json

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.mcp_server import MCPServer
from src.database import LocalDatabase


def _make_notebook(cells_source: list) -> str:
    """Write a minimal .ipynb to a temp file and return its path."""
    nb = {
        "nbformat": 4,
        "nbformat_minor": 4,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "cells": [
            {"cell_type": "code", "metadata": {}, "source": src, "outputs": [], "execution_count": None}
            for src in cells_source
        ],
    }
    with tempfile.NamedTemporaryFile(suffix=".ipynb", delete=False, mode="w", encoding="utf-8") as f:
        json.dump(nb, f)
        return f.name


def _make_py(source: str) -> str:
    """Write Python source to a temp .py file and return its path."""
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w", encoding="utf-8") as f:
        f.write(source)
        return f.name


def _make_docx(text: str) -> str:
    """Write text to a temp .docx file and return its path."""
    docx = pytest.importorskip("docx", reason="python-docx not installed")
    from docx import Document
    doc = Document()
    doc.add_paragraph(text)
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as f:
        path = f.name
    doc.save(path)
    return path


@pytest.fixture
def mcp():
    return MCPServer()


# ── .ipynb parsing ─────────────────────────────────────────────────────────────

def test_detects_random_forest(mcp):
    path = _make_notebook(["from sklearn.ensemble import RandomForestClassifier\nrf = RandomForestClassifier(n_estimators=200, max_depth=8)"])
    try:
        meta = mcp.parse_file(path)
        names = [m["name"] for m in meta["models"]]
        assert "RandomForestClassifier" in names
        assert meta["models"][0]["family"] == "Ensemble"
        assert meta["file_type"] == "ipynb"
    finally:
        os.unlink(path)


def test_detects_standard_scaler(mcp):
    path = _make_notebook(["from sklearn.preprocessing import StandardScaler\nscaler = StandardScaler()"])
    try:
        meta = mcp.parse_file(path)
        prep_names = [p["name"] for p in meta["preprocessing"]]
        assert "StandardScaler" in prep_names
    finally:
        os.unlink(path)


def test_detects_metrics(mcp):
    path = _make_notebook([
        "from sklearn.metrics import f1_score, roc_auc_score\n"
        "f1 = f1_score(y_true, y_pred)\n"
        "auc = roc_auc_score(y_true, y_prob)"
    ])
    try:
        meta = mcp.parse_file(path)
        metric_names = [m["name"] for m in meta["metrics"]]
        assert "f1" in metric_names
        assert "auc_roc" in metric_names
    finally:
        os.unlink(path)


def test_handles_syntax_error_cell(mcp):
    """A cell with a SyntaxError should be skipped, not crash the server."""
    path = _make_notebook(["def broken(\n    pass", "from sklearn.linear_model import LinearRegression\nlr = LinearRegression()"])
    try:
        meta = mcp.parse_file(path)
        names = [m["name"] for m in meta["models"]]
        assert "LinearRegression" in names
    finally:
        os.unlink(path)


def test_parse_notebook_alias(mcp):
    """parse_notebook() is an alias for backwards compatibility."""
    path = _make_notebook(["from sklearn.ensemble import RandomForestClassifier\nrf = RandomForestClassifier()"])
    try:
        meta = mcp.parse_notebook(path)
        assert len(meta["models"]) == 1
    finally:
        os.unlink(path)


# ── .py parsing ────────────────────────────────────────────────────────────────

def test_parse_py_detects_model(mcp):
    path = _make_py("from xgboost import XGBClassifier\nmodel = XGBClassifier(n_estimators=100)")
    try:
        meta = mcp.parse_file(path)
        assert meta["file_type"] == "py"
        names = [m["name"] for m in meta["models"]]
        assert "XGBClassifier" in names
    finally:
        os.unlink(path)


def test_parse_py_detects_preprocessing(mcp):
    path = _make_py("from sklearn.preprocessing import MinMaxScaler\nscaler = MinMaxScaler()")
    try:
        meta = mcp.parse_file(path)
        prep_names = [p["name"] for p in meta["preprocessing"]]
        assert "MinMaxScaler" in prep_names
    finally:
        os.unlink(path)


def test_parse_py_raw_text(mcp):
    source = "# training script\nfrom sklearn.linear_model import Ridge\nmodel = Ridge(alpha=1.0)"
    path = _make_py(source)
    try:
        meta = mcp.parse_file(path)
        assert "Ridge" in meta["raw_text"]
    finally:
        os.unlink(path)


def test_parse_py_syntax_error_graceful(mcp):
    path = _make_py("def broken(\n    pass")
    try:
        meta = mcp.parse_file(path)
        assert meta["file_type"] == "py"
        assert meta["models"] == []
    finally:
        os.unlink(path)


# ── .docx parsing ──────────────────────────────────────────────────────────────

def test_parse_docx_keywords(mcp):
    path = _make_docx("This document describes a model trained with cross-validation. We measured accuracy and f1 score.")
    try:
        meta = mcp.parse_file(path)
        assert meta["file_type"] == "docx"
        assert "model" in meta["doc_keywords"]
        assert "accuracy" in meta["doc_keywords"] or "f1" in meta["doc_keywords"]
        assert meta["word_count"] > 0
    finally:
        os.unlink(path)


def test_parse_docx_mentions_model_class(mcp):
    path = _make_docx("We used RandomForestClassifier as our baseline model.")
    try:
        meta = mcp.parse_file(path)
        model_names = [m["name"] for m in meta["models"]]
        assert "RandomForestClassifier" in model_names
    finally:
        os.unlink(path)


def test_parse_docx_raw_text(mcp):
    path = _make_docx("Experiment notes: baseline accuracy 0.82.")
    try:
        meta = mcp.parse_file(path)
        assert "baseline" in meta["raw_text"] or "accuracy" in meta["raw_text"]
    finally:
        os.unlink(path)


# ── Unsupported type ───────────────────────────────────────────────────────────

def test_unsupported_file_type_raises(mcp):
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = f.name
    try:
        with pytest.raises(ValueError, match="Unsupported"):
            mcp.parse_file(path)
    finally:
        os.unlink(path)


# ── store_in_db ────────────────────────────────────────────────────────────────

def test_stores_in_db(mcp):
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    db = LocalDatabase(db_path)
    mcp_with_db = MCPServer(db=db)

    nb_path = _make_notebook(["from sklearn.ensemble import RandomForestRegressor\nrf = RandomForestRegressor()"])
    try:
        meta = mcp_with_db.parse_file(nb_path)
        entry_id = mcp_with_db.store_in_db(meta)
        assert entry_id is not None

        entries = db.get_ledger_entries()
        assert len(entries) == 1
        assert entries[0]["file_type"] == "ipynb"
    finally:
        os.unlink(nb_path)
        db.close()
        os.unlink(db_path)


def test_stores_in_db_with_project(mcp):
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    db = LocalDatabase(db_path)
    mcp_with_db = MCPServer(db=db)
    pid = db.create_project("TestProject")

    nb_path = _make_notebook(["import pandas as pd"])
    try:
        meta = mcp_with_db.parse_file(nb_path)
        entry_id = mcp_with_db.store_in_db(meta, project_id=pid)
        entries = db.get_ledger_entries(project_id=pid)
        assert len(entries) == 1
        assert entries[0]["project_name"] == "TestProject"
    finally:
        os.unlink(nb_path)
        db.close()
        os.unlink(db_path)


def test_example_notebook():
    """Smoke test against the bundled example notebook."""
    example = os.path.join(os.path.dirname(__file__), "..", "data", "example.ipynb")
    if not os.path.exists(example):
        pytest.skip("example.ipynb not found")
    mcp = MCPServer()
    meta = mcp.parse_file(example)
    assert len(meta["models"]) >= 2
    assert len(meta["metrics"]) >= 2
    assert len(meta["preprocessing"]) >= 2
