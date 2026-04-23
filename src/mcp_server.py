"""
MCP Server — parses .ipynb, .py, and .docx files.

For .ipynb / .py: AST-extracts models, metrics, preprocessing steps.
For .docx: extracts raw text + keyword-scans for DS terminology.

Usage:
    from src.mcp_server import MCPServer
    mcp = MCPServer(db)
    metadata = mcp.parse_file("experiment.ipynb")
    metadata = mcp.parse_file("notes.docx")
    metadata = mcp.parse_file("train.py")
    mcp.store_in_db(metadata, project_id=1)
"""

import ast
import hashlib
import json as _json_mod
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import nbformat


class MCPServer:
    MODEL_CLASSES = {
        "RandomForestClassifier": "Ensemble",
        "RandomForestRegressor": "Ensemble",
        "GradientBoostingClassifier": "Ensemble",
        "GradientBoostingRegressor": "Ensemble",
        "ExtraTreesClassifier": "Ensemble",
        "ExtraTreesRegressor": "Ensemble",
        "AdaBoostClassifier": "Ensemble",
        "BaggingClassifier": "Ensemble",
        "XGBClassifier": "Ensemble",
        "XGBRegressor": "Ensemble",
        "XGBRanker": "Ensemble",
        "LGBMClassifier": "Ensemble",
        "LGBMRegressor": "Ensemble",
        "LGBMRanker": "Ensemble",
        "CatBoostClassifier": "Ensemble",
        "CatBoostRegressor": "Ensemble",
        "LogisticRegression": "Linear",
        "LinearRegression": "Linear",
        "Ridge": "Linear",
        "Lasso": "Linear",
        "ElasticNet": "Linear",
        "PoissonRegressor": "Count Data",
        "NegativeBinomialRegressor": "Count Data",
        "SVC": "SVM",
        "SVR": "SVM",
        "KNeighborsClassifier": "KNN",
        "Sequential": "Deep Neural Network",
        "LSTM": "Time Series",
        "GRU": "Time Series",
        "Transformer": "Transformer",
        "BertModel": "Transformer",
        "RobertaModel": "Transformer",
        "KMeans": "Clustering",
        "DBSCAN": "Clustering",
    }

    # Functional training APIs: (module_alias, function) → (display_name, family)
    FRAMEWORK_CALLS = {
        ("lgb", "train"):         ("LightGBM", "Ensemble"),
        ("lgb", "cv"):            ("LightGBM CV", "Ensemble"),
        ("lightgbm", "train"):    ("LightGBM", "Ensemble"),
        ("xgb", "train"):         ("XGBoost", "Ensemble"),
        ("xgb", "cv"):            ("XGBoost CV", "Ensemble"),
        ("xgboost", "train"):     ("XGBoost", "Ensemble"),
        ("catboost", "train"):    ("CatBoost", "Ensemble"),
    }

    # LightGBM / XGBoost objective → model family
    OBJECTIVE_FAMILY = {
        "binary":          "Ensemble (binary)",
        "multiclass":      "Ensemble (multiclass)",
        "regression":      "Ensemble (regression)",
        "regression_l1":   "Ensemble (regression)",
        "huber":           "Ensemble (regression)",
        "rank_xendcg":     "Ensemble (ranking)",
        "binary:logistic": "Ensemble (binary)",
        "multi:softmax":   "Ensemble (multiclass)",
        "reg:squarederror":"Ensemble (regression)",
    }

    # LightGBM / XGBoost metric param values → display metric name
    FRAMEWORK_METRICS = {
        "multi_logloss": "multi_logloss",
        "binary_logloss": "log_loss",
        "auc": "auc_roc",
        "rmse": "rmse",
        "mae": "mae",
        "mse": "mse",
        "logloss": "log_loss",
        "error": "error_rate",
        "merror": "multiclass_error",
        "map": "mean_avg_precision",
        "ndcg": "ndcg",
    }

    METRIC_FUNCTIONS = {
        "accuracy_score": "accuracy",
        "mean_squared_error": "mse",
        "mean_absolute_error": "mae",
        "r2_score": "r2",
        "f1_score": "f1",
        "roc_auc_score": "auc_roc",
        "average_precision_score": "pr_auc",
        "brier_score_loss": "brier",
        "log_loss": "log_loss",
        "confusion_matrix": "confusion_matrix",
        "classification_report": "classification_report",
    }

    PREPROCESSING_STEPS = {
        "StandardScaler": "Scaling",
        "MinMaxScaler": "Scaling",
        "RobustScaler": "Scaling",
        "Normalizer": "Normalization",
        "PCA": "Dimensionality Reduction",
        "TSNE": "Dimensionality Reduction",
        "UMAP": "Dimensionality Reduction",
        "SimpleImputer": "Imputation",
        "KNNImputer": "Imputation",
        "OneHotEncoder": "Encoding",
        "LabelEncoder": "Encoding",
        "TargetEncoder": "Encoding",
        "OrdinalEncoder": "Encoding",
        "PolynomialFeatures": "Feature Engineering",
        "SelectKBest": "Feature Selection",
        "RFE": "Feature Selection",
        "SMOTE": "Resampling",
    }

    # Keywords to surface from .docx text
    DOC_KEYWORDS = [
        "model", "accuracy", "f1", "precision", "recall", "auc", "roc",
        "train", "test", "validation", "overfitting", "baseline", "feature",
        "preprocessing", "normalize", "scale", "impute", "encode", "split",
        "cross-validation", "hyperparameter", "loss", "epoch", "batch",
        "objective", "hypothesis", "dataset", "label", "target",
    ]

    def __init__(self, db=None, github=None):
        self.db = db
        self.github = github

    # ── Public API ─────────────────────────────────────────────────────────────

    def parse_file(
        self,
        file_path: str,
        repo_name: Optional[str] = None,
        branch: str = "main",
        team_member: Optional[str] = None,
    ) -> Dict:
        """Dispatch to the correct parser based on file extension."""
        ext = Path(file_path).suffix.lower()
        if ext == ".ipynb":
            # Check for a hand-crafted ledger entry JSON alongside the notebook
            json_sidecar = str(file_path).replace(".ipynb", "_ledger_entry.json")
            if Path(json_sidecar).exists():
                return self._parse_json(json_sidecar)
            return self._parse_notebook(file_path, repo_name, branch)
        elif ext == ".py":
            return self._parse_py(file_path)
        elif ext in (".docx", ".doc"):
            return self._parse_docx(file_path)
        elif ext == ".json":
            return self._parse_json(file_path)
        elif ext == ".csv":
            return self._parse_csv(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}. Supported: .ipynb, .py, .docx, .json, .csv")

    # Keep old name for backwards compatibility
    def parse_notebook(self, file_path: str, repo_name=None, branch="main", team_member=None) -> Dict:
        return self.parse_file(file_path, repo_name, branch, team_member)

    def store_in_db(self, metadata: Dict, project_id: Optional[int] = None, **kwargs) -> Optional[int]:
        if self.db is None:
            return None
        exp_hash = generate_experiment_hash(metadata)
        return self.db.insert_ledger_entry(
            metadata, project_id=project_id, experiment_hash=exp_hash, **kwargs
        )

    # ── .ipynb parser ──────────────────────────────────────────────────────────

    def _parse_notebook(self, file_path: str, repo_name=None, branch="main") -> Dict:
        local_path = file_path
        if self.github and repo_name:
            nb_data = self.github.get_notebook(repo_name, file_path, branch)
            if not nb_data:
                raise ValueError(f"Notebook {file_path} not found in {repo_name}.")
            local_path = "_temp_notebook.ipynb"
            Path(local_path).write_text(nb_data["content"], encoding="utf-8")

        try:
            meta = self._parse_notebook_local(local_path)
            meta["file_path"] = file_path  # restore original name
        finally:
            if local_path == "_temp_notebook.ipynb":
                Path(local_path).unlink(missing_ok=True)
        return meta

    def _parse_notebook_local(self, file_path: str) -> Dict:
        with open(file_path, "r", encoding="utf-8") as f:
            notebook = nbformat.read(f, as_version=4)

        meta: Dict = {
            "file_path": file_path,
            "file_type": "ipynb",
            "cells": [],
            "environment": self._extract_environment(),
            "models": [],
            "metrics": [],
            "preprocessing": [],
            "raw_text": "",
        }

        all_source = []
        for cell_index, cell in enumerate(notebook.cells):
            if cell.cell_type == "markdown":
                all_source.append(cell.source)
                continue
            if cell.cell_type != "code":
                continue
            all_source.append(cell.source)
            try:
                tree = ast.parse(cell.source)
            except SyntaxError:
                continue

            cell_meta: Dict = {"cell_index": cell_index, "source": cell.source[:500],
                                "models": [], "metrics": [], "preprocessing": []}
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    self._extract_call(node, cell_meta)

            meta["cells"].append(cell_meta)
            meta["models"].extend(cell_meta["models"])
            meta["metrics"].extend(cell_meta["metrics"])
            meta["preprocessing"].extend(cell_meta["preprocessing"])

        meta["raw_text"] = "\n".join(all_source)[:10000]
        self._extract_source_patterns(meta["raw_text"], meta)
        return meta

    # ── .py parser ─────────────────────────────────────────────────────────────

    def _parse_py(self, file_path: str) -> Dict:
        source = Path(file_path).read_text(encoding="utf-8", errors="replace")
        meta: Dict = {
            "file_path": file_path,
            "file_type": "py",
            "cells": [],
            "environment": self._extract_environment(),
            "models": [],
            "metrics": [],
            "preprocessing": [],
            "raw_text": source[:10000],
        }
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return meta

        cell_meta = {"cell_index": 0, "source": source[:500], "models": [], "metrics": [], "preprocessing": []}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                self._extract_call(node, cell_meta)

        meta["models"] = cell_meta["models"]
        meta["metrics"] = cell_meta["metrics"]
        meta["preprocessing"] = cell_meta["preprocessing"]
        self._extract_source_patterns(source, meta)
        return meta

    # ── .json ledger entry parser ──────────────────────────────────────────────

    def _parse_json(self, file_path: str) -> Dict:
        """
        Ingest a hand-crafted *_ledger_entry.json file directly into the ledger.

        The JSON must match the schema produced by the notebook logging cell:
          { file_path, models, metrics, preprocessing, environment, timestamp, ... }

        Any missing fields default to empty lists/dicts so partial entries work.
        """
        import json as _json
        with open(file_path, "r", encoding="utf-8") as f:
            data = _json.load(f)

        # Normalise: ensure all required keys exist
        return {
            "file_path": data.get("file_path", str(file_path)),
            "file_type": "json",
            "cells": [],
            "environment": data.get("environment", {}),
            "models": data.get("models", []),
            "metrics": data.get("metrics", []),
            "preprocessing": data.get("preprocessing", []),
            "raw_text": _json.dumps(data, indent=2)[:10000],
            # Pass through any extra fields (team_member, timestamp, etc.)
            **{k: v for k, v in data.items()
               if k not in ("file_path", "environment", "models",
                            "metrics", "preprocessing")},
        }

    # ── .docx parser ───────────────────────────────────────────────────────────

    def _parse_docx(self, file_path: str) -> Dict:
        raw_text = self._read_docx(file_path)
        text_lower = raw_text.lower()

        # Keyword scan to surface DS mentions
        found_keywords = [kw for kw in self.DOC_KEYWORDS if kw in text_lower]

        # Heuristic: scan for model class names mentioned as text
        mentioned_models = [
            {"name": name, "family": family, "source": "doc_text"}
            for name, family in self.MODEL_CLASSES.items()
            if name.lower() in text_lower
        ]

        return {
            "file_path": file_path,
            "file_type": "docx",
            "cells": [],
            "environment": {},
            "models": mentioned_models,
            "metrics": [],
            "preprocessing": [],
            "raw_text": raw_text[:5000],
            "doc_keywords": found_keywords,
            "word_count": len(raw_text.split()),
        }

    def _read_docx(self, file_path: str) -> str:
        try:
            from docx import Document
            doc = Document(file_path)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            # Also extract table cells
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        if cell.text.strip():
                            paragraphs.append(cell.text.strip())
            return "\n".join(paragraphs)
        except ImportError:
            raise ImportError("python-docx is required for .docx parsing. Run: pip install python-docx")
        except Exception as e:
            raise ValueError(f"Could not read {file_path}: {e}")

    # ── AST helpers ────────────────────────────────────────────────────────────

    def _extract_call(self, node: ast.Call, cell_meta: Dict):
        func = node.func

        # ── Class instantiation: RandomForestClassifier(), etc. ──
        if isinstance(func, ast.Name):
            func_name = func.id
            if func_name in self.MODEL_CLASSES:
                cell_meta["models"].append({
                    "name": func_name,
                    "family": self.MODEL_CLASSES[func_name],
                    "params": {kw.arg: ast.unparse(kw.value) for kw in node.keywords if kw.arg},
                    "cell_index": cell_meta["cell_index"],
                    "line_number": node.lineno,
                    "code": ast.unparse(node)[:200],
                })
            elif func_name in self.METRIC_FUNCTIONS:
                cell_meta["metrics"].append({
                    "name": self.METRIC_FUNCTIONS[func_name],
                    "function": func_name,
                    "args": [ast.unparse(a) for a in node.args],
                    "cell_index": cell_meta["cell_index"],
                    "line_number": node.lineno,
                    "code": ast.unparse(node)[:200],
                })
            elif func_name in self.PREPROCESSING_STEPS:
                cell_meta["preprocessing"].append({
                    "name": func_name,
                    "type": self.PREPROCESSING_STEPS[func_name],
                    "params": {kw.arg: ast.unparse(kw.value) for kw in node.keywords if kw.arg},
                    "cell_index": cell_meta["cell_index"],
                    "line_number": node.lineno,
                    "code": ast.unparse(node)[:200],
                })

        # ── Attribute call: lgb.train(), xgb.train(), model.fit(), etc. ──
        elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            module = func.value.id
            attr = func.attr
            fw_key = (module, attr)

            if fw_key in self.FRAMEWORK_CALLS:
                display_name, family = self.FRAMEWORK_CALLS[fw_key]
                # Pull params dict from first positional arg (if it's a Name like `params`)
                # or from keyword arg `params=`
                params_kw = {kw.arg: ast.unparse(kw.value) for kw in node.keywords if kw.arg}
                cell_meta["models"].append({
                    "name": display_name,
                    "family": family,
                    "params": params_kw,
                    "cell_index": cell_meta["cell_index"],
                    "line_number": node.lineno,
                    "code": ast.unparse(node)[:200],
                })

            # module.ClassName() — e.g. lgb.LGBMClassifier(), catboost.CatBoostClassifier()
            elif attr in self.MODEL_CLASSES:
                cell_meta["models"].append({
                    "name": attr,
                    "family": self.MODEL_CLASSES[attr],
                    "params": {kw.arg: ast.unparse(kw.value) for kw in node.keywords if kw.arg},
                    "cell_index": cell_meta["cell_index"],
                    "line_number": node.lineno,
                    "code": ast.unparse(node)[:200],
                })
            elif attr in self.METRIC_FUNCTIONS:
                cell_meta["metrics"].append({
                    "name": self.METRIC_FUNCTIONS[attr],
                    "function": attr,
                    "args": [ast.unparse(a) for a in node.args],
                    "cell_index": cell_meta["cell_index"],
                    "line_number": node.lineno,
                    "code": ast.unparse(node)[:200],
                })
            elif attr in self.PREPROCESSING_STEPS:
                cell_meta["preprocessing"].append({
                    "name": attr,
                    "type": self.PREPROCESSING_STEPS[attr],
                    "params": {kw.arg: ast.unparse(kw.value) for kw in node.keywords if kw.arg},
                    "cell_index": cell_meta["cell_index"],
                    "line_number": node.lineno,
                    "code": ast.unparse(node)[:200],
                })

    def _extract_source_patterns(self, source: str, meta: Dict):
        """Regex-based detection for patterns AST misses (category encoding, class weights,
        framework params dicts with metric/objective keys)."""
        import re

        # ── Preprocessing: category encoding ──────────────────────────────────
        if re.search(r"astype\(['\"]category['\"]\)", source):
            if not any(p["name"] == "CategoryEncoding" for p in meta["preprocessing"]):
                meta["preprocessing"].append({
                    "name": "CategoryEncoding", "type": "Encoding",
                    "params": {}, "source": "astype('category')",
                })

        # ── Preprocessing: sample/class weighting ─────────────────────────────
        if re.search(r"\b(sample_weight|class_weight|sample_weights|class_weights)\b", source):
            if not any(p["name"] == "ClassWeighting" for p in meta["preprocessing"]):
                meta["preprocessing"].append({
                    "name": "ClassWeighting", "type": "Resampling",
                    "params": {}, "source": "sample_weight / class_weight",
                })

        # ── Metrics: LightGBM / XGBoost 'metric' key in params dict ──────────
        for m in re.findall(r"['\"]metric['\"]\s*:\s*['\"]([^'\"]+)['\"]", source):
            for part in m.split(","):
                part = part.strip()
                display = self.FRAMEWORK_METRICS.get(part, part)
                if not any(x["name"] == display for x in meta["metrics"]):
                    meta["metrics"].append({
                        "name": display, "function": "framework_metric",
                        "args": [], "source": f"params dict: metric={part}",
                    })

        # ── Models: regex fallback for framework training calls ───────────────
        # Catches lgb.train(), xgb.train() etc. when AST walk misses them
        # (e.g. large cells where lgb.train is deep inside a for-loop body
        # and the cell source was truncated before reaching it in raw_text)
        fw_regex = [
            (r"\blgb\.train\s*\(",      "LightGBM",    "Ensemble"),
            (r"\blgb\.cv\s*\(",         "LightGBM CV", "Ensemble"),
            (r"\blightgbm\.train\s*\(", "LightGBM",    "Ensemble"),
            (r"\bxgb\.train\s*\(",      "XGBoost",     "Ensemble"),
            (r"\bxgb\.cv\s*\(",         "XGBoost CV",  "Ensemble"),
            (r"\bxgboost\.train\s*\(",  "XGBoost",     "Ensemble"),
            (r"\bcatboost\.train\s*\(", "CatBoost",    "Ensemble"),
        ]
        for pattern, name, family in fw_regex:
            if re.search(pattern, source):
                if not any(m.get("name") == name for m in meta["models"]):
                    meta["models"].append({
                        "name": name, "family": family,
                        "params": {}, "source": "regex",
                    })

        # ── Models: LightGBM / XGBoost 'objective' key → annotate/create ─────
        for obj in re.findall(r"['\"]objective['\"]\s*:\s*['\"]([^'\"]+)['\"]", source):
            family = self.OBJECTIVE_FAMILY.get(obj, f"Ensemble ({obj})")
            patched = False
            for m in meta["models"]:
                if m.get("name") in ("LightGBM", "XGBoost", "LightGBM CV", "XGBoost CV",
                                     "CatBoost"):
                    m["family"] = family
                    m.setdefault("params", {})["objective"] = obj
                    patched = True
            # If nothing was found at all, create a generic entry from the objective
            if not patched and obj in self.OBJECTIVE_FAMILY:
                meta["models"].append({
                    "name": "GradientBoosting", "family": family,
                    "params": {"objective": obj}, "source": "objective key",
                })

    # ── .csv dataset parser ────────────────────────────────────────────────────

    def _parse_csv(self, file_path: str) -> Dict:
        """
        Parse a CSV dataset file — extracts shape, column info, dtypes, and
        heuristically guesses the target column and task type.

        Reads only the header + a small sample to keep memory usage low —
        the full dataset is loaded later by SwarmOrchestrator.
        """
        import pandas as pd

        COMMON_TARGETS = {
            "target", "label", "y", "class", "outcome", "response",
            "cancel", "churn", "default", "fraud", "is_fraud", "survived",
            "status", "result", "price", "sales", "revenue",
        }

        try:
            df_sample = pd.read_csv(file_path, nrows=500, low_memory=False)
        except Exception as e:
            return {
                "file_path": file_path,
                "file_type": "csv",
                "cells": [],
                "environment": {},
                "models": [],
                "metrics": [],
                "preprocessing": [],
                "raw_text": f"Error reading CSV: {e}",
                "dataset_info": {},
            }

        # Count total rows cheaply (no full load)
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                n_rows = sum(1 for _ in f) - 1  # subtract header
        except Exception:
            n_rows = len(df_sample)

        columns = df_sample.columns.tolist()
        dtypes = {col: str(df_sample[col].dtype) for col in columns}

        # Guess target column from names
        target_col = None
        for col in columns:
            if col.lower() in COMMON_TARGETS:
                target_col = col
                break

        # Guess task type from target column cardinality
        task_type = "unknown"
        if target_col:
            nunique = df_sample[target_col].nunique()
            task_type = "classification" if nunique <= 20 else "regression"

        dataset_info = {
            "n_rows": n_rows,
            "n_cols": len(columns),
            "columns": columns,
            "dtypes": dtypes,
            "target_col_guess": target_col,
            "task_type_guess": task_type,
            "sample": df_sample.head(3).to_dict(orient="records"),
        }

        col_summary = ", ".join(columns[:15]) + ("…" if len(columns) > 15 else "")
        raw_text = (
            f"CSV Dataset: {n_rows:,} rows × {len(columns)} cols.\n"
            f"Columns: {col_summary}\n"
            f"Guessed target: {target_col or 'unknown'} | Task: {task_type}"
        )

        return {
            "file_path": file_path,
            "file_type": "csv",
            "cells": [],
            "environment": {},
            "models": [],
            "metrics": [],
            "preprocessing": [],
            "raw_text": raw_text,
            "dataset_info": dataset_info,
        }

    def _extract_environment(self) -> Dict:
        try:
            pip_out = subprocess.check_output(
                [sys.executable, "-m", "pip", "freeze"], stderr=subprocess.DEVNULL
            ).decode("utf-8")
            return {"pip_freeze": pip_out,
                    "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"}
        except Exception as e:
            return {"error": str(e)}


def generate_experiment_hash(metadata: Dict) -> str:
    """
    Stable MD5 hash of (model names + params + preprocessing names).
    Identical experiments produce the same hash regardless of run order.
    """
    key = _json_mod.dumps(
        {
            "models": sorted(
                [{"name": m.get("name", ""), "params": m.get("params") or {}}
                 for m in metadata.get("models", [])],
                key=lambda x: x["name"],
            ),
            "preprocessing": sorted(
                [p.get("name", "") for p in metadata.get("preprocessing", [])],
            ),
        },
        sort_keys=True,
    )
    return hashlib.md5(key.encode()).hexdigest()
