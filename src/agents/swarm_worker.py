"""
SwarmWorkerAgent — one leaf node in the divide-and-conquer swarm.

Called by SwarmOrchestrator._run_workers() inside a ThreadPoolExecutor.
Each worker gets a stratified chunk of the full dataset and runs FLAML AutoML
for `time_budget` seconds, returning its best model.

Falls back to scikit-learn cross-validation if FLAML is not installed
(useful for local dev without the full AutoML stack).
"""

import uuid
from typing import Dict, List, Optional

import pandas as pd


class SwarmWorkerAgent:
    # Maps partial model class names → display family
    _FAMILY_MAP = {
        "randomforest":      "Ensemble",
        "extratree":         "Ensemble",
        "gradientboosting":  "Ensemble",
        "xgb":               "Ensemble",
        "lgbm":              "Ensemble",
        "lightgbm":          "Ensemble",
        "catboost":          "Ensemble",
        "histgradient":      "Ensemble",
        "logisticregression":"Linear",
        "linearregression":  "Linear",
        "ridge":             "Linear",
        "lasso":             "Linear",
        "elasticnet":        "Linear",
        "svc":               "SVM",
        "svr":               "SVM",
        "kneighbors":        "KNN",
        "decisiontree":      "Tree",
    }

    def __init__(self, db=None):
        self.db = db

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(
        self,
        chunk: pd.DataFrame,
        target_col: str,
        task_type: str = "classification",
        time_budget: int = 60,
        run_id: Optional[str] = None,
        agent_id: int = 0,
    ) -> Dict:
        run_id = run_id or uuid.uuid4().hex[:8]
        agent_label = f"swarm_worker_{agent_id}"
        n_rows = len(chunk)

        self._log(run_id, agent_label,
                  f"Worker {agent_id} starting — {n_rows:,} rows, "
                  f"target={target_col}, task={task_type}, budget={time_budget}s")

        # ── Prepare X, y ───────────────────────────────────────────────────────
        try:
            X, y = self._prepare_features(chunk, target_col)
        except Exception as e:
            return {"status": "error",
                    "error": f"Feature prep failed: {e}",
                    "agent_id": agent_id, "run_id": run_id}

        self._log(run_id, agent_label,
                  f"Worker {agent_id} feature matrix: "
                  f"{X.shape[0]:,} rows × {X.shape[1]} features")

        # ── Try FLAML first ────────────────────────────────────────────────────
        result = self._run_flaml(X, y, task_type, time_budget,
                                 run_id, agent_label, agent_id)
        if result["status"] == "ok":
            return result

        # ── Sklearn fallback ───────────────────────────────────────────────────
        return self._run_sklearn_fallback(X, y, task_type,
                                          run_id, agent_label, agent_id)

    # ── Feature preparation ────────────────────────────────────────────────────

    def _prepare_features(
        self, chunk: pd.DataFrame, target_col: str
    ) -> tuple:
        """
        Returns (X, y).
        - Drops non-numeric columns (FLAML handles categoricals, but the
          sklearn fallback needs numerics only)
        - Fills NaN with column median for numerics, 0 for others
        """
        y = chunk[target_col].copy()
        X = chunk.drop(columns=[target_col])

        # Keep only numeric columns for portability across FLAML + sklearn
        num_cols: List[str] = X.select_dtypes(include=["number"]).columns.tolist()
        X = X[num_cols].copy()

        # Fill NaN: median for each column (fast, preserves scale)
        for col in X.columns:
            if X[col].isna().any():
                X[col] = X[col].fillna(X[col].median())

        return X, y

    # ── FLAML runner ───────────────────────────────────────────────────────────

    def _run_flaml(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        task_type: str,
        time_budget: int,
        run_id: str,
        agent_label: str,
        agent_id: int,
    ) -> Dict:
        try:
            from flaml import AutoML  # noqa: F401
        except ImportError:
            return {"status": "error", "error": "flaml not installed", "agent_id": agent_id}

        try:
            # FLAML 2.x classification metrics: accuracy, log_loss, f1, micro_f1, macro_f1
            # Use macro_f1 — works for binary AND multiclass; avoids binary-only f1 error
            metric = "macro_f1" if task_type == "classification" else "r2"

            automl = __import__("flaml").AutoML()
            automl.fit(
                X, y,
                task=task_type,
                time_budget=time_budget,
                metric=metric,
                verbose=0,
                n_jobs=1,           # parallelism is at the swarm level
                seed=42 + agent_id,
            )

            best_cls_name = type(automl.model.estimator).__name__

            # FLAML minimises loss: for maximisation metrics, loss = 1 - metric
            raw_loss = automl.best_loss
            # FLAML minimises loss. For maximisation metrics: loss = 1 - metric
            # For r2: loss = -r2
            maximise_metrics = {"f1", "macro_f1", "micro_f1", "accuracy",
                                 "roc_auc", "roc_auc_ovr", "roc_auc_ovo", "ap"}
            if metric in maximise_metrics:
                val_score = round(1.0 - raw_loss, 4)
            else:
                val_score = round(-raw_loss, 4)

            # Sanitise config for JSON serialisation
            best_config = {
                k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                for k, v in (automl.best_config or {}).items()
            }

            best_model = {
                "name":           best_cls_name,
                "family":         self._get_family(best_cls_name),
                "params":         best_config,
                "metrics":        {"val_score": val_score, metric: val_score},
                "agent_id":       agent_id,
                "n_rows_trained": len(X),
                "flaml":          True,
            }

            self._log(run_id, agent_label,
                      f"Worker {agent_id} FLAML done — "
                      f"{best_cls_name} {metric}={val_score:.4f} ✅",
                      payload=best_model)
            return {
                "status": "ok",
                "best_model": best_model,
                "agent_id": agent_id,
                "run_id": run_id,
            }

        except Exception as e:
            self._log(run_id, agent_label,
                      f"Worker {agent_id} FLAML error: {e}")
            return {"status": "error", "error": str(e), "agent_id": agent_id}

    # ── Sklearn fallback ───────────────────────────────────────────────────────

    def _run_sklearn_fallback(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        task_type: str,
        run_id: str,
        agent_label: str,
        agent_id: int,
    ) -> Dict:
        """Quick 3-fold CV across a small model grid — used when FLAML is absent."""
        self._log(run_id, agent_label,
                  f"Worker {agent_id} — FLAML unavailable, "
                  f"falling back to sklearn grid search")
        try:
            import numpy as np
            from sklearn.model_selection import cross_val_score

            if task_type == "classification":
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.linear_model import LogisticRegression
                candidates = [
                    ("RandomForestClassifier",
                     RandomForestClassifier(
                         n_estimators=100, n_jobs=1,
                         random_state=42 + agent_id),
                     "Ensemble"),
                    ("LogisticRegression",
                     LogisticRegression(
                         max_iter=300, random_state=42 + agent_id),
                     "Linear"),
                ]
                scoring = "f1_macro"  # sklearn: macro avg across classes
            else:
                from sklearn.ensemble import RandomForestRegressor
                from sklearn.linear_model import Ridge
                candidates = [
                    ("RandomForestRegressor",
                     RandomForestRegressor(
                         n_estimators=100, n_jobs=1,
                         random_state=42 + agent_id),
                     "Ensemble"),
                    ("Ridge", Ridge(), "Linear"),
                ]
                scoring = "r2"

            # Cap rows for speed in the fallback path
            cap = 5_000
            X_sub = X.iloc[:cap]
            y_sub = y.iloc[:cap]

            best_name, best_score, best_family, best_params = None, -999.0, "Unknown", {}
            for name, clf, family in candidates:
                try:
                    scores = cross_val_score(
                        clf, X_sub, y_sub, cv=3, scoring=scoring, n_jobs=1
                    )
                    s = float(np.mean(scores))
                    if s > best_score:
                        best_score, best_name = s, name
                        best_family = family
                        best_params = {
                            k: v for k, v in clf.get_params().items()
                            if isinstance(v, (int, float, str, bool, type(None)))
                        }
                except Exception:
                    continue

            if best_name is None:
                return {
                    "status": "error",
                    "error": "All sklearn fallback models failed",
                    "agent_id": agent_id,
                }

            best_model = {
                "name":           best_name,
                "family":         best_family,
                "params":         best_params,
                "metrics":        {
                    "val_score": round(best_score, 4),
                    "f1" if "f1" in scoring else scoring: round(best_score, 4),
                },
                "agent_id":       agent_id,
                "n_rows_trained": len(X_sub),
                "flaml":          False,
            }

            self._log(run_id, agent_label,
                      f"Worker {agent_id} sklearn fallback done — "
                      f"{best_name} {scoring}={best_score:.4f}",
                      payload=best_model)
            return {
                "status": "ok",
                "best_model": best_model,
                "agent_id": agent_id,
                "run_id": run_id,
            }

        except Exception as e:
            return {
                "status": "error",
                "error": f"Sklearn fallback failed: {e}",
                "agent_id": agent_id,
            }

    # ── Utilities ──────────────────────────────────────────────────────────────

    def _get_family(self, model_name: str) -> str:
        name_lower = model_name.lower()
        for key, family in self._FAMILY_MAP.items():
            if key in name_lower:
                return family
        return "Other"

    def _log(
        self,
        run_id: str,
        agent: str,
        message: str,
        payload: Optional[Dict] = None,
    ):
        if self.db:
            try:
                self.db.log_trace(run_id, agent, message, payload=payload)
            except Exception:
                pass
