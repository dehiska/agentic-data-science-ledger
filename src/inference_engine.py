"""inference_engine.py — confidence-scored structured experiment extraction.

Takes the raw dict from MCPServer.parse_file() and produces a structured
experiment summary with per-item confidence scores and a task_type inference.
"""

from __future__ import annotations

import re
from typing import Any

# Known framework keywords that boost confidence
_FRAMEWORK_KEYWORDS = {
    "lgb", "xgb", "lgbm", "lightgbm", "xgboost",
    "sklearn", "skl", "torch", "tf", "keras",
    "catboost", "cb", "rf", "svm", "sgd",
}

_CLASSIFICATION_METRICS = {
    "accuracy", "f1", "roc_auc", "auc", "log_loss", "logloss",
    "precision", "recall", "f1_score", "fbeta", "average_precision",
    "balanced_accuracy", "matthews_corrcoef", "mcc",
}

_REGRESSION_METRICS = {
    "mse", "rmse", "mae", "r2", "r2_score", "mean_squared_error",
    "mean_absolute_error", "root_mean_squared_error", "explained_variance",
    "max_error", "mean_absolute_percentage_error", "mape",
}

_CLASSIFICATION_OBJECTIVES = {
    "binary", "multiclass", "softmax", "cross_entropy", "binary_crossentropy",
    "categorical_crossentropy", "binary:logistic", "multi:softmax", "multi:softprob",
}

_REGRESSION_OBJECTIVES = {
    "regression", "regression_l1", "regression_l2", "mse", "mae",
    "reg:squarederror", "reg:absoluteerror", "huber",
}

FAMILY_OPTIONS = [
    "Ensemble", "Linear", "Deep Neural Network", "Boosting",
    "Tree", "Bayesian", "Clustering", "Other",
]


class InferenceEngine:
    """Convert raw parse_file() output into a confidence-scored experiment dict."""

    def infer(self, parse_result: dict) -> dict:
        """
        Parameters
        ----------
        parse_result : dict
            Full return value of MCPServer.parse_file().

        Returns
        -------
        dict with keys: models, metrics, preprocessing, hyperparameters,
                         task_type, overall_confidence
        """
        file_type = parse_result.get("file_type", "ipynb")
        raw_text = parse_result.get("raw_text", "") or ""
        raw_models = parse_result.get("models", []) or []
        raw_metrics = parse_result.get("metrics", []) or []
        raw_preprocessing = parse_result.get("preprocessing", []) or []

        is_sidecar = (file_type == "json")

        models = self._score_models(raw_models, raw_text, is_sidecar)
        metrics = self._score_metrics(raw_metrics, is_sidecar)
        preprocessing = self._score_preprocessing(raw_preprocessing, is_sidecar)
        hyperparameters = self._extract_hyperparameters(models)
        task_type = self._infer_task_type(metrics, models)
        overall_confidence = self._overall_confidence(models, metrics, preprocessing)

        return {
            "models": models,
            "metrics": metrics,
            "preprocessing": preprocessing,
            "hyperparameters": hyperparameters,
            "task_type": task_type,
            "overall_confidence": overall_confidence,
        }

    # ── Models ─────────────────────────────────────────────────────────────────

    def _score_models(
        self, raw_models: list, raw_text: str, is_sidecar: bool
    ) -> list[dict]:
        seen: dict[str, dict] = {}  # name → best entry

        for m in raw_models:
            name = (m.get("name") or "").strip()
            if not name:
                continue

            confidence = self._model_base_confidence(m, is_sidecar)
            confidence += self._model_bonuses(m, name, raw_text)
            confidence = round(min(1.0, max(0.0, confidence)), 4)

            source = self._detect_source(m, is_sidecar)

            entry = {
                "name": name,
                "family": m.get("family", "Other"),
                "params": m.get("params") or {},
                "confidence": confidence,
                "source": source,
            }

            # Deduplicate: keep highest-confidence entry per name
            key = name.lower()
            if key not in seen or confidence > seen[key]["confidence"]:
                seen[key] = entry

        return list(seen.values())

    def _model_base_confidence(self, m: dict, is_sidecar: bool) -> float:
        if is_sidecar:
            return 1.0
        source = self._detect_source(m, False)
        mapping = {
            "ast": 0.95,
            "ast_functional": 0.85,
            "regex": 0.70,
            "framework_metric": 0.75,
        }
        return mapping.get(source, 0.70)

    def _model_bonuses(self, m: dict, name: str, raw_text: str) -> float:
        bonus = 0.0
        params = m.get("params") or {}
        if params:
            bonus += 0.03
        name_lower = name.lower()
        if any(kw in name_lower for kw in _FRAMEWORK_KEYWORDS):
            bonus += 0.02
        if name_lower in raw_text.lower():
            bonus += 0.02
        return bonus

    # ── Metrics ────────────────────────────────────────────────────────────────

    def _score_metrics(self, raw_metrics: list, is_sidecar: bool) -> list[dict]:
        seen: dict[str, dict] = {}

        for m in raw_metrics:
            name = (m.get("name") or "").strip()
            if not name:
                continue

            if is_sidecar:
                confidence = 1.0
            else:
                source = self._detect_source(m, False)
                confidence = 0.95 if source == "ast" else 0.75

            confidence = round(min(1.0, max(0.0, confidence)), 4)

            entry = {
                "name": name,
                "value": m.get("value"),
                "confidence": confidence,
                "source": self._detect_source(m, is_sidecar),
            }

            key = name.lower()
            if key not in seen or confidence > seen[key]["confidence"]:
                seen[key] = entry

        return list(seen.values())

    # ── Preprocessing ──────────────────────────────────────────────────────────

    def _score_preprocessing(self, raw_preprocessing: list, is_sidecar: bool) -> list[dict]:
        seen: dict[str, dict] = {}

        for p in raw_preprocessing:
            name = (p.get("name") or "").strip()
            if not name:
                continue

            if is_sidecar:
                confidence = 1.0
            else:
                source = self._detect_source(p, False)
                confidence = 0.95 if source == "ast" else 0.70

            confidence = round(min(1.0, max(0.0, confidence)), 4)

            entry = {
                "name": name,
                "type": p.get("type", "Unknown"),
                "confidence": confidence,
                "source": self._detect_source(p, is_sidecar),
            }

            key = name.lower()
            if key not in seen or confidence > seen[key]["confidence"]:
                seen[key] = entry

        return list(seen.values())

    # ── Hyperparameters ────────────────────────────────────────────────────────

    def _extract_hyperparameters(self, scored_models: list[dict]) -> dict:
        """Merge params dicts from all models; last-seen wins for duplicate keys."""
        merged: dict[str, Any] = {}
        for m in scored_models:
            params = m.get("params") or {}
            merged.update(params)
        return merged

    # ── Task type ─────────────────────────────────────────────────────────────

    def _infer_task_type(self, metrics: list[dict], models: list[dict]) -> str:
        metric_names = {m["name"].lower() for m in metrics}

        if metric_names & _CLASSIFICATION_METRICS:
            return "classification"
        if metric_names & _REGRESSION_METRICS:
            return "regression"

        # Check model params for objective/metric keys
        for model in models:
            params = model.get("params") or {}
            obj = str(params.get("objective", "")).lower()
            metric_val = str(params.get("metric", "")).lower()
            for val in (obj, metric_val):
                if val in _CLASSIFICATION_OBJECTIVES:
                    return "classification"
                if val in _REGRESSION_OBJECTIVES:
                    return "regression"

        return "unknown"

    # ── Overall confidence ─────────────────────────────────────────────────────

    def _overall_confidence(
        self,
        models: list[dict],
        metrics: list[dict],
        preprocessing: list[dict],
    ) -> float:
        all_items = models + metrics + preprocessing
        if not all_items:
            return 0.0
        total = sum(item["confidence"] for item in all_items)
        return round(total / len(all_items), 4)

    # ── Source detection ───────────────────────────────────────────────────────

    @staticmethod
    def _detect_source(item: dict, is_sidecar: bool) -> str:
        if is_sidecar:
            return "json_sidecar"
        code = (item.get("code") or "").lower()
        # Regex-detected items have code that looks like a raw string pattern
        if re.search(r"lgb\.train|xgb\.train|catboost\.train", code):
            return "ast_functional"
        # Items from _extract_source_patterns use "regex" in their code field
        if not item.get("line_number") and not item.get("cell_index"):
            return "regex"
        return "ast"
