"""
EDAAgent — suggests EDA and preprocessing improvements.

Analyzes the current ledger state and returns actionable EDA suggestions
focusing on data quality, feature engineering, and distribution analysis.
"""

from typing import Dict, List
from .base_agent import BaseAgent


class EDAAgent(BaseAgent):
    def suggest(self, context: Dict) -> Dict:
        """Analyze ledger state and return EDA suggestions."""
        models = context.get("models", [])
        preprocessing = context.get("preprocessing", [])
        metrics = context.get("metrics", [])

        suggestions = []
        warnings = []

        prep_types = {p.get("type", "") for p in preprocessing}
        prep_names = {p.get("name", "") for p in preprocessing}
        model_families = {m.get("family", "") for m in models}

        # ── Missing value handling ──────────────────────────────────────────────
        if "Imputation" not in prep_types:
            suggestions.append({
                "category": "Missing Values",
                "action": "Add imputation step (SimpleImputer or KNNImputer). Check df.isnull().sum() first.",
                "priority": "high",
                "code_hint": "from sklearn.impute import SimpleImputer\nimp = SimpleImputer(strategy='median')\nX = imp.fit_transform(X)",
            })

        # ── Scaling ────────────────────────────────────────────────────────────
        needs_scaling = model_families & {"Linear", "Deep Neural Network", "SVM", "KNN"}
        if needs_scaling and "Scaling" not in prep_types:
            warnings.append({
                "category": "Missing Scaling",
                "message": f"Models {needs_scaling} require feature scaling. Add StandardScaler.",
                "severity": "high",
            })

        # ── Encoding ───────────────────────────────────────────────────────────
        if "Encoding" not in prep_types:
            suggestions.append({
                "category": "Categorical Encoding",
                "action": "Check for categorical columns. Use TargetEncoder for high-cardinality, OrdinalEncoder for ordinal.",
                "priority": "medium",
                "code_hint": "cat_cols = df.select_dtypes('object').columns\nprint(df[cat_cols].nunique())",
            })

        # ── Feature selection ──────────────────────────────────────────────────
        if "Feature Selection" not in prep_types:
            suggestions.append({
                "category": "Feature Selection",
                "action": "Check feature importance or correlation. Remove features with >95% missing or zero variance.",
                "priority": "medium",
                "code_hint": "from sklearn.feature_selection import mutual_info_classif\nscores = mutual_info_classif(X, y)\nprint(dict(zip(feature_names, scores)))",
            })

        # ── Imbalance check ────────────────────────────────────────────────────
        metric_names = {m.get("name", "") for m in metrics}
        if "accuracy" in metric_names and "f1" not in metric_names:
            warnings.append({
                "category": "Potential Class Imbalance",
                "message": "Only accuracy reported — may be misleading if classes are imbalanced. Add F1, PR-AUC.",
                "severity": "medium",
            })

        # ── Dimensionality reduction ───────────────────────────────────────────
        if "Dimensionality Reduction" not in prep_types:
            suggestions.append({
                "category": "High Dimensionality Check",
                "action": "If n_features > 50, consider PCA or feature selection to reduce noise.",
                "priority": "low",
                "code_hint": "print(f'Feature count: {X.shape[1]}')\n# If > 50: pca = PCA(n_components=0.95).fit_transform(X_scaled)",
            })

        # ── RAG-enhanced suggestions ───────────────────────────────────────────
        rag_hits = self._get_rag_context("EDA preprocessing data quality feature engineering", k=2)

        return {
            "agent": "EDAAgent",
            "suggestions": suggestions,
            "warnings": warnings,
            "rag_context": rag_hits,
            "summary": f"{len(suggestions)} EDA suggestions, {len(warnings)} warnings found.",
        }
