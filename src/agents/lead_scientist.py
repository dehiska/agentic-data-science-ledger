"""
LeadScientistAgent — proposes a high-level DS improvement plan.

Given the current ledger state (models/metrics/preprocessing used so far)
and a goal string, it returns a structured plan with:
  - next_steps (list of action items)
  - suggested_models (list of model names to try)
  - suggested_search_strategy ("random" | "bayesian" | "grid")
  - rationale (plain text explanation)
  - uncertainty_methods (list if DNN detected)
"""

from typing import Dict, List
from .base_agent import BaseAgent


_RULE_BASED_SUGGESTIONS = {
    "Ensemble": [
        "Try XGBoost or LightGBM for potentially higher accuracy than Random Forest.",
        "Tune n_estimators (200-1000) and max_depth (3-8) with Bayesian optimization.",
        "Add feature importance analysis (SHAP or permutation importance).",
    ],
    "Linear": [
        "Consider tree-based models if non-linear patterns are suspected.",
        "Run VIF analysis to check for multicollinearity.",
        "Try polynomial features if the relationship looks non-linear in residual plots.",
    ],
    "Deep Neural Network": [
        "Add MC Dropout for uncertainty quantification (T=30 forward passes).",
        "Use learning rate scheduling (ReduceLROnPlateau) to improve convergence.",
        "Consider adding BatchNorm layers if training is unstable.",
    ],
    "Count Data": [
        "Verify overdispersion with a Poisson test (var/mean ratio).",
        "Consider Negative Binomial if variance >> mean.",
        "Add offset term if exposure variable differs across observations.",
    ],
    "Time Series": [
        "Ensure temporal train/val/test split to prevent data leakage.",
        "Add lag features and rolling statistics (mean, std, min, max over windows).",
        "Consider ARIMA baseline before switching to complex DL models.",
    ],
}


class LeadScientistAgent(BaseAgent):
    def __init__(self, rag_system=None, llm=None, db=None):
        super().__init__(rag_system, llm)
        self.db = db

    def suggest(self, context: Dict) -> Dict:
        return self.propose_plan(context, context.get("goal", "Improve model performance"))

    def propose_plan(self, ledger_state: Dict, goal: str) -> Dict:
        models = ledger_state.get("models", [])
        metrics = ledger_state.get("metrics", [])
        preprocessing = ledger_state.get("preprocessing", [])

        model_families = list({m.get("family", "Unknown") for m in models})
        metric_names = [m.get("name", "") for m in metrics]

        # Gather RAG context for the goal + current state
        rag_query = f"{goal} with models: {', '.join(model_families)} metrics: {', '.join(metric_names)}"
        rag_context = self._get_rag_context(rag_query, k=4)

        # Try LLM plan
        if self.llm:
            plan = self._llm_plan(ledger_state, goal, rag_context)
            if plan:
                return plan

        # Rule-based fallback
        return self._rule_based_plan(models, metrics, preprocessing, goal, model_families)

    # ── LLM path ───────────────────────────────────────────────────────────────

    def _llm_plan(self, ledger_state: Dict, goal: str, rag_context: List[str]) -> Dict:
        context_text = "\n---\n".join(rag_context[:3])
        prompt = f"""You are a Lead Data Scientist reviewing a notebook experiment.

Current state:
- Models used: {[m['name'] for m in ledger_state.get('models', [])]}
- Metrics computed: {[m['name'] for m in ledger_state.get('metrics', [])]}
- Preprocessing steps: {[p['name'] for p in ledger_state.get('preprocessing', [])]}

Goal: {goal}

Relevant best practices:
{context_text}

Return a JSON object (no markdown) with:
{{
  "next_steps": ["step1", "step2", "step3"],
  "suggested_models": ["ModelName1", "ModelName2"],
  "suggested_search_strategy": "bayesian",
  "rationale": "plain text explanation",
  "uncertainty_methods": ["MC Dropout"] or [],
  "priority": "high|medium|low"
}}"""
        return self._llm_json(prompt)

    # ── Rule-based path ────────────────────────────────────────────────────────

    def _rule_based_plan(
        self,
        models: List[Dict],
        metrics: List[Dict],
        preprocessing: List[Dict],
        goal: str,
        model_families: List[str],
    ) -> Dict:
        next_steps = []
        suggested_models = []
        uncertainty_methods = []

        # Gather family-specific suggestions
        for family in model_families:
            steps = _RULE_BASED_SUGGESTIONS.get(family, [])
            next_steps.extend(steps[:2])  # max 2 per family

        # Add generic steps if few models found
        if not models:
            next_steps = [
                "Start with a baseline model (LogisticRegression or RandomForestClassifier).",
                "Perform EDA: check class distribution, missing values, feature correlations.",
                "Split data with a stratified 80/10/10 train/val/test split.",
            ]
            suggested_models = ["RandomForestClassifier", "LogisticRegression"]
        else:
            # Suggest complementary models
            families_set = set(model_families)
            if "Linear" in families_set and "Ensemble" not in families_set:
                suggested_models.append("RandomForestClassifier")
            if "Ensemble" in families_set:
                suggested_models.append("XGBClassifier")
            if "Deep Neural Network" in families_set:
                uncertainty_methods = ["MC Dropout (T=30)", "Variance of predictions"]

        # Evaluate completeness
        metric_names = {m.get("name") for m in metrics}
        if not metric_names:
            next_steps.append("Add evaluation metrics: F1, AUC-ROC, PR-AUC at minimum.")
        if "f1" not in metric_names:
            next_steps.append("Add F1-score — critical for imbalanced datasets.")
        if "auc_roc" not in metric_names:
            next_steps.append("Add AUC-ROC for threshold-independent evaluation.")

        # Preprocessing gaps
        prep_types = {p.get("type") for p in preprocessing}
        if "Scaling" not in prep_types and any(f in model_families for f in ("Linear", "Deep Neural Network")):
            next_steps.append("Add feature scaling (StandardScaler) — required for Linear/DNN models.")

        # Decide search strategy
        if len(models) < 2:
            strategy = "random"
        elif "Bayesian" in goal or len(models) >= 3:
            strategy = "bayesian"
        else:
            strategy = "random"

        return {
            "next_steps": next_steps[:6],
            "suggested_models": suggested_models[:3],
            "suggested_search_strategy": strategy,
            "rationale": (
                f"Based on current experiment ({len(models)} model(s), {len(metrics)} metric(s)), "
                f"the primary goal '{goal}' can be addressed by: {', '.join(next_steps[:2]) if next_steps else 'establishing a proper baseline'}."
            ),
            "uncertainty_methods": uncertainty_methods,
            "priority": "high" if not models else "medium",
        }
