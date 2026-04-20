"""
LLMJudge — validates DS plans against quality criteria.

Checks:
  - Plan completeness (has next_steps, suggested_models, rationale)
  - Scientific soundness (no contradictory advice)
  - Feasibility (cost vs benefit)
  - Ethics flags (no shortcuts that could cause data leakage, target leakage, etc.)
"""

from typing import Dict, List
from .base_agent import BaseAgent


_QUALITY_CRITERIA = [
    ("has_next_steps",     lambda p: bool(p.get("next_steps")),         "Plan must have next_steps"),
    ("has_rationale",      lambda p: bool(p.get("rationale")),          "Plan must have a rationale"),
    ("has_models",         lambda p: bool(p.get("suggested_models")),   "Plan should suggest at least one model"),
    ("cost_reasonable",    lambda p: p.get("resource_estimate", {}).get("total_cost", 0) < 100,
                                                                         "Estimated cost > $100 — verify before proceeding"),
]

_LEAKAGE_KEYWORDS = [
    "fit on test", "test set scaling", "future data", "leakage",
    "look-ahead", "train test split after", "scale before split",
]

_CONTRADICTION_PAIRS = [
    ("StandardScaler", "MinMaxScaler"),
    ("grid search", "random search"),
]


class LLMJudge(BaseAgent):
    def suggest(self, context: Dict) -> Dict:
        return self.validate_plan(context)

    def validate_plan(self, plan: Dict) -> Dict:
        issues = []
        suggestions = []
        warnings = []

        # ── Structural checks ──────────────────────────────────────────────────
        for criterion_id, check_fn, message in _QUALITY_CRITERIA:
            if not check_fn(plan):
                issues.append({"id": criterion_id, "message": message, "severity": "error"})

        # ── Leakage detection ──────────────────────────────────────────────────
        plan_text = str(plan).lower()
        for keyword in _LEAKAGE_KEYWORDS:
            if keyword in plan_text:
                warnings.append({
                    "type": "potential_data_leakage",
                    "message": f"Detected '{keyword}' — verify there is no train/test contamination.",
                    "severity": "warning",
                })

        # ── Evaluate next steps quality ────────────────────────────────────────
        next_steps = plan.get("next_steps", [])
        if len(next_steps) > 8:
            suggestions.append("Plan has many steps — consider prioritizing the top 3-5 most impactful.")
        if len(next_steps) == 1:
            suggestions.append("Plan has only one step — consider if more context is needed.")

        # ── Check uncertainty methods for DNNs ─────────────────────────────────
        models = plan.get("suggested_models", [])
        uncertainty = plan.get("uncertainty_methods", [])
        dnn_keywords = {"Sequential", "LSTM", "GRU", "TabularNet", "deep"}
        has_dnn = any(any(dk in m for dk in dnn_keywords) for m in models)
        if has_dnn and not uncertainty:
            suggestions.append(
                "DNN detected but no uncertainty quantification method proposed. Consider MC Dropout."
            )

        # ── LLM validation (if available) ─────────────────────────────────────
        if self.llm and not issues:
            llm_result = self._llm_validate(plan)
            if llm_result:
                issues.extend(llm_result.get("issues", []))
                suggestions.extend(llm_result.get("suggestions", []))
                warnings.extend(llm_result.get("warnings", []))

        is_valid = not any(i.get("severity") == "error" for i in issues)

        return {
            "is_valid": is_valid,
            "issues": issues,
            "warnings": warnings,
            "suggestions": suggestions,
            "score": self._compute_score(is_valid, issues, warnings),
            "verdict": "APPROVED" if is_valid and not warnings else ("APPROVED WITH WARNINGS" if is_valid else "REJECTED"),
        }

    # ── LLM validation ─────────────────────────────────────────────────────────

    def _llm_validate(self, plan: Dict) -> Dict:
        prompt = f"""You are a senior ML engineer reviewing a DS improvement plan.

Plan:
{plan}

Check for: data leakage risks, scientific soundness, feasibility, and missing steps.

Return JSON (no markdown):
{{
  "issues": [{{"id": "issue_id", "message": "...", "severity": "error|warning"}}],
  "suggestions": ["suggestion1", "suggestion2"],
  "warnings": [{{"type": "type", "message": "...", "severity": "warning"}}]
}}"""
        return self._llm_json(prompt)

    def _compute_score(self, is_valid: bool, issues: List, warnings: List) -> int:
        """Score 0-100 for plan quality."""
        score = 100
        score -= len([i for i in issues if i.get("severity") == "error"]) * 25
        score -= len([i for i in issues if i.get("severity") == "warning"]) * 10
        score -= len(warnings) * 5
        return max(0, score)
