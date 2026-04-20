"""Tests for the multi-agent layer (no LLM required — rule-based paths only)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.agents import LeadScientistAgent, EDAAgent, DNNAgent, CostEstimatorAgent, LLMJudge


# ── Shared fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def empty_state():
    return {"file_path": "test.ipynb", "models": [], "metrics": [], "preprocessing": []}


@pytest.fixture
def ensemble_state():
    return {
        "file_path": "test.ipynb",
        "models": [{"name": "RandomForestClassifier", "family": "Ensemble", "params": {}}],
        "metrics": [{"name": "accuracy", "function": "accuracy_score", "line_number": 5}],
        "preprocessing": [{"name": "StandardScaler", "type": "Scaling"}],
    }


@pytest.fixture
def dnn_state():
    return {
        "file_path": "test.ipynb",
        "models": [{"name": "Sequential", "family": "Deep Neural Network", "params": {}}],
        "metrics": [{"name": "accuracy", "function": "accuracy_score", "line_number": 10}],
        "preprocessing": [],
    }


# ── LeadScientistAgent ─────────────────────────────────────────────────────────

class TestLeadScientistAgent:
    def test_returns_plan_structure(self, empty_state):
        agent = LeadScientistAgent()
        plan = agent.propose_plan(empty_state, "Improve F1")
        assert "next_steps" in plan
        assert "suggested_models" in plan
        assert "suggested_search_strategy" in plan
        assert "rationale" in plan

    def test_suggests_baseline_when_no_models(self, empty_state):
        agent = LeadScientistAgent()
        plan = agent.propose_plan(empty_state, "Build a classifier")
        assert len(plan["next_steps"]) > 0
        assert len(plan["suggested_models"]) > 0

    def test_suggests_xgboost_for_ensemble(self, ensemble_state):
        agent = LeadScientistAgent()
        plan = agent.propose_plan(ensemble_state, "Improve accuracy")
        assert "XGBClassifier" in plan["suggested_models"]

    def test_flags_missing_f1_metric(self, ensemble_state):
        agent = LeadScientistAgent()
        plan = agent.propose_plan(ensemble_state, "Improve model")
        steps_text = " ".join(plan["next_steps"]).lower()
        assert "f1" in steps_text

    def test_suggests_mc_dropout_for_dnn(self, dnn_state):
        agent = LeadScientistAgent()
        plan = agent.propose_plan(dnn_state, "Add uncertainty")
        assert len(plan["uncertainty_methods"]) > 0


# ── EDAAgent ───────────────────────────────────────────────────────────────────

class TestEDAAgent:
    def test_returns_eda_structure(self, empty_state):
        agent = EDAAgent()
        result = agent.suggest(empty_state)
        assert "suggestions" in result
        assert "warnings" in result
        assert result["agent"] == "EDAAgent"

    def test_warns_missing_scaling_for_linear(self):
        agent = EDAAgent()
        state = {
            "models": [{"name": "LogisticRegression", "family": "Linear"}],
            "metrics": [],
            "preprocessing": [],
        }
        result = agent.suggest(state)
        warning_msgs = " ".join(w["message"] for w in result["warnings"]).lower()
        assert "scal" in warning_msgs

    def test_warns_accuracy_only_no_f1(self):
        agent = EDAAgent()
        state = {
            "models": [],
            "metrics": [{"name": "accuracy", "function": "accuracy_score"}],
            "preprocessing": [],
        }
        result = agent.suggest(state)
        warning_msgs = " ".join(w["message"] for w in result["warnings"]).lower()
        assert "f1" in warning_msgs or "imbalance" in warning_msgs


# ── DNNAgent ───────────────────────────────────────────────────────────────────

class TestDNNAgent:
    def test_not_applicable_without_dnn(self, ensemble_state):
        agent = DNNAgent()
        result = agent.suggest(ensemble_state)
        assert result["applicable"] is False

    def test_applicable_with_dnn(self, dnn_state):
        agent = DNNAgent()
        result = agent.suggest(dnn_state)
        assert result["applicable"] is True
        assert len(result["suggestions"]) > 0
        assert len(result["uncertainty_methods"]) > 0

    def test_suggests_dropout(self, dnn_state):
        agent = DNNAgent()
        result = agent.suggest(dnn_state)
        suggestion_text = " ".join(s["action"] for s in result["suggestions"]).lower()
        assert "dropout" in suggestion_text


# ── CostEstimatorAgent ─────────────────────────────────────────────────────────

class TestCostEstimatorAgent:
    def test_default_estimate_no_models(self, empty_state):
        agent = CostEstimatorAgent()
        est = agent.estimate_cost(empty_state)
        assert "gcp_instance" in est
        assert est["total_cost"] > 0

    def test_dnn_gets_gpu_instance(self):
        agent = CostEstimatorAgent()
        plan = {
            "models": [{"name": "Sequential", "family": "Deep Neural Network"}],
            "suggested_models": [],
            "suggested_search_strategy": "random",
        }
        est = agent.estimate_cost(plan)
        assert est["gpu"] is not None

    def test_bayesian_costs_more_than_random(self):
        agent = CostEstimatorAgent()
        base = {"models": [{"family": "Ensemble"}], "suggested_models": [], "suggested_search_strategy": "random"}
        bayesian = {**base, "suggested_search_strategy": "bayesian"}
        assert agent.estimate_cost(bayesian)["total_cost"] > agent.estimate_cost(base)["total_cost"]


# ── LLMJudge ───────────────────────────────────────────────────────────────────

class TestLLMJudge:
    def test_approves_complete_plan(self):
        judge = LLMJudge()
        plan = {
            "next_steps": ["Try XGBoost", "Add F1 metric"],
            "rationale": "Current baseline is weak.",
            "suggested_models": ["XGBClassifier"],
            "resource_estimate": {"total_cost": 0.50},
            "uncertainty_methods": [],
        }
        result = judge.validate_plan(plan)
        assert result["is_valid"] is True
        assert result["score"] > 0

    def test_rejects_plan_missing_next_steps(self):
        judge = LLMJudge()
        plan = {
            "next_steps": [],
            "rationale": "",
            "suggested_models": [],
            "resource_estimate": {"total_cost": 0.10},
        }
        result = judge.validate_plan(plan)
        assert result["is_valid"] is False

    def test_high_cost_triggers_warning(self):
        judge = LLMJudge()
        plan = {
            "next_steps": ["step"],
            "rationale": "rationale",
            "suggested_models": ["Model"],
            "resource_estimate": {"total_cost": 200.0},
        }
        result = judge.validate_plan(plan)
        assert result["is_valid"] is False  # cost > $100 → error


# ── MultiAgentOrchestrator ─────────────────────────────────────────────────────

class TestOrchestrator:
    def test_full_pipeline(self):
        import tempfile
        from src.database import LocalDatabase
        from src.multi_agent_orchestrator import MultiAgentOrchestrator

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        db = LocalDatabase(db_path)
        orch = MultiAgentOrchestrator(db=db)

        try:
            result = orch.run_pipeline("notebooks/exp.ipynb", "Improve F1")
            assert "plan" in result
            assert "validation" in result
            assert "ledger_state" in result
            assert result["validation"]["is_valid"] in (True, False)
        finally:
            db.close()
            os.unlink(db_path)
