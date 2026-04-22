"""
MultiAgentOrchestrator — coordinates all agents to produce a validated DS plan.

Pipeline:
  1. Fetch ledger state from DB
  2. Lead Scientist proposes high-level plan
  3. EDA Agent adds preprocessing suggestions
  4. DNN Agent adds uncertainty methods (if DNN detected)
  5. Cost Estimator adds resource estimates
  6. LLM Judge validates the final plan
  7. (Phase 2) Optionally run autoresearch

Usage:
    from src.multi_agent_orchestrator import MultiAgentOrchestrator
    orch = MultiAgentOrchestrator(db=db, rag=rag)
    result = orch.run_pipeline("path/to/notebook.ipynb", "Improve F1 score")
"""

import os
from typing import Dict, Optional

from src.rag_system import RAGSystem
from src.agents import (
    LeadScientistAgent,
    EDAAgent,
    DNNAgent,
    CostEstimatorAgent,
    LLMJudge,
)


class MultiAgentOrchestrator:
    def __init__(
        self,
        db=None,
        rag: Optional[RAGSystem] = None,
        anthropic_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,  # kept for backward compat, ignored
        github_token: Optional[str] = None,
    ):
        self.db = db

        # Lazy-load RAG (FAISS model download can take time on first run)
        self._rag = rag
        self._rag_initialized = rag is not None

        # LLM setup
        self.llm = self._init_llm(anthropic_api_key)

        # Agents
        self.lead_scientist = LeadScientistAgent(self._get_rag(), self.llm, db)
        self.eda_agent = EDAAgent(self._get_rag(), self.llm)
        self.dnn_agent = DNNAgent(self._get_rag(), self.llm)
        self.cost_estimator = CostEstimatorAgent(self._get_rag(), self.llm)
        self.judge = LLMJudge(self._get_rag(), self.llm)

        # Phase 2
        self._github_token = github_token

    # ── Public API ─────────────────────────────────────────────────────────────

    def run_pipeline(
        self,
        file_path: str,
        goal: str = "Improve model performance",
        repo_name: Optional[str] = None,
        branch: str = "main",
        team_member: Optional[str] = None,
        execute_autoresearch: bool = False,
    ) -> Dict:
        # 1. Get ledger state
        ledger_state = self._get_ledger_state(file_path)

        # 2. Lead Scientist plan
        plan = self.lead_scientist.propose_plan(ledger_state, goal)

        # 3. EDA suggestions
        eda_result = self.eda_agent.suggest(ledger_state)
        plan["eda_suggestions"] = eda_result.get("suggestions", [])
        plan["eda_warnings"] = eda_result.get("warnings", [])

        # 4. DNN analysis
        dnn_result = self.dnn_agent.analyze_dnn(
            ledger_state.get("models", []),
            ledger_state.get("metrics", []),
        )
        if dnn_result.get("applicable"):
            plan["dnn_suggestions"] = dnn_result.get("suggestions", [])
            if dnn_result.get("uncertainty_methods"):
                plan["uncertainty_methods"] = dnn_result["uncertainty_methods"]

        # 5. Cost estimate
        plan["resource_estimate"] = self.cost_estimator.estimate_cost(plan)

        # 6. Judge validation
        validation = self.judge.validate_plan(plan)

        # 7. Optional autoresearch (Phase 2)
        autoresearch_result = None
        if execute_autoresearch and repo_name:
            autoresearch_result = self._run_autoresearch(
                repo_name=repo_name,
                file_path=file_path,
                branch=branch,
                goal=goal,
                team_member=team_member,
                plan=plan,
                ledger_state=ledger_state,
            )

        return {
            "plan": plan,
            "validation": validation,
            "ledger_state": ledger_state,
            "autoresearch_result": autoresearch_result,
        }

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _get_ledger_state(self, file_path: str) -> Dict:
        if self.db is None:
            return {"file_path": file_path, "models": [], "metrics": [], "preprocessing": []}
        entries = self.db.get_ledger_entries(file_path=file_path)
        if entries:
            entry = entries[0]
            return {
                "file_path": entry["file_path"],
                "models": entry.get("models", []) if isinstance(entry.get("models"), list) else [],
                "metrics": entry.get("metrics", []) if isinstance(entry.get("metrics"), list) else [],
                "preprocessing": entry.get("preprocessing", []) if isinstance(entry.get("preprocessing"), list) else [],
            }
        return {"file_path": file_path, "models": [], "metrics": [], "preprocessing": []}

    def _init_llm(self, api_key: Optional[str]):
        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            return None
        try:
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(model="claude-3-5-haiku-20241022", anthropic_api_key=key, temperature=0.2)
        except Exception:
            return None

    def _get_rag(self) -> Optional[RAGSystem]:
        if not self._rag_initialized:
            try:
                self._rag = RAGSystem()
                self._rag_initialized = True
            except Exception:
                pass
        return self._rag

    def _run_autoresearch(
        self,
        repo_name: str,
        file_path: str,
        branch: str,
        goal: str,
        team_member: Optional[str],
        plan: Dict,
        ledger_state: Dict,
    ) -> Dict:
        try:
            from src.autoresearch_wrapper import AutoResearchWrapper
            wrapper = AutoResearchWrapper(self.db, self._github_token)
            return wrapper.run_autoresearch(
                repo_name=repo_name,
                file_path=file_path,
                branch=branch,
                goal=goal,
                team_member=team_member,
                execute=True,
            )
        except Exception as e:
            return {"status": "error", "error": str(e)}
