"""
BaseAgent — abstract base class for all multi-agent workers.

Every agent has access to the RAG system and optionally an LLM.
Agents should implement `suggest(context)` and/or their specific method.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseAgent(ABC):
    def __init__(self, rag_system=None, llm=None):
        self.rag = rag_system
        self.llm = llm

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _get_rag_context(self, query: str, k: int = 3) -> List[str]:
        if self.rag is not None:
            try:
                return self.rag.retrieve_context(query, k=k)
            except Exception:
                pass
        return []

    def _call_llm(self, prompt: str) -> str:
        """Call LLM and return string response. Returns empty string if no LLM."""
        if self.llm is None:
            return ""
        try:
            result = self.llm.invoke(prompt)
            # Handle both str and AIMessage returns
            if hasattr(result, "content"):
                return result.content
            return str(result)
        except Exception as e:
            return f"[LLM error: {e}]"

    def _llm_json(self, prompt: str) -> Optional[Dict]:
        """Call LLM, parse JSON response. Returns None on failure."""
        import json
        raw = self._call_llm(prompt)
        if not raw:
            return None
        # Strip markdown code fences if present
        raw = raw.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1])
        try:
            return json.loads(raw)
        except Exception:
            return None

    @abstractmethod
    def suggest(self, context: Dict) -> Dict:
        """Return a suggestions dict given a context dict."""
