"""
DeepEval adapter for the Agentic DS Ledger.

Uses the same AnthropicJudge pattern as the Data Privacy Auditor project:
  - FaithfulnessMetric:        Are the reported metrics consistent with worker evidence?
  - AnswerRelevancyMetric:     Does the champion selection address the optimisation goal?
  - ContextualRelevancyMetric: Did the agent use dataset/worker context appropriately?

All three metrics reuse the same DeepEvalBaseLLM subclass that wraps the
Anthropic Claude API — no OpenAI dependency required.

Timeout: 300 s (5 min) per API call.
"""

from __future__ import annotations

import asyncio
import json
import os

ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
TIMEOUT = 300  # seconds


# ── Anthropic judge (deferred import so deepeval is optional) ─────────────────

def _build_anthropic_judge(model: str = ANTHROPIC_MODEL, timeout: int = TIMEOUT):
    """Return an AnthropicJudge instance (DeepEvalBaseLLM subclass)."""
    from deepeval.models import DeepEvalBaseLLM
    import anthropic

    class AnthropicJudge(DeepEvalBaseLLM):
        def __init__(self):
            super().__init__()
            self._model_name = model
            self._client = anthropic.Anthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
                timeout=timeout,
            )

        def get_model_name(self) -> str:
            return self._model_name

        def load_model(self):
            return None

        def generate(self, prompt: str, schema=None) -> str:
            resp = self._client.messages.create(
                model=self._model_name,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            result = resp.content[0].text
            if schema is not None:
                try:
                    parsed = json.loads(result)
                    return schema(**parsed)
                except Exception:
                    return result
            return result

        async def a_generate(self, prompt: str, schema=None) -> str:
            return await asyncio.to_thread(self.generate, prompt, schema)

    return AnthropicJudge()


# ── Public API ─────────────────────────────────────────────────────────────────

def compute_deepeval_metrics(
    question: str,
    answer: str,
    contexts: list[str],
) -> dict:
    """
    Evaluate an agent run using three DeepEval LLM-based metrics.

    Args:
        question : The optimisation goal / prompt given to the agent.
        answer   : The agent's final answer / champion model description.
        contexts : List of context strings (worker results, iteration logs, etc.)

    Returns:
        {
          "faithfulness":        float | None,
          "answer_relevancy":    float | None,
          "context_relevancy":   float | None,
          "reasons":             {metric: reason_str},
          "passed":              bool,   # all three >= 0.5
        }
        On error: {"error": "message"}
    """
    try:
        from deepeval.test_case import LLMTestCase
        from deepeval.metrics import (
            FaithfulnessMetric,
            AnswerRelevancyMetric,
            ContextualRelevancyMetric,
        )
    except ImportError:
        raise ImportError("deepeval not installed — run: pip install deepeval")

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return {"error": "ANTHROPIC_API_KEY not set — DeepEval metrics require an API key."}

    try:
        judge = _build_anthropic_judge()
    except Exception as exc:
        return {"error": f"Failed to initialise Anthropic judge: {exc}"}

    faithfulness = FaithfulnessMetric(
        threshold=0.5, model=judge, include_reason=True, async_mode=False,
    )
    answer_relevancy = AnswerRelevancyMetric(
        threshold=0.5, model=judge, include_reason=True, async_mode=False,
    )
    context_relevancy = ContextualRelevancyMetric(
        threshold=0.5, model=judge, include_reason=True, async_mode=False,
    )

    ctx_list = contexts if isinstance(contexts, list) else [str(contexts)]
    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        retrieval_context=ctx_list,
    )

    scores: dict = {}
    reasons: dict = {}
    metrics_list = [
        ("faithfulness",      faithfulness),
        ("answer_relevancy",  answer_relevancy),
        ("context_relevancy", context_relevancy),
    ]
    for name, metric in metrics_list:
        try:
            metric.measure(test_case)
            scores[name] = round(float(metric.score), 4)
            if metric.reason:
                reasons[name] = metric.reason
        except Exception as exc:
            scores[name] = None
            reasons[name] = f"Error: {exc}"

    scores["reasons"] = reasons
    scores["passed"] = all(
        v is not None and v >= 0.5
        for k, v in scores.items()
        if k not in ("reasons", "passed")
    )
    return scores
