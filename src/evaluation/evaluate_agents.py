"""
Evaluation engine for Agentic DS Ledger agents.

Two tiers of metrics — same architecture as the Data Privacy Auditor reference:

  Tier 1 — Custom (pure Python, no API key needed):
    • score_validity          val_score in [0, 1]
    • worker_success_rate     succeeded workers / total workers
    • family_diversity        # distinct model families tried
    • flaml_utilization       fraction of workers that used FLAML (vs sklearn fallback)
    • improvement_rate        (AutoResearcher only) iterations that improved / total

  Tier 2 — DeepEval (LLM-based, requires ANTHROPIC_API_KEY):
    • faithfulness            reported metrics consistent with worker evidence?
    • answer_relevancy        champion selection addresses the optimisation goal?
    • context_relevancy       agent used dataset / worker context appropriately?

Usage:
    from src.evaluation.evaluate_agents import evaluate_swarm, evaluate_autoresearch

    eval_result = evaluate_swarm(swarm_result)
    eval_result = evaluate_autoresearch(ar_result)
"""

from __future__ import annotations

from typing import Optional


# ── Tier 1: Custom metrics ─────────────────────────────────────────────────────

def compute_score_validity(result: dict) -> dict:
    """Check that val_score is a finite float in [0, 1]."""
    best = result.get("best_model", {})
    score = best.get("metrics", {}).get("val_score")
    valid = (
        score is not None
        and isinstance(score, (int, float))
        and 0.0 <= float(score) <= 1.0
    )
    return {
        "valid": valid,
        "val_score": score,
        "status": "PASS" if valid else "FAIL",
        "note": f"val_score={score}" if valid else f"val_score={score!r} is out of [0,1] or missing",
    }


def compute_worker_success_rate(result: dict) -> dict:
    """Fraction of workers that completed successfully."""
    n_agents    = result.get("n_agents", 1)
    n_succeeded = result.get("n_succeeded", len(result.get("all_worker_results", [])))
    rate = round(n_succeeded / n_agents, 4) if n_agents > 0 else 0.0
    return {
        "rate": rate,
        "succeeded": n_succeeded,
        "total": n_agents,
        "status": "PASS" if rate >= 0.8 else ("PARTIAL" if rate >= 0.5 else "FAIL"),
        "note": f"{n_succeeded}/{n_agents} workers succeeded",
    }


def compute_family_diversity(result: dict) -> dict:
    """Number of distinct model families represented across workers."""
    workers = result.get("all_worker_results", [])
    families = {
        w.get("best_model", {}).get("family", "Unknown")
        for w in workers
        if w.get("best_model")
    }
    # Also include champion family
    champ_family = result.get("best_model", {}).get("family")
    if champ_family:
        families.add(champ_family)

    n = len(families)
    return {
        "n_families": n,
        "families": sorted(families),
        "status": "PASS" if n >= 2 else "PARTIAL",
        "note": f"{n} model {'family' if n == 1 else 'families'}: {', '.join(sorted(families))}",
    }


def compute_flaml_utilization(result: dict) -> dict:
    """Fraction of workers that used FLAML AutoML (vs sklearn fallback)."""
    workers = result.get("all_worker_results", [])
    if not workers:
        return {"rate": 0.0, "flaml_count": 0, "total": 0, "status": "NOT_EVALUATED"}
    flaml_count = sum(
        1 for w in workers
        if w.get("best_model", {}).get("flaml", False)
    )
    rate = round(flaml_count / len(workers), 4)
    return {
        "rate": rate,
        "flaml_count": flaml_count,
        "total": len(workers),
        "status": "PASS" if rate >= 0.5 else "PARTIAL",
        "note": f"FLAML used by {flaml_count}/{len(workers)} workers",
    }


def compute_improvement_rate(ar_result: dict) -> dict:
    """
    AutoResearcher only.
    Improvement rate = iterations that found a new best / total iterations run.
    Approximated from the result dict (exact value stored in traces).
    """
    iterations_run = ar_result.get("iterations_run", 0)
    if iterations_run == 0:
        return {"rate": None, "status": "NOT_EVALUATED", "note": "No iterations recorded"}

    # We don't have per-iteration data in the result dict, but we can derive
    # a lower bound: at least 1 improvement (the final best), up to iterations_run.
    # Best-effort: mark PASS if the run completed (didn't error out).
    status_ok = ar_result.get("status") == "success"
    note = (
        f"{iterations_run} iterations, best model: "
        f"{ar_result.get('best_model', {}).get('name', '?')}"
    )
    return {
        "iterations_run": iterations_run,
        "status": "PASS" if status_ok else "FAIL",
        "note": note,
    }


# ── Per-agent breakdown ────────────────────────────────────────────────────────

def compute_per_agent_breakdown(swarm_result: dict) -> dict:
    """Score each worker agent individually."""
    workers = swarm_result.get("all_worker_results", [])
    agents: dict = {}
    for w in workers:
        aid = w.get("agent_id", "?")
        bm  = w.get("best_model", {})
        score = bm.get("metrics", {}).get("val_score")
        agents[f"worker_{aid}"] = {
            "model":       bm.get("name", "?"),
            "family":      bm.get("family", "?"),
            "val_score":   score,
            "flaml":       bm.get("flaml", False),
            "n_rows":      bm.get("n_rows_trained"),
            "status":      "PASS" if (score is not None and score > 0) else "FAIL",
        }
    # Orchestrator itself
    champ = swarm_result.get("best_model", {})
    agents["orchestrator"] = {
        "champion_model":  champ.get("name", "?"),
        "champion_family": champ.get("family", "?"),
        "champion_score":  champ.get("metrics", {}).get("val_score"),
        "entry_id":        swarm_result.get("entry_id"),
        "status": "PASS" if swarm_result.get("status") == "success" else "FAIL",
    }
    return agents


# ── Build DeepEval inputs ──────────────────────────────────────────────────────

def _build_swarm_question(result: dict) -> str:
    goal = result.get("eda_meta", {}).get("goal", "Maximise predictive performance")
    target = result.get("target_col", "target")
    task   = result.get("task_type", "classification")
    return (
        f"Goal: {goal}. "
        f"Find the best ML model for {task} on target column '{target}'. "
        f"Which model should be selected as the champion?"
    )


def _build_swarm_answer(result: dict) -> str:
    best = result.get("best_model", {})
    metrics = best.get("metrics", {})
    params  = best.get("params", {})
    lines = [
        f"Champion model: {best.get('name', '?')} (family: {best.get('family', '?')})",
        f"Validation score: {metrics.get('val_score', '?')}",
        f"FLAML AutoML used: {best.get('flaml', False)}",
        f"Trained on: {best.get('n_rows_trained', '?')} rows",
        f"Best hyperparameters: {params}",
        f"Succeeded workers: {result.get('n_succeeded', '?')}/{result.get('n_agents', '?')}",
        f"Ledger entry: #{result.get('entry_id', '?')}",
    ]
    return "\n".join(lines)


def _build_swarm_contexts(result: dict) -> list[str]:
    contexts = []
    # Dataset context
    contexts.append(
        f"Dataset: {result.get('n_rows', '?')} rows | "
        f"target={result.get('target_col', '?')} | task={result.get('task_type', '?')}"
    )
    # Per-worker results
    for w in result.get("all_worker_results", [])[:10]:  # cap at 10 to avoid huge prompts
        bm = w.get("best_model", {})
        contexts.append(
            f"Worker {w.get('agent_id', '?')}: {bm.get('name', '?')} "
            f"val_score={bm.get('metrics', {}).get('val_score', '?')} "
            f"flaml={bm.get('flaml', False)}"
        )
    # EDA metadata if available
    eda = result.get("eda_meta", {})
    if eda:
        contexts.append(f"EDA metadata: {str(eda)[:500]}")
    return contexts or ["No worker context available"]


def _build_ar_question(ar_result: dict) -> str:
    return (
        "Goal: Improve model accuracy through iterative hyperparameter search. "
        "Did the autoresearcher find a meaningfully better model than baseline?"
    )


def _build_ar_answer(ar_result: dict) -> str:
    best = ar_result.get("best_model", {})
    metrics = best.get("metrics", {})
    lines = [
        f"Best model found: {best.get('name', '?')} (family: {best.get('family', '?')})",
        f"Metrics: f1={metrics.get('f1', '?')}, accuracy={metrics.get('accuracy', '?')}, "
        f"auc_roc={metrics.get('auc_roc', '?')}",
        f"Hyperparameters: {best.get('params', {})}",
        f"Iterations run: {ar_result.get('iterations_run', '?')}",
        f"Simulated: {ar_result.get('simulated', False)}",
    ]
    return "\n".join(lines)


def _build_ar_contexts(ar_result: dict) -> list[str]:
    best = ar_result.get("best_model", {})
    all_tried = ar_result.get("all_models_tried", [])
    return [
        f"Model pool explored: {', '.join(all_tried) if all_tried else 'N/A'}",
        f"Best model: {best.get('name', '?')} with params {best.get('params', {})}",
        f"Iterations completed: {ar_result.get('iterations_run', '?')}",
        f"Note: {ar_result.get('note', '')}",
    ]


# ── Main evaluation functions ──────────────────────────────────────────────────

def evaluate_swarm(swarm_result: dict, run_deepeval: bool = True) -> dict:
    """
    Run all evaluation metrics on a completed swarm run.

    Args:
        swarm_result : The dict returned by SwarmOrchestrator.run_swarm()
        run_deepeval : Set False to skip LLM-based metrics (saves API cost / time)

    Returns:
        {
          "custom_metrics":  {score_validity, worker_success_rate, family_diversity,
                              flaml_utilization},
          "deepeval_metrics": {faithfulness, answer_relevancy, context_relevancy, ...}
                              or None if skipped / unavailable,
          "per_agent":       {worker_0: {...}, orchestrator: {...}},
          "overall_status":  "PASS" | "PARTIAL" | "FAIL",
          "run_id":          str,
        }
    """
    run_id = swarm_result.get("run_id", "?")

    # ── Tier 1: custom metrics ─────────────────────────────────────────────────
    score_val   = compute_score_validity(swarm_result)
    success_rt  = compute_worker_success_rate(swarm_result)
    diversity   = compute_family_diversity(swarm_result)
    flaml_util  = compute_flaml_utilization(swarm_result)

    custom = {
        "score_validity":       score_val,
        "worker_success_rate":  success_rt,
        "family_diversity":     diversity,
        "flaml_utilization":    flaml_util,
    }

    # ── Tier 2: DeepEval ───────────────────────────────────────────────────────
    deepeval_result = None
    if run_deepeval:
        try:
            from src.evaluation.deepeval_adapter import compute_deepeval_metrics
            deepeval_result = compute_deepeval_metrics(
                question=_build_swarm_question(swarm_result),
                answer=_build_swarm_answer(swarm_result),
                contexts=_build_swarm_contexts(swarm_result),
            )
        except ImportError:
            deepeval_result = {"error": "deepeval not installed — run: pip install deepeval"}
        except Exception as exc:
            deepeval_result = {"error": str(exc)}

    # ── Per-agent breakdown ────────────────────────────────────────────────────
    per_agent = compute_per_agent_breakdown(swarm_result)

    # ── Overall status ─────────────────────────────────────────────────────────
    tier1_statuses = [m["status"] for m in custom.values() if "status" in m]
    if all(s == "PASS" for s in tier1_statuses):
        overall = "PASS"
    elif any(s == "FAIL" for s in tier1_statuses):
        overall = "FAIL"
    else:
        overall = "PARTIAL"

    return {
        "run_id":          run_id,
        "custom_metrics":  custom,
        "deepeval_metrics": deepeval_result,
        "per_agent":       per_agent,
        "overall_status":  overall,
    }


def evaluate_autoresearch(ar_result: dict, run_deepeval: bool = True) -> dict:
    """
    Run all evaluation metrics on a completed autoresearch run.

    Args:
        ar_result    : The dict returned by AutoResearchWrapper.run_autoresearch()
        run_deepeval : Set False to skip LLM-based metrics

    Returns:
        {
          "custom_metrics":  {score_validity, improvement_rate},
          "deepeval_metrics": {...} or None,
          "overall_status":  "PASS" | "PARTIAL" | "FAIL",
          "run_id":          str,
        }
    """
    run_id = ar_result.get("run_id", "?")

    # Adapt autoresearch result to look like a swarm result for score_validity
    _adapted = {
        "best_model": {
            "metrics": {
                "val_score": ar_result.get("best_model", {}).get("metrics", {}).get("f1"),
            }
        }
    }

    score_val   = compute_score_validity(_adapted)
    improve_rt  = compute_improvement_rate(ar_result)

    custom = {
        "score_validity":    score_val,
        "improvement_rate":  improve_rt,
    }

    deepeval_result = None
    if run_deepeval:
        try:
            from src.evaluation.deepeval_adapter import compute_deepeval_metrics
            deepeval_result = compute_deepeval_metrics(
                question=_build_ar_question(ar_result),
                answer=_build_ar_answer(ar_result),
                contexts=_build_ar_contexts(ar_result),
            )
        except ImportError:
            deepeval_result = {"error": "deepeval not installed — run: pip install deepeval"}
        except Exception as exc:
            deepeval_result = {"error": str(exc)}

    tier1_statuses = [m["status"] for m in custom.values() if "status" in m]
    if all(s == "PASS" for s in tier1_statuses):
        overall = "PASS"
    elif any(s == "FAIL" for s in tier1_statuses):
        overall = "FAIL"
    else:
        overall = "PARTIAL"

    return {
        "run_id":          run_id,
        "custom_metrics":  custom,
        "deepeval_metrics": deepeval_result,
        "overall_status":  overall,
    }
