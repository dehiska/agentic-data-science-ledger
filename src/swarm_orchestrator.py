"""
SwarmOrchestrator — divide-and-conquer AutoML using a swarm of agents.

Architecture (tree model):
  Root  : SwarmOrchestrator — owns the dataset + EDA metadata
  Branch: one SwarmWorkerAgent per model family (determined by FLAML)
  Leaf  : each worker trains on a stratified chunk, returns best params

Pipeline:
  1. Parse EDA file (any supported format) → extract target_col, task_type
  2. Load CSV with memory-optimised dtypes
  3. n_agents = ceil(n_rows / max_rows_per_agent)
  4. Split dataset into n stratified chunks
  5. Run each SwarmWorkerAgent in a ThreadPoolExecutor
     (each worker runs FLAML AutoML on its chunk for time_budget seconds)
  6. Aggregate: best per model family → global champion
  7. Store champion in ledger + log all steps to traces table

Usage:
    from src.swarm_orchestrator import SwarmOrchestrator
    orch = SwarmOrchestrator(db=db)
    result = orch.run_swarm(
        csv_path="train.csv",
        eda_file="notebooks/eda.ipynb",
        target_col="cancel",
        task_type="classification",
        time_budget_per_agent=60,
        max_rows_per_agent=150_000,
    )
"""

import math
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _oprint(msg: str) -> None:
    """Formatted print for the swarm orchestrator."""
    print(f"[{_ts()}] [Swarm] {msg}", flush=True)

# FLAML / XGBoost / LightGBM are not thread-safe when run concurrently in the
# same process (they share internal thread-pool state).  This lock ensures only
# one FLAML fit runs at a time; the ThreadPoolExecutor still parallelises CSV
# reading and feature prep.
_FLAML_LOCK = threading.Lock()


class SwarmOrchestrator:
    # These can be overridden via environment variables for Cloud Run tuning
    DEFAULT_MAX_ROWS = int(os.getenv("MAX_ROWS_PER_AGENT", "150000"))
    DEFAULT_TIME_BUDGET = int(os.getenv("TIME_BUDGET_PER_AGENT", "60"))  # seconds

    def __init__(self, db=None):
        self.db = db

    # ── Public API ─────────────────────────────────────────────────────────────

    def run_swarm(
        self,
        csv_path: str,
        eda_file: Optional[str] = None,
        target_col: Optional[str] = None,
        task_type: Optional[str] = None,
        time_budget_per_agent: int = DEFAULT_TIME_BUDGET,
        max_rows_per_agent: int = DEFAULT_MAX_ROWS,
        max_parallel_agents: int = 4,
        project_id: Optional[int] = None,
        team_member: Optional[str] = None,
        goal: str = "Maximize F1",
        run_id: Optional[str] = None,
    ) -> Dict:
        run_id = run_id or uuid.uuid4().hex[:8]

        print(f"\n{'='*64}", flush=True)
        print(f"[{_ts()}]  SWARM ORCHESTRATOR  |  run_id={run_id}", flush=True)
        print(f"{'='*64}", flush=True)
        _oprint(f"CSV: {Path(csv_path).name}  |  goal: {goal}")

        # ── 1. Parse EDA file ──────────────────────────────────────────────────
        eda_meta: Dict = {}
        if eda_file and Path(eda_file).exists():
            _oprint(f"Parsing EDA file: {Path(eda_file).name}")
            eda_meta = self._parse_eda(eda_file, run_id)
        elif eda_file:
            _oprint(f"EDA file not found: {eda_file} — skipping")
            self._log(run_id, "swarm_orchestrator",
                      f"EDA file not found: {eda_file} — skipping EDA parse")

        # ── 2. Resolve target_col / task_type ──────────────────────────────────
        target_col = target_col or eda_meta.get("target_col")
        task_type  = task_type  or eda_meta.get("task_type", "classification")

        _oprint(f"Target column: {target_col!r}  |  Task: {task_type}")
        self._log(run_id, "swarm_orchestrator",
                  f"Swarm AutoML starting — csv={Path(csv_path).name}, "
                  f"target={target_col}, task={task_type}, "
                  f"max_rows_per_agent={max_rows_per_agent:,}, "
                  f"time_budget={time_budget_per_agent}s",
                  payload={
                      "csv_path": csv_path, "target_col": target_col,
                      "task_type": task_type, "max_rows_per_agent": max_rows_per_agent,
                      "time_budget_per_agent": time_budget_per_agent, "goal": goal,
                  })

        # ── 3. Load dataset ─────────────────────────────────────────────────────
        _oprint(f"Loading dataset...")
        try:
            df = self._load_csv(csv_path, run_id)
        except Exception as e:
            _oprint(f"ERROR loading CSV: {e}")
            return {"status": "error", "error": f"Failed to load CSV: {e}", "run_id": run_id}
        _oprint(f"Dataset loaded: {len(df):,} rows x {len(df.columns)} cols")

        # Final target-column resolution (fall back to last col if still None)
        if target_col is None:
            target_col = self._guess_target_col(df.columns.tolist())
            self._log(run_id, "swarm_orchestrator",
                      f"No target_col specified — guessing: {target_col}")

        if target_col not in df.columns:
            return {
                "status": "error",
                "error": (
                    f"Target column '{target_col}' not found in CSV. "
                    f"Available: {df.columns.tolist()[:15]}"
                ),
                "run_id": run_id,
            }

        # Drop rows where target is null (can't train on them)
        pre_len = len(df)
        df = df.dropna(subset=[target_col]).reset_index(drop=True)
        if len(df) < pre_len:
            self._log(run_id, "swarm_orchestrator",
                      f"Dropped {pre_len - len(df):,} rows with null target")

        # ── 4. Determine swarm topology ─────────────────────────────────────────
        n_rows  = len(df)
        n_agents = max(1, math.ceil(n_rows / max_rows_per_agent))
        actual_workers = min(n_agents, max_parallel_agents, os.cpu_count() or 2)

        rows_per_chunk = math.ceil(n_rows / n_agents)
        _oprint(
            f"Topology: {n_rows:,} rows -> {n_agents} agents x ~{rows_per_chunk:,} rows"
            f" | {actual_workers} parallel workers | budget={time_budget_per_agent}s/agent"
        )
        self._log(run_id, "swarm_orchestrator",
                  f"Swarm topology: {n_rows:,} rows → {n_agents} agents × "
                  f"~{rows_per_chunk:,} rows | {actual_workers} parallel workers",
                  payload={
                      "n_rows": n_rows, "n_agents": n_agents,
                      "n_cols": len(df.columns), "actual_workers": actual_workers,
                  })

        # ── 5. Split into stratified chunks ─────────────────────────────────────
        _oprint(f"Splitting into {n_agents} stratified chunks...")
        chunks = self._split_stratified(df, target_col, task_type, n_agents, run_id)

        # ── 6. Run swarm workers in parallel ────────────────────────────────────
        _oprint(f"Launching {n_agents} workers ({actual_workers} parallel)...")
        worker_results = self._run_workers(
            chunks=chunks,
            target_col=target_col,
            task_type=task_type,
            time_budget=time_budget_per_agent,
            max_workers=actual_workers,
            run_id=run_id,
        )

        if not worker_results:
            _oprint("ERROR: All workers failed — check traces for details.")
            return {
                "status": "error",
                "error": "All swarm workers failed — check traces for details.",
                "run_id": run_id,
            }

        _oprint(f"{len(worker_results)}/{n_agents} workers succeeded")

        # ── 7. Aggregate (divide-and-conquer tree merge) ────────────────────────
        _oprint("Running two-level tournament (family -> global)...")
        best_global = self._aggregate(worker_results, run_id)

        self._log(run_id, "swarm_orchestrator",
                  f"Swarm complete -- champion: {best_global['name']} "
                  f"val_score={best_global['metrics'].get('val_score', '?'):.4f} "
                  f"({len(worker_results)}/{n_agents} agents succeeded)",
                  payload={
                      "champion": best_global["name"],
                      "val_score": best_global["metrics"].get("val_score"),
                      "n_succeeded": len(worker_results),
                      "flaml_used": best_global.get("flaml", False),
                  })

        # ── 8. Store champion in ledger ──────────────────────────────────────────
        entry_id = self._store_in_ledger(
            csv_path=csv_path,
            best_model=best_global,
            target_col=target_col,
            task_type=task_type,
            n_rows=n_rows,
            run_id=run_id,
            project_id=project_id,
            team_member=team_member,
        )

        swarm_result = {
            "status": "success",
            "run_id": run_id,
            "best_model": best_global,
            "all_worker_results": worker_results,
            "n_agents": n_agents,
            "n_succeeded": len(worker_results),
            "n_rows": n_rows,
            "target_col": target_col,
            "task_type": task_type,
            "entry_id": entry_id,
            "eda_meta": eda_meta,
            "flaml_available": best_global.get("flaml", False),
        }

        # ── 9. Run evaluation + print final summary ──────────────────────────────
        eval_result = self._evaluate_and_print(swarm_result, run_id)
        swarm_result["eval"] = eval_result

        return swarm_result

    # ── EDA parsing ────────────────────────────────────────────────────────────

    def _parse_eda(self, eda_file: str, run_id: str) -> Dict:
        """Parse EDA file (any supported format) to extract target_col + task_type."""
        self._log(run_id, "swarm_orchestrator",
                  f"Parsing EDA file: {Path(eda_file).name}")
        try:
            from src.mcp_server import MCPServer
            meta = MCPServer().parse_file(eda_file)
            raw = meta.get("raw_text", "")

            # For CSV files the EDA parser itself surfaces dataset_info
            if meta.get("dataset_info"):
                info = meta["dataset_info"]
                result = {
                    "target_col": info.get("target_col_guess"),
                    "task_type":  info.get("task_type_guess", "classification"),
                }
            else:
                result = {
                    "target_col": self._extract_target_col(raw),
                    "task_type":  self._infer_task_type(raw, meta),
                }

            self._log(run_id, "swarm_orchestrator",
                      f"EDA parsed — target={result['target_col']}, "
                      f"task={result['task_type']}",
                      payload=result)
            return result
        except Exception as e:
            self._log(run_id, "swarm_orchestrator",
                      f"EDA parse warning (continuing without): {e}")
            return {}

    def _extract_target_col(self, source: str) -> Optional[str]:
        """Scan source code / document text for target column assignments."""
        import re
        patterns = [
            r"y\s*=\s*df\s*\[\s*['\"](\w+)['\"]\s*\]",           # y = df['col']
            r"\b[Tt][Aa][Rr][Gg][Ee][Tt]\s*=\s*['\"](\w+)['\"]", # TARGET = 'col'
            r"target_col\s*=\s*['\"](\w+)['\"]",                  # target_col = 'col'
            r"\blabel\s*=\s*['\"](\w+)['\"]",                     # label = 'col'
            r"(?:train_)?labels?\s*=\s*df\s*\[\s*['\"](\w+)['\"]\s*\]",
            r"df\.drop\s*\(\s*['\"](\w+)['\"]",                   # X = df.drop('col')
        ]
        _skip = {"columns", "index", "shape", "values", "dtype", "axis", "inplace"}
        for pattern in patterns:
            m = re.search(pattern, source)
            if m:
                candidate = m.group(1)
                if candidate not in _skip:
                    return candidate
        return None

    def _infer_task_type(self, source: str, meta: Dict) -> str:
        """Infer 'classification' or 'regression' from metrics / source patterns."""
        clf_metrics = {"f1", "accuracy", "auc_roc", "precision", "recall",
                       "pr_auc", "log_loss", "classification_report"}
        reg_metrics = {"mse", "mae", "r2", "rmse"}
        found = {m["name"] for m in meta.get("metrics", [])}
        if found & clf_metrics:
            return "classification"
        if found & reg_metrics:
            return "regression"
        src_lower = source.lower()
        clf_kw = ["classif", "f1_score", "accuracy_score", "roc_auc",
                  "predict_proba", "stratified", "class_weight"]
        reg_kw = ["regression", "mean_squared", "r2_score", "rmse",
                  "mean_absolute"]
        if sum(kw in src_lower for kw in clf_kw) >= sum(kw in src_lower for kw in reg_kw):
            return "classification"
        return "regression"

    # ── CSV loading ────────────────────────────────────────────────────────────

    def _load_csv(self, csv_path: str, run_id: str) -> pd.DataFrame:
        """
        Load CSV with memory-efficient dtype downcasting.
        Accepts either a local path or a gs:// GCS URI.
        """
        # ── Download from GCS if needed ────────────────────────────────────────
        local_path = csv_path
        _tmp_path: str | None = None
        if csv_path.startswith("gs://"):
            from src.gcs_storage import download_to_tmp
            self._log(run_id, "swarm_orchestrator",
                      f"Downloading from GCS: {csv_path}")
            _tmp_path = download_to_tmp(csv_path)
            local_path = _tmp_path
            self._log(run_id, "swarm_orchestrator",
                      f"Downloaded to {local_path}")

        self._log(run_id, "swarm_orchestrator",
                  f"Loading {Path(local_path).name}…")

        try:
            return self._read_csv_optimised(local_path, run_id)
        finally:
            if _tmp_path:
                Path(_tmp_path).unlink(missing_ok=True)

    def _read_csv_optimised(self, csv_path: str, run_id: str) -> pd.DataFrame:
        """Internal: read a local CSV with dtype downcasting."""
        # Small sample pass to find downcast opportunities
        sample = pd.read_csv(csv_path, nrows=1000, low_memory=False)
        dtype_map: Dict = {}
        for col in sample.columns:
            dt = sample[col].dtype
            if dt == "float64":
                dtype_map[col] = "float32"
            elif dt == "int64":
                mn, mx = sample[col].min(), sample[col].max()
                if mn >= -32768 and mx <= 32767:
                    dtype_map[col] = "int32"   # conservative — full file may differ

        df = pd.read_csv(csv_path, dtype=dtype_map, low_memory=False)
        mb = df.memory_usage(deep=True).sum() / 1e6
        self._log(run_id, "swarm_orchestrator",
                  f"Loaded: {len(df):,} rows × {len(df.columns)} cols | {mb:.1f} MB",
                  payload={"n_rows": len(df), "n_cols": len(df.columns), "mb": round(mb, 1)})
        return df

    def _guess_target_col(self, columns: List[str]) -> Optional[str]:
        candidates = ["target", "label", "y", "class", "outcome", "response",
                      "cancel", "churn", "default", "fraud", "survived", "status"]
        for name in candidates:
            for col in columns:
                if col.lower() == name:
                    return col
        return columns[-1] if columns else None

    # ── Stratified splitting ───────────────────────────────────────────────────

    def _split_stratified(
        self,
        df: pd.DataFrame,
        target_col: str,
        task_type: str,
        n_agents: int,
        run_id: str,
    ) -> List[pd.DataFrame]:
        if n_agents == 1:
            return [df]

        if task_type == "classification":
            try:
                from sklearn.model_selection import StratifiedKFold
                skf = StratifiedKFold(n_splits=n_agents, shuffle=True, random_state=42)
                y = df[target_col]
                chunks = [
                    df.iloc[idx].reset_index(drop=True)
                    for _, idx in skf.split(df, y)
                ]
                self._log(run_id, "swarm_orchestrator",
                          f"Split into {n_agents} stratified folds "
                          f"(~{len(chunks[0]):,} rows each)")
                return chunks
            except Exception as e:
                self._log(run_id, "swarm_orchestrator",
                          f"Stratified split failed ({e}), falling back to random split")

        # Fallback: random shuffle + sequential chunks
        df_sh = df.sample(frac=1, random_state=42).reset_index(drop=True)
        chunk_size = math.ceil(len(df_sh) / n_agents)
        chunks = [
            df_sh.iloc[i * chunk_size: (i + 1) * chunk_size].reset_index(drop=True)
            for i in range(n_agents)
        ]
        self._log(run_id, "swarm_orchestrator",
                  f"Split into {n_agents} random chunks (~{chunk_size:,} rows each)")
        return chunks

    # ── Worker execution ───────────────────────────────────────────────────────

    def _run_workers(
        self,
        chunks: List[pd.DataFrame],
        target_col: str,
        task_type: str,
        time_budget: int,
        max_workers: int,
        run_id: str,
    ) -> List[Dict]:
        from src.agents.swarm_worker import SwarmWorkerAgent

        n_agents = len(chunks)
        results: List[Dict] = []

        def _run_one(agent_id: int, chunk: pd.DataFrame) -> Dict:
            worker = SwarmWorkerAgent(db=self.db)
            # Acquire lock before FLAML fit to avoid XGBoost/LightGBM thread conflicts.
            # Feature prep (inside worker.run) happens before fit, so we hold the
            # lock only during the actual training portion.
            with _FLAML_LOCK:
                return worker.run(
                    chunk=chunk,
                    target_col=target_col,
                    task_type=task_type,
                    time_budget=time_budget,
                    run_id=run_id,
                    agent_id=agent_id,
                )

        self._log(run_id, "swarm_orchestrator",
                  f"Launching {n_agents} workers ({max_workers} parallel)…")

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_id = {
                pool.submit(_run_one, i, chunk): i
                for i, chunk in enumerate(chunks)
            }
            for future in as_completed(future_to_id):
                agent_id = future_to_id[future]
                try:
                    r = future.result()
                    if r.get("status") == "ok":
                        results.append(r)
                        bm = r["best_model"]
                        score = bm["metrics"].get("val_score", 0)
                        _oprint(
                            f"Worker {agent_id} done [{len(results)}/{n_agents}] -> "
                            f"{bm['name']} val_score={score:.4f} flaml={bm.get('flaml', False)}"
                        )
                        self._log(run_id, "swarm_orchestrator",
                                  f"Worker {agent_id + 1}/{n_agents} — "
                                  f"best: {bm['name']} val_score={score:.4f}",
                                  payload={
                                      "agent_id": agent_id,
                                      "best_model": bm["name"],
                                      "val_score": score,
                                      "flaml": bm.get("flaml", False),
                                  })
                    else:
                        _oprint(f"Worker {agent_id} FAILED: {r.get('error', '?')}")
                        self._log(run_id, "swarm_orchestrator",
                                  f"Worker {agent_id + 1}/{n_agents} failed: "
                                  f"{r.get('error', '?')}")
                except Exception as exc:
                    _oprint(f"Worker {agent_id} raised exception: {exc}")
                    self._log(run_id, "swarm_orchestrator",
                              f"Worker {agent_id + 1}/{n_agents} raised: {exc}")

        return results

    # ── Aggregation (divide-and-conquer tree merge) ────────────────────────────

    def _aggregate(self, results: List[Dict], run_id: str) -> Dict:
        """
        Two-level tournament:
          Level 1 (branches) — best model within each model family
          Level 2 (root)     — global champion across all families
        """
        by_family: Dict[str, Dict] = {}
        for r in results:
            bm = r["best_model"]
            family = bm.get("family", "Unknown")
            score  = bm["metrics"].get("val_score", 0.0)
            if family not in by_family or score > by_family[family]["metrics"]["val_score"]:
                by_family[family] = bm

        family_summary = {f: round(m["metrics"]["val_score"], 4)
                          for f, m in by_family.items()}
        _oprint(f"Family winners: {family_summary}")
        self._log(run_id, "swarm_orchestrator",
                  f"Family winners: {family_summary}",
                  payload={"family_bests": family_summary})

        champion = max(by_family.values(),
                       key=lambda x: x["metrics"].get("val_score", 0.0))
        _oprint(f"Champion -> {champion['name']} | val_score={champion['metrics'].get('val_score', '?'):.4f} | FLAML={champion.get('flaml', False)}")
        return champion

    # ── Ledger storage ─────────────────────────────────────────────────────────

    def _store_in_ledger(
        self,
        csv_path: str,
        best_model: Dict,
        target_col: str,
        task_type: str,
        n_rows: int,
        run_id: str,
        project_id: Optional[int],
        team_member: Optional[str],
    ) -> Optional[int]:
        if self.db is None:
            return None
        try:
            metadata = {
                "file_path": csv_path,
                "file_type": "csv",
                "models": [best_model],
                "metrics": [
                    {"name": k, "value": round(v, 4)}
                    for k, v in best_model.get("metrics", {}).items()
                    if isinstance(v, float)
                ],
                "preprocessing": [],
                "environment": {},
                "raw_text": (
                    f"Swarm AutoML result | {n_rows:,} rows | "
                    f"target={target_col} | task={task_type} | run_id={run_id} | "
                    f"champion={best_model.get('name')} "
                    f"val_score={best_model['metrics'].get('val_score', '?')}"
                ),
            }
            entry_id = self.db.insert_ledger_entry(
                metadata,
                project_id=project_id,
                team_member=team_member,
            )
            self._log(run_id, "swarm_orchestrator",
                      f"Champion stored in ledger → entry #{entry_id}")
            return entry_id
        except Exception as e:
            self._log(run_id, "swarm_orchestrator", f"Ledger store failed: {e}")
            return None

    # ── Evaluation + final summary print ──────────────────────────────────────

    def _evaluate_and_print(self, swarm_result: dict, run_id: str) -> dict:
        """Run evaluation metrics and print the final summary banner."""
        eval_result: dict = {}
        try:
            from src.evaluation.evaluate_agents import evaluate_swarm
            # Skip DeepEval by default in production (needs API key + adds latency).
            # Set env var DEEPEVAL_ENABLED=1 to enable it.
            run_de = os.getenv("DEEPEVAL_ENABLED", "0") == "1"
            eval_result = evaluate_swarm(swarm_result, run_deepeval=run_de)
        except Exception as exc:
            eval_result = {"error": str(exc)}

        best = swarm_result.get("best_model", {})
        metrics = best.get("metrics", {})
        score = metrics.get("val_score", 0.0)
        n_succeeded = swarm_result.get("n_succeeded", 0)
        n_agents    = swarm_result.get("n_agents", 1)
        flaml_count = sum(
            1 for w in swarm_result.get("all_worker_results", [])
            if w.get("best_model", {}).get("flaml", False)
        )

        # Custom metric statuses
        custom = eval_result.get("custom_metrics", {})
        overall = eval_result.get("overall_status", "?")

        print(f"\n{'='*64}", flush=True)
        print(f"[{_ts()}]  SWARM RESULT  |  run_id={run_id}", flush=True)
        print(f"{'='*64}", flush=True)
        print(f"  Champion     : {best.get('name', '?')}  ({best.get('family', '?')})", flush=True)
        print(f"  Val score    : {score:.4f}", flush=True)
        print(f"  Workers      : {n_succeeded}/{n_agents} succeeded  "
              f"({100 * n_succeeded / max(n_agents, 1):.0f}%)", flush=True)
        print(f"  FLAML used   : {flaml_count}/{n_succeeded} workers", flush=True)
        print(f"  Dataset rows : {swarm_result.get('n_rows', '?'):,}", flush=True)
        print(f"  Target       : {swarm_result.get('target_col', '?')} ({swarm_result.get('task_type', '?')})", flush=True)
        if swarm_result.get("entry_id"):
            print(f"  Ledger entry : #{swarm_result['entry_id']}", flush=True)

        # Evaluation summary
        print(f"\n  --- Evaluation ({overall}) ---", flush=True)
        for metric_name, metric_val in custom.items():
            if isinstance(metric_val, dict):
                st = metric_val.get("status", "?")
                note = metric_val.get("note", "")
                print(f"  {metric_name:<26}: [{st}]  {note}", flush=True)

        de = eval_result.get("deepeval_metrics")
        if de and not de.get("error"):
            print(f"\n  --- DeepEval (LLM-based) ---", flush=True)
            for k in ("faithfulness", "answer_relevancy", "context_relevancy"):
                v = de.get(k)
                print(f"  {k:<26}: {v}", flush=True)
        elif de and de.get("error"):
            print(f"  DeepEval     : skipped ({de['error'][:60]})", flush=True)

        print(f"{'='*64}\n", flush=True)

        # Log eval to traces
        self._log(run_id, "swarm_orchestrator",
                  f"Evaluation complete — overall={overall}",
                  payload=eval_result)
        return eval_result

    # ── Utilities ──────────────────────────────────────────────────────────────

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
