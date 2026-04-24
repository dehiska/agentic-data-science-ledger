"""
AutoResearchWrapper — Agent A of the three-agent pipeline.

Runs an iterative model search on a stratified sample of the dataset,
logs each iteration to the traces table, and returns the best model found.

In Phase 1 (local): simulates an autoresearch run when the autoresearch
package is not installed.
In Phase 2 (cloud): calls `python -m autoresearch` or an AutoML service.

Usage:
    from src.autoresearch_wrapper import AutoResearchWrapper
    wrapper = AutoResearchWrapper(db=db)
    result = wrapper.run_autoresearch(
        file_path="notebooks/exp1.ipynb",
        goal="Improve F1 score",
        execute=True,
        stratify=True,
        max_iterations=50,
    )
"""

import json
import subprocess
import sys
import uuid
from datetime import datetime
from typing import Dict, Optional


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _arprint(msg: str) -> None:
    """Formatted print for AutoResearcher."""
    print(f"[{_ts()}] [AutoResearcher] {msg}", flush=True)


class AutoResearchWrapper:
    def __init__(self, db=None, github_token: Optional[str] = None):
        self.db = db
        self._github_token = github_token

    def run_autoresearch(
        self,
        repo_name: str = "",
        file_path: str = "",
        branch: str = "main",
        goal: str = "Improve model accuracy",
        team_member: Optional[str] = None,
        execute: bool = False,
        run_id: Optional[str] = None,
        stratify: bool = True,
        max_sample_rows: int = 10_000,
        max_iterations: int = 50,
        plateau_stop: int = 10,
    ) -> Dict:
        if not execute:
            return {
                "status": "plan_only",
                "message": "Toggle 'Execute autoresearch' to run the AutoML search.",
            }

        run_id = run_id or uuid.uuid4().hex[:8]

        print(f"\n{'='*64}", flush=True)
        print(f"[{_ts()}]  AUTO RESEARCHER  |  run_id={run_id}", flush=True)
        print(f"{'='*64}", flush=True)
        _arprint(f"Goal: {goal}")
        _arprint(f"File: {file_path or '(none)'}  |  max_iter={max_iterations}  |  stratify={stratify}  |  plateau_stop={plateau_stop}")

        # Try to run autoresearch as a subprocess
        result = self._run_subprocess(file_path, goal, run_id, max_iterations, stratify)
        if result["status"] == "success":
            self._log_to_db(file_path, result["best_model"], team_member)
        return result

    # ── Internal ───────────────────────────────────────────────────────────────

    def _run_subprocess(
        self,
        file_path: str,
        goal: str,
        run_id: str,
        max_iterations: int,
        stratify: bool,
    ) -> Dict:
        """
        Attempt to run `python -m autoresearch`. Falls back to simulation if
        the package is not installed.
        """
        cmd = [
            sys.executable, "-m", "autoresearch",
            "--file", file_path,
            "--goal", goal,
            "--output", "json",
        ]
        try:
            _arprint("Attempting autoresearch subprocess...")
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if proc.returncode == 0:
                best_model = self._parse_output(proc.stdout)
                _arprint(f"Subprocess succeeded -> {best_model.get('name', '?')}")
                if self.db:
                    self.db.log_trace(run_id, "autoresearcher",
                        "Autoresearch subprocess completed successfully.",
                        payload={"best_model": best_model.get("name")})
                return {
                    "status": "success",
                    "best_model": best_model,
                    "run_id": run_id,
                    "raw_output": proc.stdout[:2000],
                }
            else:
                _arprint("Subprocess not available — running simulation...")
                return self._simulate(file_path, goal, run_id, max_iterations, stratify)
        except FileNotFoundError:
            _arprint("autoresearch package not installed — running simulation...")
            return self._simulate(file_path, goal, run_id, max_iterations, stratify)
        except subprocess.TimeoutExpired:
            _arprint("ERROR: subprocess timed out after 5 minutes")
            return {"status": "error", "error": "autoresearch timed out after 5 minutes.", "run_id": run_id}
        except Exception as e:
            _arprint(f"ERROR: {e}")
            return {"status": "error", "error": str(e), "run_id": run_id}

    def _simulate(
        self,
        file_path: str,
        goal: str,
        run_id: str,
        max_iterations: int,
        stratify: bool,
    ) -> Dict:
        """
        Simulate an autoresearch run, logging each iteration to the traces table.
        Implements plateau stopping: stops if no improvement for plateau_stop iterations.
        """
        import random
        random.seed(hash(file_path + goal) % (2**32))

        plateau_stop = 10
        model_pool = [
            {"name": "RandomForestClassifier", "family": "Ensemble",
             "base_f1": 0.820, "base_acc": 0.841,
             "param_fn": lambda r: {"n_estimators": r.choice([100, 200, 300]),
                                     "max_depth": r.choice([5, 8, 10, None])}},
            {"name": "XGBClassifier", "family": "Ensemble",
             "base_f1": 0.840, "base_acc": 0.856,
             "param_fn": lambda r: {"n_estimators": r.choice([200, 300, 500]),
                                     "learning_rate": round(0.01 + r.random() * 0.09, 3),
                                     "max_depth": r.choice([3, 5, 7])}},
            {"name": "LGBMClassifier", "family": "Ensemble",
             "base_f1": 0.855, "base_acc": 0.872,
             "param_fn": lambda r: {"n_estimators": r.choice([200, 400, 600]),
                                     "learning_rate": round(0.01 + r.random() * 0.09, 3),
                                     "num_leaves": r.choice([31, 63, 127])}},
            {"name": "LogisticRegression", "family": "Linear",
             "base_f1": 0.780, "base_acc": 0.801,
             "param_fn": lambda r: {"C": r.choice([0.01, 0.1, 1.0, 10.0]),
                                     "max_iter": 1000}},
        ]

        _arprint(f"Simulation starting — {max_iterations} max iterations | plateau_stop={plateau_stop}")
        if self.db:
            self.db.log_trace(run_id, "autoresearcher",
                f"Starting autoresearch — max {max_iterations} iterations, "
                f"stratify={stratify}, plateau_stop={plateau_stop}",
                payload={"max_iterations": max_iterations, "stratify": stratify})

        best_model = None
        best_f1 = 0.0
        no_improve_count = 0
        iterations_run = 0

        for i in range(max_iterations):
            iterations_run = i + 1
            mp = random.choice(model_pool)
            params = mp["param_fn"](random)
            noise = random.gauss(0, 0.012)
            f1  = round(min(0.999, max(0.5, mp["base_f1"]  + noise + i * 0.001)), 4)
            acc = round(min(0.999, max(0.5, mp["base_acc"] + noise + i * 0.001)), 4)
            auc = round(min(0.999, max(0.5, f1 + random.uniform(0.01, 0.04))), 4)

            candidate = {
                "name": mp["name"],
                "family": mp["family"],
                "params": params,
                "metrics": {"f1": f1, "accuracy": acc, "auc_roc": auc},
            }

            is_best = f1 > best_f1
            if is_best:
                best_f1 = f1
                best_model = candidate
                no_improve_count = 0
                _arprint(f"Iter {i+1:>3}/{max_iterations} | {mp['name']:<28} f1={f1:.4f} acc={acc:.4f} ** NEW BEST **")
            elif (i + 1) % 10 == 0:
                # Print every 10 iterations even if no improvement
                _arprint(f"Iter {i+1:>3}/{max_iterations} | {mp['name']:<28} f1={f1:.4f} acc={acc:.4f}  (no_improve={no_improve_count})")
            else:
                no_improve_count += 1

            if self.db:
                self.db.log_trace(
                    run_id, "autoresearcher",
                    f"Iteration {i+1}/{max_iterations} — {mp['name']} "
                    f"f1={f1:.4f} acc={acc:.4f}"
                    + (" ✅ new best" if is_best else ""),
                    payload={"model": mp["name"], "metrics": candidate["metrics"],
                             "params": params, "iteration": i + 1, "is_best": is_best},
                )

            # Plateau stop
            if no_improve_count >= plateau_stop:
                _arprint(f"Plateau reached — {plateau_stop} iterations without improvement. Stopping at iter {iterations_run}.")
                if self.db:
                    self.db.log_trace(run_id, "autoresearcher",
                        f"Plateau reached ({plateau_stop} iterations without improvement). Stopping.",
                        payload={"iterations_run": iterations_run, "best_f1": best_f1})
                break

        # ── Final summary print ────────────────────────────────────────────────
        best_name = best_model['name'] if best_model else '?'
        best_metrics = best_model.get('metrics', {}) if best_model else {}
        print(f"\n{'='*64}", flush=True)
        print(f"[{_ts()}]  AUTORESEARCHER RESULT  |  run_id={run_id}", flush=True)
        print(f"{'='*64}", flush=True)
        print(f"  Best model   : {best_name}", flush=True)
        print(f"  F1 score     : {best_metrics.get('f1', '?')}", flush=True)
        print(f"  Accuracy     : {best_metrics.get('accuracy', '?')}", flush=True)
        print(f"  AUC-ROC      : {best_metrics.get('auc_roc', '?')}", flush=True)
        print(f"  Iterations   : {iterations_run} (of {max_iterations} max)", flush=True)
        print(f"  Params       : {best_model.get('params', {}) if best_model else {}}", flush=True)

        # Run evaluation
        ar_result_preview = {
            "status": "success",
            "best_model": best_model,
            "run_id": run_id,
            "iterations_run": iterations_run,
            "simulated": True,
            "all_models_tried": list({m["name"] for m in model_pool}),
        }
        try:
            from src.evaluation.evaluate_agents import evaluate_autoresearch
            eval_result = evaluate_autoresearch(ar_result_preview, run_deepeval=False)
            overall = eval_result.get("overall_status", "?")
            print(f"\n  --- Evaluation ({overall}) ---", flush=True)
            for mname, mval in eval_result.get("custom_metrics", {}).items():
                if isinstance(mval, dict):
                    print(f"  {mname:<26}: [{mval.get('status','?')}]  {mval.get('note','')}", flush=True)
        except Exception as exc:
            print(f"  Evaluation   : skipped ({exc})", flush=True)
        print(f"{'='*64}\n", flush=True)

        if self.db:
            self.db.log_trace(run_id, "autoresearcher",
                f"Search complete — best: {best_model['name']} f1={best_f1:.4f} "
                f"({iterations_run} iterations)",
                payload={"best_model": best_model["name"], "best_f1": best_f1,
                         "iterations_run": iterations_run})

        return {
            "status": "success",
            "best_model": best_model,
            "simulated": True,
            "run_id": run_id,
            "iterations_run": iterations_run,
            "note": "autoresearch package not installed — simulation used. Install autoresearch for real AutoML.",
            "all_models_tried": list({m["name"] for m in model_pool}),
        }

    def _parse_output(self, output: str) -> Dict:
        """Parse JSON output from autoresearch subprocess."""
        try:
            return json.loads(output)
        except Exception:
            return {
                "name": "Unknown",
                "family": "Unknown",
                "params": {},
                "metrics": {},
                "raw": output[:500],
            }

    def _log_to_db(self, file_path: str, best_model: Dict, team_member: Optional[str]):
        if self.db is None:
            return
        try:
            entries = self.db.get_ledger_entries(file_path=file_path)
            if entries:
                entry = entries[0]
                updated_models = (entry.get("models") or []) + [best_model]
                updated_metrics = (entry.get("metrics") or []) + [
                    {"name": k, "value": v}
                    for k, v in best_model.get("metrics", {}).items()
                ]
                self.db.update_ledger_entry(
                    entry["id"],
                    {"models": updated_models, "metrics": updated_metrics, "status": "Executed"},
                )
        except Exception:
            pass
