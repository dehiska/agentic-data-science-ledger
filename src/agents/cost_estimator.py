"""
CostEstimatorAgent — estimates GCP compute costs for a training plan.

Returns recommended GCP instance, estimated runtime, and total cost.
Uses heuristics based on model family, dataset size hints, and training steps.
"""

from typing import Dict
from .base_agent import BaseAgent


# GCP pricing estimates (USD/hour, 2024 on-demand)
_GCP_INSTANCES = {
    "n1-standard-2":  {"vcpus": 2,  "ram_gb": 7.5,   "usd_hr": 0.095},
    "n1-standard-4":  {"vcpus": 4,  "ram_gb": 15,    "usd_hr": 0.190},
    "n1-standard-8":  {"vcpus": 8,  "ram_gb": 30,    "usd_hr": 0.380},
    "n1-standard-16": {"vcpus": 16, "ram_gb": 60,    "usd_hr": 0.760},
    "n1-highmem-4":   {"vcpus": 4,  "ram_gb": 26,    "usd_hr": 0.237},
    "a2-highgpu-1g":  {"vcpus": 12, "ram_gb": 85,    "usd_hr": 3.673, "gpu": "A100"},
    "n1-standard-4+T4": {"vcpus": 4, "ram_gb": 15,   "usd_hr": 0.584, "gpu": "T4"},
}

_FAMILY_INSTANCE_MAP = {
    "Linear":                "n1-standard-2",
    "Ensemble":               "n1-standard-4",
    "Deep Neural Network":    "n1-standard-4+T4",
    "Transformer":            "a2-highgpu-1g",
    "Time Series":            "n1-standard-4",
    "Count Data":             "n1-standard-2",
    "Clustering":             "n1-standard-2",
    "SVM":                    "n1-standard-4",
    "KNN":                    "n1-standard-2",
    "Unknown":                "n1-standard-2",
}

_FAMILY_HOURS_MAP = {
    "Linear":                0.1,
    "Ensemble":               0.5,
    "Deep Neural Network":    2.0,
    "Transformer":            4.0,
    "Time Series":            1.0,
    "Count Data":             0.1,
    "Clustering":             0.3,
    "SVM":                    0.5,
    "KNN":                    0.2,
    "Unknown":                0.3,
}


class CostEstimatorAgent(BaseAgent):
    def suggest(self, context: Dict) -> Dict:
        return self.estimate_cost(context)

    def estimate_cost(self, plan: Dict) -> Dict:
        models = plan.get("models", [])
        suggested = plan.get("suggested_models", [])
        search_strategy = plan.get("suggested_search_strategy", "random")

        if not models and not suggested:
            return self._default_estimate()

        # Determine most expensive model family in the plan
        all_model_names = (
            [m.get("family", "Unknown") for m in models]
            + suggested
        )

        # Map model names to families for suggested models
        from src.mcp_server import MCPServer
        family_for_suggested = [
            MCPServer.MODEL_CLASSES.get(name, "Ensemble")
            for name in suggested
        ]

        families = [m.get("family", "Unknown") for m in models] + family_for_suggested
        dominant_family = max(set(families), key=families.count) if families else "Unknown"

        instance = _FAMILY_INSTANCE_MAP.get(dominant_family, "n1-standard-4")
        base_hours = _FAMILY_HOURS_MAP.get(dominant_family, 0.5)

        # Adjust for hyperparameter search
        search_multipliers = {"grid": 10.0, "bayesian": 4.0, "random": 3.0}
        multiplier = search_multipliers.get(search_strategy, 1.0)
        total_hours = base_hours * multiplier

        instance_info = _GCP_INSTANCES.get(instance, _GCP_INSTANCES["n1-standard-4"])
        total_cost = instance_info["usd_hr"] * total_hours

        return {
            "gcp_instance": instance,
            "vcpus": instance_info["vcpus"],
            "ram_gb": instance_info["ram_gb"],
            "gpu": instance_info.get("gpu"),
            "time_hours": round(total_hours, 2),
            "usd_per_hour": instance_info["usd_hr"],
            "total_cost": round(total_cost, 3),
            "search_strategy": search_strategy,
            "dominant_family": dominant_family,
            "notes": (
                f"Estimated for {search_strategy} search with {len(suggested)} candidate model(s). "
                f"Actual cost depends on dataset size and epochs."
            ),
        }

    def _default_estimate(self) -> Dict:
        return {
            "gcp_instance": "n1-standard-4",
            "vcpus": 4,
            "ram_gb": 15,
            "gpu": None,
            "time_hours": 0.5,
            "usd_per_hour": 0.190,
            "total_cost": 0.095,
            "search_strategy": "random",
            "dominant_family": "Unknown",
            "notes": "Default estimate — no models detected yet.",
        }
