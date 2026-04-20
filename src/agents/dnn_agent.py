"""
DNNAgent — analyzes Deep Neural Network configurations and suggests improvements.

Specializes in:
  - Architecture suggestions (layers, dropout, batch norm)
  - Uncertainty quantification (MC Dropout, ensemble)
  - Training stability (learning rate, batch size)
  - Regularization analysis
"""

from typing import Dict, List
from .base_agent import BaseAgent


class DNNAgent(BaseAgent):
    def suggest(self, context: Dict) -> Dict:
        return self.analyze_dnn(
            context.get("models", []),
            context.get("metrics", []),
        )

    def analyze_dnn(self, models: List[Dict], metrics: List[Dict]) -> Dict:
        dnn_models = [m for m in models if m.get("family") == "Deep Neural Network"]

        if not dnn_models:
            return {
                "agent": "DNNAgent",
                "applicable": False,
                "message": "No Deep Neural Network models detected in this notebook.",
                "suggestions": [],
                "uncertainty_methods": [],
            }

        suggestions = []
        uncertainty_methods = []
        architecture_notes = []

        for model in dnn_models:
            params = model.get("params", {})
            name = model.get("name", "")

            # ── Dropout ────────────────────────────────────────────────────────
            has_dropout = any("dropout" in str(v).lower() for v in params.values())
            if not has_dropout:
                suggestions.append({
                    "category": "Regularization",
                    "action": "Add Dropout(0.2-0.5) to prevent overfitting.",
                    "code_hint": "nn.Dropout(0.3)  # between dense layers",
                    "impact": "Reduces val loss gap by 5-15%",
                })
                uncertainty_methods.append("MC Dropout")

            # ── Batch Norm ─────────────────────────────────────────────────────
            has_bn = "batchnorm" in str(params).lower() or "batch_norm" in str(params).lower()
            if not has_bn:
                suggestions.append({
                    "category": "Training Stability",
                    "action": "Add BatchNorm1d after each linear layer (before activation) for faster convergence.",
                    "code_hint": "nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3)",
                    "impact": "2-3x faster convergence; more stable training",
                })

            architecture_notes.append({
                "model": name,
                "detected_params": params,
                "has_dropout": has_dropout,
                "has_batch_norm": has_bn,
            })

        # ── Uncertainty quantification ─────────────────────────────────────────
        suggestions.append({
            "category": "Uncertainty Quantification",
            "action": "Enable MC Dropout at inference (model.train() mode) with T=30 forward passes.",
            "code_hint": (
                "def mc_predict(model, X, T=30):\n"
                "    model.train()\n"
                "    preds = torch.stack([model(X) for _ in range(T)])\n"
                "    return preds.mean(0), preds.var(0)  # mean, epistemic uncertainty"
            ),
            "impact": "Identifies ~10-20% of samples with high uncertainty for active learning",
        })
        uncertainty_methods.extend(["MC Dropout (T=30)", "Predictive Variance (Aleatoric)"])

        # ── Learning rate ──────────────────────────────────────────────────────
        suggestions.append({
            "category": "Learning Rate",
            "action": "Use ReduceLROnPlateau scheduler: reduce LR by 0.5 when val_loss plateaus for 3 epochs.",
            "code_hint": (
                "scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(\n"
                "    optimizer, mode='min', factor=0.5, patience=3\n"
                ")\nscheduler.step(val_loss)"
            ),
            "impact": "Typically improves final val metric by 1-5%",
        })

        # ── Early stopping ─────────────────────────────────────────────────────
        suggestions.append({
            "category": "Training",
            "action": "Implement early stopping (patience=10) to avoid wasted epochs and overfitting.",
            "code_hint": (
                "# Track best val_loss, save checkpoint, stop if no improvement for 10 epochs"
            ),
            "impact": "Prevents 5-30% overfitting; saves compute time",
        })

        rag_context = self._get_rag_context("deep neural network uncertainty MC dropout regularization", k=2)

        return {
            "agent": "DNNAgent",
            "applicable": True,
            "dnn_count": len(dnn_models),
            "suggestions": suggestions,
            "uncertainty_methods": list(set(uncertainty_methods)),
            "architecture_notes": architecture_notes,
            "rag_context": rag_context,
            "summary": f"Found {len(dnn_models)} DNN(s). {len(suggestions)} improvement suggestions.",
        }
