from __future__ import annotations

from typing import Any

from biofake.evaluation.error_analysis import attack_family_breakdown, top_uncertain_predictions
from biofake.evaluation.reporting import render_markdown_report


def render_full_report(metrics: dict[str, Any], predictions: list[dict[str, Any]]) -> str:
    clean = metrics.get("clean", {})
    attacked = metrics.get("attacked", {})
    lines = [
        "# BioFake Evaluation Report",
        "",
        "## Clean Metrics",
        f"- AUROC: {clean.get('auroc', 0.0):.4f}",
        f"- PR-AUC: {clean.get('pr_auc', 0.0):.4f}",
        f"- Macro-F1: {clean.get('macro_f1', 0.0):.4f}",
        f"- Accuracy: {clean.get('accuracy', 0.0):.4f}",
        f"- TPR@5% FPR: {clean.get('tpr_at_5_fpr', 0.0):.4f}",
        f"- ECE: {clean.get('ece', 0.0):.4f}",
        "",
    ]
    if attacked:
        lines.extend(
            [
                "## Attacked Metrics",
                f"- AUROC: {attacked.get('auroc', 0.0):.4f}",
                f"- PR-AUC: {attacked.get('pr_auc', 0.0):.4f}",
                f"- Macro-F1: {attacked.get('macro_f1', 0.0):.4f}",
                f"- Accuracy: {attacked.get('accuracy', 0.0):.4f}",
                f"- TPR@5% FPR: {attacked.get('tpr_at_5_fpr', 0.0):.4f}",
                f"- Attack Success Rate: {metrics.get('attack_success_rate', 0.0):.4f}",
                f"- Robustness Gap: {metrics.get('robustness_gap', 0.0):.4f}",
                "",
            ]
        )
    if "robustness" in metrics:
        lines.append("## Robustness Summary")
        lines.append(render_markdown_report(metrics["robustness"]))
        lines.append("")
    lines.append("## Uncertain Predictions")
    for row in top_uncertain_predictions(predictions):
        lines.append(
            f"- {row.get('id')}: label={row.get('label')} pred={row.get('prediction')} p={row.get('probability_synthetic', 0.0):.4f}"
        )
    lines.append("")
    lines.append("## Attack Family Breakdown")
    for family, counts in attack_family_breakdown(predictions).items():
        lines.append(f"- {family}: correct={counts['correct']} incorrect={counts['incorrect']}")
    return "\n".join(lines)

