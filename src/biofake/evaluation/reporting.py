"""Report generation utilities."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any, Mapping

from biofake.adversary.schema import extract_value, row_copy
from .metrics import accuracy_score, attack_success_rate, family_attack_summary, robustness_gap, robustness_ratio


def build_robustness_report(
    rows: list[Mapping[str, Any] | Any],
    *,
    label_field: str = "label",
    baseline_prediction_field: str = "baseline_prediction",
    adversarial_prediction_field: str = "adversarial_prediction",
) -> dict[str, Any]:
    copied_rows = [row_copy(row) for row in rows]
    baseline_accuracy = accuracy_score(
        copied_rows,
        label_field=label_field,
        prediction_field=baseline_prediction_field,
    )
    attacked_accuracy = accuracy_score(
        copied_rows,
        label_field=label_field,
        prediction_field=adversarial_prediction_field,
    )
    report = {
        "sample_count": len(copied_rows),
        "baseline_accuracy": baseline_accuracy,
        "attacked_accuracy": attacked_accuracy,
        "robustness_gap": robustness_gap(baseline_accuracy, attacked_accuracy),
        "robustness_ratio": robustness_ratio(baseline_accuracy, attacked_accuracy),
        "attack_success_rate": attack_success_rate(
            copied_rows,
            label_field=label_field,
            baseline_prediction_field=baseline_prediction_field,
            adversarial_prediction_field=adversarial_prediction_field,
        ),
        "attack_families": family_attack_summary(copied_rows),
        "attack_name_counts": dict(
            Counter(str(extract_value(row, ("attack_name",), default="unknown")) for row in copied_rows)
        ),
    }
    return report


def render_markdown_report(report: Mapping[str, Any]) -> str:
    lines = ["# BioFake Robustness Report", ""]
    lines.append(f"- Sample count: {report.get('sample_count', 0)}")
    lines.append(f"- Baseline accuracy: {report.get('baseline_accuracy', 0.0):.4f}")
    lines.append(f"- Attacked accuracy: {report.get('attacked_accuracy', 0.0):.4f}")
    lines.append(f"- Robustness gap: {report.get('robustness_gap', 0.0):.4f}")
    lines.append(f"- Robustness ratio: {report.get('robustness_ratio', 0.0):.4f}")
    lines.append(f"- Attack success rate: {report.get('attack_success_rate', 0.0):.4f}")
    lines.append("")
    lines.append("## Attack Families")
    family_summary = report.get("attack_families", {})
    if not family_summary:
        lines.append("- None")
    else:
        for family, stats in sorted(family_summary.items()):
            lines.append(
                f"- {family}: count={stats.get('count', 0)}, "
                f"success_rate={stats.get('success_rate', 0.0):.4f}, "
                f"fallback_rate={stats.get('fallback_rate', 0.0):.4f}"
            )
    return "\n".join(lines)


def render_json_report(report: Mapping[str, Any], *, indent: int = 2) -> str:
    return json.dumps(report, indent=indent, sort_keys=True)

