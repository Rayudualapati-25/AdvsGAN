"""Evaluation helpers for BioFake."""

from .ablation import (
    build_ablation_scenarios,
    filter_rows_by_attack_metadata,
    leave_one_family_out,
)
from .metrics import (
    accuracy_score,
    attack_success_rate,
    family_attack_summary,
    robustness_gap,
    robustness_ratio,
)
from .reporting import build_robustness_report, render_markdown_report, render_json_report

__all__ = [
    "accuracy_score",
    "attack_success_rate",
    "family_attack_summary",
    "robustness_gap",
    "robustness_ratio",
    "build_ablation_scenarios",
    "filter_rows_by_attack_metadata",
    "leave_one_family_out",
    "build_robustness_report",
    "render_markdown_report",
    "render_json_report",
]

