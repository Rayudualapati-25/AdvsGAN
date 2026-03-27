from __future__ import annotations

from typing import Any


def top_uncertain_predictions(predictions: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    ranked = sorted(
        predictions,
        key=lambda row: abs(float(row.get("probability_synthetic", 0.5)) - 0.5),
    )
    return ranked[:limit]


def attack_family_breakdown(predictions: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for row in predictions:
        family = str(row.get("attack") or row.get("meta", {}).get("attack_family") or "none")
        bucket = summary.setdefault(family, {"correct": 0, "incorrect": 0})
        bucket["correct" if row.get("prediction") == row.get("label") else "incorrect"] += 1
    return summary

