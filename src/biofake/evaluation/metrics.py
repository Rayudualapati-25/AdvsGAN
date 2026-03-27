"""Core robustness metrics."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping

from biofake.adversary.schema import coerce_bool, extract_value, row_copy


def accuracy_score(
    rows: Iterable[Mapping[str, Any] | Any],
    *,
    label_field: str = "label",
    prediction_field: str = "prediction",
) -> float:
    items = list(rows)
    if not items:
        return 0.0
    correct = 0
    total = 0
    for row in items:
        label = extract_value(row, (label_field,), default=None)
        prediction = extract_value(row, (prediction_field,), default=None)
        if label is None or prediction is None:
            continue
        total += 1
        if str(label) == str(prediction):
            correct += 1
    return correct / total if total else 0.0


def attack_success_rate(
    rows: Iterable[Mapping[str, Any] | Any],
    *,
    label_field: str = "label",
    baseline_prediction_field: str = "baseline_prediction",
    adversarial_prediction_field: str = "adversarial_prediction",
    targeted_label_field: str = "attack_target",
    success_field: str = "attack_success",
) -> float:
    items = list(rows)
    if not items:
        return 0.0
    successes = 0
    total = 0
    for row in items:
        if extract_value(row, (success_field,), default=None) is not None:
            total += 1
            if coerce_bool(extract_value(row, (success_field,), default=False)):
                successes += 1
            continue

        baseline_prediction = extract_value(row, (baseline_prediction_field,), default=None)
        adversarial_prediction = extract_value(row, (adversarial_prediction_field,), default=None)
        target = extract_value(row, (targeted_label_field,), default=None)
        label = extract_value(row, (label_field,), default=None)

        if adversarial_prediction is None:
            continue

        total += 1
        if target is not None:
            if str(adversarial_prediction) == str(target):
                successes += 1
        elif label is not None and baseline_prediction is not None:
            if str(baseline_prediction) == str(label) and str(adversarial_prediction) != str(label):
                successes += 1
        elif baseline_prediction is not None:
            if str(baseline_prediction) != str(adversarial_prediction):
                successes += 1
    return successes / total if total else 0.0


def robustness_gap(baseline_score: float, attacked_score: float) -> float:
    return baseline_score - attacked_score


def robustness_ratio(baseline_score: float, attacked_score: float) -> float:
    if baseline_score == 0:
        return 0.0
    return attacked_score / baseline_score


def family_attack_summary(
    rows: Iterable[Mapping[str, Any] | Any],
    *,
    family_field: str = "attack_family",
    success_field: str = "attack_success",
    fallback_field: str = "fallback_used",
) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "count": 0,
        "success_rate": 0.0,
        "fallback_rate": 0.0,
    })
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        family = str(extract_value(row, (family_field,), default="unknown"))
        grouped[family].append(row_copy(row))

    for family, group in grouped.items():
        count = len(group)
        success_rate = attack_success_rate(group, success_field=success_field)
        fallback_count = sum(1 for row in group if coerce_bool(row.get(fallback_field, False)))
        summary[family] = {
            "count": count,
            "success_rate": success_rate,
            "fallback_rate": fallback_count / count if count else 0.0,
        }
    return dict(summary)
