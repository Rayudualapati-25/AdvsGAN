from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from biofake.evaluation.reporting import build_robustness_report


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for start, end in zip(edges[:-1], edges[1:]):
        mask = (y_prob >= start) & (y_prob < end if end < 1.0 else y_prob <= end)
        if not np.any(mask):
            continue
        accuracy = np.mean(y_true[mask] == (y_prob[mask] >= 0.5))
        confidence = np.mean(y_prob[mask])
        ece += (np.sum(mask) / len(y_prob)) * abs(accuracy - confidence)
    return float(ece)


def tpr_at_fpr(y_true: np.ndarray, y_prob: np.ndarray, target_fpr: float) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    valid = np.where(fpr <= target_fpr)[0]
    if len(valid) == 0:
        return 0.0
    return float(np.max(tpr[valid]))


def classification_metrics(y_true: Iterable[int], y_prob: Iterable[float], threshold: float) -> dict[str, float]:
    y_true_arr = np.asarray(list(y_true), dtype=int)
    y_prob_arr = np.asarray(list(y_prob), dtype=float)
    y_pred = (y_prob_arr >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred, labels=[0, 1]).ravel()
    precision, recall, _ = precision_recall_curve(y_true_arr, y_prob_arr)
    pr_auc = average_precision_score(y_true_arr, y_prob_arr)
    return {
        "accuracy": float(accuracy_score(y_true_arr, y_pred)),
        "macro_f1": float(f1_score(y_true_arr, y_pred, average="macro", zero_division=0)),
        "f1_synthetic": float(f1_score(y_true_arr, y_pred, zero_division=0)),
        "auroc": float(roc_auc_score(y_true_arr, y_prob_arr)) if len(set(y_true_arr.tolist())) > 1 else 0.0,
        "pr_auc": float(pr_auc),
        "precision_end": float(precision[-1]) if len(precision) else 0.0,
        "recall_end": float(recall[-1]) if len(recall) else 0.0,
        "tpr_at_1_fpr": tpr_at_fpr(y_true_arr, y_prob_arr, 0.01),
        "tpr_at_5_fpr": tpr_at_fpr(y_true_arr, y_prob_arr, 0.05),
        "ece": expected_calibration_error(y_true_arr, y_prob_arr),
        "brier": float(brier_score_loss(y_true_arr, y_prob_arr)),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def build_prediction_rows(records: list[dict[str, Any]], probabilities: list[float], threshold: float) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for record, probability in zip(records, probabilities):
        output.append(
            {
                **record,
                "prediction": "synthetic" if probability >= threshold else "human",
                "probability_synthetic": float(probability),
            }
        )
    return output


def build_attacked_comparison(
    clean_predictions: list[dict[str, Any]],
    attacked_predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    clean_by_id = {row["id"]: row for row in clean_predictions}
    comparison: list[dict[str, Any]] = []

    human_rows = [row for row in clean_predictions if row["label"] == "human"]
    for row in human_rows:
        comparison.append(
            {
                **row,
                "baseline_prediction": row["prediction"],
                "adversarial_prediction": row["prediction"],
                "attack_family": "none",
                "attack_name": "none",
                "attack_success": False,
                "fallback_used": False,
            }
        )

    for row in attacked_predictions:
        parent = clean_by_id.get(row.get("parent_id", ""))
        if parent is None:
            continue
        baseline_prediction = parent["prediction"]
        adversarial_prediction = row["prediction"]
        comparison.append(
            {
                **row,
                "baseline_prediction": baseline_prediction,
                "adversarial_prediction": adversarial_prediction,
                "attack_family": row.get("attack") or row.get("meta", {}).get("attack_family", "unknown"),
                "attack_name": row.get("meta", {}).get("attack_name", row.get("attack", "unknown")),
                "fallback_used": bool(row.get("meta", {}).get("attack_metadata", {}).get("fallback_used", False)),
                "attack_success": baseline_prediction == "synthetic" and adversarial_prediction != "synthetic",
            }
        )
    return comparison


def summarize_errors(predictions: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in predictions:
        key = row.get("attack") or row.get("generator") or row.get("source", "unknown")
        correct = row.get("prediction") == row.get("label")
        summary[str(key)]["correct" if correct else "incorrect"] += 1
    return {key: dict(value) for key, value in summary.items()}


def evaluate_prediction_sets(
    clean_predictions: list[dict[str, Any]],
    attacked_predictions: list[dict[str, Any]] | None,
    threshold: float,
) -> dict[str, Any]:
    clean_y = [1 if row["label"] == "synthetic" else 0 for row in clean_predictions]
    clean_prob = [row["probability_synthetic"] for row in clean_predictions]
    report: dict[str, Any] = {
        "clean": classification_metrics(clean_y, clean_prob, threshold),
        "clean_errors": summarize_errors(clean_predictions),
    }
    if attacked_predictions:
        attacked_mix = [
            row for row in clean_predictions if row["label"] == "human" and row["split"] == attacked_predictions[0]["split"]
        ] + attacked_predictions
        attacked_y = [1 if row["label"] == "synthetic" else 0 for row in attacked_mix]
        attacked_prob = [row["probability_synthetic"] for row in attacked_mix]
        comparison = build_attacked_comparison(clean_predictions, attacked_predictions)
        report["attacked"] = classification_metrics(attacked_y, attacked_prob, threshold)
        report["robustness"] = build_robustness_report(comparison)
        report["attack_success_rate"] = report["robustness"]["attack_success_rate"]
        report["robustness_gap"] = report["robustness"]["robustness_gap"]
        report["attacked_errors"] = summarize_errors(attacked_predictions)
    return report

