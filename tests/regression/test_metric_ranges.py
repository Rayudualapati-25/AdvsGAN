from __future__ import annotations

from biofake.evaluation.robustness import build_prediction_rows, evaluate_prediction_sets
from biofake.models.baseline import BaselineDetector
from biofake.models.calibrate import pick_threshold
from biofake.models.hybrid import HybridDetector


def test_metric_ranges_on_fixture_data(
    mini_human_records: list[dict],
    mini_generated_records: list[dict],
    mini_rewritten_records: list[dict],
) -> None:
    train_rows = mini_human_records[:2] + mini_generated_records[:2]
    clean_test_rows = [mini_human_records[-1], mini_generated_records[-1]]
    attacked_test_rows = [mini_rewritten_records[-1]]

    baseline = BaselineDetector().fit(train_rows)
    baseline_threshold = pick_threshold(
        baseline.predict_proba(train_rows),
        [0, 0, 1, 1],
    )
    baseline_clean = build_prediction_rows(clean_test_rows, baseline.predict_proba(clean_test_rows).tolist(), baseline_threshold)
    baseline_attacked = build_prediction_rows(attacked_test_rows, baseline.predict_proba(attacked_test_rows).tolist(), baseline_threshold)
    baseline_metrics = evaluate_prediction_sets(baseline_clean, baseline_attacked, baseline_threshold)

    hybrid = HybridDetector().fit(train_rows + mini_rewritten_records[:2])
    hybrid_threshold = pick_threshold(
        hybrid.predict_proba(train_rows),
        [0, 0, 1, 1],
    )
    hybrid_clean = build_prediction_rows(clean_test_rows, hybrid.predict_proba(clean_test_rows).tolist(), hybrid_threshold)
    hybrid_attacked = build_prediction_rows(attacked_test_rows, hybrid.predict_proba(attacked_test_rows).tolist(), hybrid_threshold)
    hybrid_metrics = evaluate_prediction_sets(hybrid_clean, hybrid_attacked, hybrid_threshold)

    assert 0.0 <= baseline_metrics["clean"]["macro_f1"] <= 1.0
    assert 0.0 <= hybrid_metrics["clean"]["macro_f1"] <= 1.0
    assert 0.0 <= baseline_metrics["attacked"]["macro_f1"] <= 1.0
    assert 0.0 <= hybrid_metrics["attacked"]["macro_f1"] <= 1.0

