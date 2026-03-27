from __future__ import annotations

from pathlib import Path

from biofake.io import load_config
from biofake.models.baseline import BaselineDetector
from biofake.models.calibrate import pick_threshold
from biofake.models.hybrid import HybridDetector


def test_load_config_merges_stage_files() -> None:
    config, raw = load_config("configs/experiments/full_report.yaml")
    assert config.name == "full_report"
    assert "detector" in raw
    assert config.detector.kind == "hybrid_cpu"


def test_baseline_detector_trains_and_predicts(mini_human_records: list[dict], mini_generated_records: list[dict], tmp_path: Path) -> None:
    records = mini_human_records[:2] + mini_generated_records[:2]
    detector = BaselineDetector().fit(records)
    probabilities = detector.predict_proba(records)
    assert len(probabilities) == len(records)
    model_path = tmp_path / "baseline.joblib"
    detector.save(str(model_path))
    restored = BaselineDetector.load(str(model_path))
    assert len(restored.predict_proba(records)) == len(records)


def test_hybrid_detector_uses_fallback_embeddings(mini_human_records: list[dict], mini_generated_records: list[dict], tmp_path: Path) -> None:
    records = mini_human_records[:2] + mini_generated_records[:2]
    detector = HybridDetector().fit(records)
    probabilities = detector.predict_proba(records)
    assert len(probabilities) == len(records)
    model_path = tmp_path / "hybrid.joblib"
    detector.save(str(model_path))
    restored = HybridDetector.load(str(model_path))
    assert len(restored.predict(records)) == len(records)


def test_pick_threshold_returns_unit_interval() -> None:
    threshold = pick_threshold([0.1, 0.3, 0.8, 0.9], [0, 0, 1, 1])
    assert 0.0 <= threshold <= 1.0
