from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from biofake.cli import app


def test_pipeline_run_end_to_end(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "pipeline",
            "run",
            "--config",
            "configs/experiments/full_report.yaml",
            "--override",
            f"paths.runs_dir={tmp_path / 'runs'}",
            "--override",
            f"detector.model_artifact={tmp_path / 'model.joblib'}",
            "--override",
            f"detector.threshold_artifact={tmp_path / 'threshold.json'}",
            "--override",
            f"eval.metrics_output={tmp_path / 'metrics.json'}",
            "--override",
            f"eval.report_output={tmp_path / 'report.md'}",
            "--override",
            f"eval.prediction_output={tmp_path / 'predictions.jsonl'}",
            "--override",
            f"data.processed_path={tmp_path / 'processed.jsonl'}",
            "--override",
            f"generation.output_path={tmp_path / 'generated.jsonl'}",
            "--override",
            f"adversary.output_path={tmp_path / 'attacked.jsonl'}",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "predictions.jsonl").exists()
