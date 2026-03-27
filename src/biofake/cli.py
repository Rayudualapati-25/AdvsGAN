from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import typer

from biofake.adversary.rewrite_agent import rewrite_synthetic_rows
from biofake.data.loaders import prepare_dataset
from biofake.evaluation.ablations import build_ablation_scenarios
from biofake.evaluation.report import render_full_report
from biofake.evaluation.robustness import build_prediction_rows, evaluate_prediction_sets
from biofake.generation.synthesize import generate_synthetic_rows
from biofake.io import init_run, load_config, read_jsonl, resolved_path, write_json, write_jsonl
from biofake.models.baseline import BaselineConfig, BaselineDetector
from biofake.models.calibrate import pick_threshold
from biofake.models.hybrid import HybridConfig, HybridDetector
from biofake.seed import set_seed

app = typer.Typer(help="BioFake CLI")
data_app = typer.Typer(help="Data preparation commands")
generate_app = typer.Typer(help="Synthetic generation commands")
attack_app = typer.Typer(help="Adversarial rewrite commands")
train_app = typer.Typer(help="Training commands")
eval_app = typer.Typer(help="Evaluation commands")
demo_app = typer.Typer(help="Demo asset commands")
pipeline_app = typer.Typer(help="Pipeline commands")

app.add_typer(data_app, name="data")
app.add_typer(generate_app, name="generate")
app.add_typer(attack_app, name="attack")
app.add_typer(train_app, name="train")
app.add_typer(eval_app, name="eval")
app.add_typer(demo_app, name="demo")
app.add_typer(pipeline_app, name="pipeline")


def _runtime(
    config_path: str,
    run_id: str | None,
    seed: int | None,
    overrides: list[str] | None,
) -> tuple[Any, Any, Any]:
    override_list = list(overrides or [])
    if seed is not None:
        override_list.append(f"seed={seed}")
    config, raw = load_config(config_path, override_list)
    set_seed(config.seed)
    metadata = init_run(config, config_path, sys.argv, run_id=run_id)
    return config, raw, metadata


def _load_stage_records(config: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    humans = read_jsonl(config.data.processed_path)
    generated = read_jsonl(config.generation.output_path)
    attacked = read_jsonl(config.adversary.output_path)
    return humans, generated, attacked


def _train_rows(config: Any, detector_kind: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    humans, generated, attacked = _load_stage_records(config)
    train_rows = [row for row in humans + generated if row["split"] == "train"]
    val_rows = [row for row in humans + generated if row["split"] == "val"]
    if detector_kind == "hybrid_cpu":
        train_rows += [row for row in attacked if row["split"] == "train"]
        val_rows += [row for row in attacked if row["split"] == "val"]
    return train_rows, val_rows or train_rows


def _test_rows(config: Any, detector_kind: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    humans, generated, attacked = _load_stage_records(config)
    clean = [row for row in humans + generated if row["split"] == "test"]
    attacked_rows = [row for row in attacked if row["split"] == "test"] if detector_kind == "hybrid_cpu" else []
    return clean, attacked_rows


def _build_detector(config: Any) -> Any:
    if config.detector.kind == "baseline_tfidf_lr":
        return BaselineDetector(
            BaselineConfig(
                max_word_features=config.detector.max_word_features,
                max_char_features=config.detector.max_char_features,
                word_ngram_max=config.detector.tfidf_word_ngram_max,
                char_ngram_max=config.detector.tfidf_char_ngram_max,
                logistic_c=config.detector.logistic_c,
            )
        )
    return HybridDetector(
        HybridConfig(
            max_word_features=config.detector.max_word_features,
            max_char_features=config.detector.max_char_features,
            word_ngram_max=config.detector.tfidf_word_ngram_max,
            char_ngram_max=config.detector.tfidf_char_ngram_max,
            logistic_c=config.detector.logistic_c,
            calibration_cv=config.detector.calibration_cv,
            use_embeddings=config.detector.use_embeddings,
            embedding_model_name=config.detector.embedding_model_name,
            embedding_dim_fallback=config.detector.embedding_dim_fallback,
            embedding_batch_size=config.detector.embedding_batch_size,
            local_files_only=config.detector.local_files_only,
        )
    )


def _save_threshold(config: Any, threshold: float, run_output: Path) -> None:
    payload = {"threshold": threshold, "metric": config.detector.threshold_metric}
    write_json(config.detector.threshold_artifact, payload)
    write_json(run_output / "threshold.json", payload)


@data_app.command("prepare")
def data_prepare(
    config: str = typer.Option(..., "--config"),
    run_id: str | None = typer.Option(None, "--run-id"),
    seed: int | None = typer.Option(None, "--seed"),
    force: bool = typer.Option(False, "--force"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    override: list[str] | None = typer.Option(None, "--override"),
) -> None:
    cfg, _, meta = _runtime(config, run_id, seed, override)
    records, summary = prepare_dataset(cfg)
    if dry_run:
        typer.echo(json.dumps(summary, indent=2))
        return
    write_json(meta.output_path() / "data_prepare_summary.json", summary)
    typer.echo(f"Prepared {summary['record_count']} human records")


@generate_app.command("synth")
def generate_synth(
    config: str = typer.Option(..., "--config"),
    run_id: str | None = typer.Option(None, "--run-id"),
    seed: int | None = typer.Option(None, "--seed"),
    force: bool = typer.Option(False, "--force"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    override: list[str] | None = typer.Option(None, "--override"),
) -> None:
    cfg, _, meta = _runtime(config, run_id, seed, override)
    humans = read_jsonl(cfg.data.processed_path) or prepare_dataset(cfg)[0]
    generated, summary = generate_synthetic_rows(humans, cfg)
    if dry_run:
        typer.echo(json.dumps(summary, indent=2))
        return
    write_json(meta.output_path() / "generation_summary.json", summary)
    typer.echo(f"Generated {len(generated)} synthetic records")


@attack_app.command("rewrite")
def attack_rewrite(
    config: str = typer.Option(..., "--config"),
    run_id: str | None = typer.Option(None, "--run-id"),
    seed: int | None = typer.Option(None, "--seed"),
    force: bool = typer.Option(False, "--force"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    override: list[str] | None = typer.Option(None, "--override"),
) -> None:
    cfg, _, meta = _runtime(config, run_id, seed, override)
    generated = read_jsonl(cfg.generation.output_path)
    if not generated:
        humans = read_jsonl(cfg.data.processed_path) or prepare_dataset(cfg)[0]
        generated = generate_synthetic_rows(humans, cfg)[0]
    attacked, summary = rewrite_synthetic_rows(generated, cfg)
    if dry_run:
        typer.echo(json.dumps(summary, indent=2))
        return
    write_json(meta.output_path() / "attack_summary.json", summary)
    typer.echo(f"Generated {len(attacked)} adversarial rewrites")


@train_app.command("detector")
def train_detector(
    config: str = typer.Option(..., "--config"),
    run_id: str | None = typer.Option(None, "--run-id"),
    seed: int | None = typer.Option(None, "--seed"),
    force: bool = typer.Option(False, "--force"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    override: list[str] | None = typer.Option(None, "--override"),
) -> None:
    cfg, _, meta = _runtime(config, run_id, seed, override)
    train_rows, val_rows = _train_rows(cfg, cfg.detector.kind)
    if not train_rows:
        prepare_dataset(cfg)
        generate_synthetic_rows(read_jsonl(cfg.data.processed_path), cfg)
        if cfg.detector.kind == "hybrid_cpu":
            rewrite_synthetic_rows(read_jsonl(cfg.generation.output_path), cfg)
        train_rows, val_rows = _train_rows(cfg, cfg.detector.kind)
    detector = _build_detector(cfg)
    detector.fit(train_rows)
    val_prob = detector.predict_proba(val_rows)
    val_labels = [1 if row["label"] == "synthetic" else 0 for row in val_rows]
    threshold = pick_threshold(val_prob, val_labels, metric=cfg.detector.threshold_metric)
    if dry_run:
        typer.echo(json.dumps({"train_rows": len(train_rows), "val_rows": len(val_rows), "threshold": threshold}, indent=2))
        return
    detector.save(cfg.detector.model_artifact)
    detector.save(str(meta.output_path() / "model.joblib"))
    _save_threshold(cfg, threshold, meta.output_path())
    typer.echo(f"Trained {cfg.detector.kind} on {len(train_rows)} rows")


@eval_app.command("run")
def eval_run(
    config: str = typer.Option(..., "--config"),
    run_id: str | None = typer.Option(None, "--run-id"),
    seed: int | None = typer.Option(None, "--seed"),
    force: bool = typer.Option(False, "--force"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    override: list[str] | None = typer.Option(None, "--override"),
) -> None:
    cfg, _, meta = _runtime(config, run_id, seed, override)
    threshold_payload = json.loads(Path(cfg.detector.threshold_artifact).read_text(encoding="utf-8"))
    threshold = float(threshold_payload["threshold"])
    detector = (
        BaselineDetector.load(cfg.detector.model_artifact)
        if cfg.detector.kind == "baseline_tfidf_lr"
        else HybridDetector.load(cfg.detector.model_artifact)
    )
    clean_rows, attacked_rows = _test_rows(cfg, cfg.detector.kind)
    clean_predictions = build_prediction_rows(clean_rows, detector.predict_proba(clean_rows).tolist(), threshold)
    attacked_predictions = (
        build_prediction_rows(attacked_rows, detector.predict_proba(attacked_rows).tolist(), threshold)
        if attacked_rows
        else []
    )
    metrics = evaluate_prediction_sets(clean_predictions, attacked_predictions, threshold)
    all_predictions = clean_predictions + attacked_predictions
    if dry_run:
        typer.echo(json.dumps(metrics, indent=2))
        return
    write_jsonl(cfg.eval.prediction_output, all_predictions)
    write_json(meta.output_path() / "metrics.json", metrics)
    write_json(cfg.eval.metrics_output, metrics)
    report = render_full_report(metrics, all_predictions)
    report_path = Path(cfg.eval.report_output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    (meta.output_path() / "report.md").write_text(report, encoding="utf-8")
    typer.echo(f"Evaluated {len(all_predictions)} rows")


@eval_app.command("ablate")
def eval_ablate(
    config: str = typer.Option(..., "--config"),
    run_id: str | None = typer.Option(None, "--run-id"),
    seed: int | None = typer.Option(None, "--seed"),
    force: bool = typer.Option(False, "--force"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    override: list[str] | None = typer.Option(None, "--override"),
) -> None:
    cfg, _, meta = _runtime(config, run_id, seed, override)
    attacked = read_jsonl(cfg.adversary.output_path)
    scenarios = build_ablation_scenarios(attacked)
    payload = {name: len(rows) for name, rows in scenarios.items()}
    write_json(meta.output_path() / "ablations.json", payload)
    typer.echo(json.dumps(payload, indent=2))


@demo_app.command("build-assets")
def demo_build_assets(
    config: str = typer.Option(..., "--config"),
    run_id: str | None = typer.Option(None, "--run-id"),
    seed: int | None = typer.Option(None, "--seed"),
    force: bool = typer.Option(False, "--force"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    override: list[str] | None = typer.Option(None, "--override"),
) -> None:
    cfg, _, meta = _runtime(config, run_id, seed, override)
    predictions = read_jsonl(cfg.eval.prediction_output)
    examples = predictions[:12]
    asset_path = meta.output_path() / "demo_assets.json"
    write_json(asset_path, {"examples": examples})
    typer.echo(f"Demo assets written to {asset_path}")


@pipeline_app.command("run")
def pipeline_run(
    config: str = typer.Option(..., "--config"),
    run_id: str | None = typer.Option(None, "--run-id"),
    seed: int | None = typer.Option(None, "--seed"),
    force: bool = typer.Option(False, "--force"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    override: list[str] | None = typer.Option(None, "--override"),
) -> None:
    cfg, _, meta = _runtime(config, run_id, seed, override)
    prepare_dataset(cfg)
    humans = read_jsonl(cfg.data.processed_path)
    generate_synthetic_rows(humans, cfg)
    if cfg.detector.kind == "hybrid_cpu":
        rewrite_synthetic_rows(read_jsonl(cfg.generation.output_path), cfg)
    detector = _build_detector(cfg)
    train_rows, val_rows = _train_rows(cfg, cfg.detector.kind)
    detector.fit(train_rows)
    threshold = pick_threshold(
        detector.predict_proba(val_rows),
        [1 if row["label"] == "synthetic" else 0 for row in val_rows],
        metric=cfg.detector.threshold_metric,
    )
    detector.save(cfg.detector.model_artifact)
    _save_threshold(cfg, threshold, meta.output_path())
    clean_rows, attacked_rows = _test_rows(cfg, cfg.detector.kind)
    clean_predictions = build_prediction_rows(clean_rows, detector.predict_proba(clean_rows).tolist(), threshold)
    attacked_predictions = (
        build_prediction_rows(attacked_rows, detector.predict_proba(attacked_rows).tolist(), threshold)
        if attacked_rows
        else []
    )
    metrics = evaluate_prediction_sets(clean_predictions, attacked_predictions, threshold)
    report = render_full_report(metrics, clean_predictions + attacked_predictions)
    write_json(cfg.eval.metrics_output, metrics)
    write_json(meta.output_path() / "metrics.json", metrics)
    write_jsonl(cfg.eval.prediction_output, clean_predictions + attacked_predictions)
    report_path = Path(cfg.eval.report_output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    (meta.output_path() / "report.md").write_text(report, encoding="utf-8")
    typer.echo(f"Pipeline complete: {meta.output_dir}")


if __name__ == "__main__":
    app()
