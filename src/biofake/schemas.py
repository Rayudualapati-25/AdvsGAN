from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PathsConfig(BaseModel):
    root_dir: str = "."
    data_dir: str = "data"
    external_dir: str = "data/external"
    interim_dir: str = "data/interim"
    processed_dir: str = "data/processed"
    splits_dir: str = "data/splits"
    artifacts_dir: str = "artifacts"
    runs_dir: str = "artifacts/runs"
    models_dir: str = "artifacts/models"
    reports_dir: str = "artifacts/reports"
    figures_dir: str = "artifacts/figures"
    tables_dir: str = "artifacts/tables"


class DataConfig(BaseModel):
    dataset_name: str = "pubmed_rct"
    raw_train_path: str = "data/external/pubmed-rct/train.txt"
    raw_val_path: str = "data/external/pubmed-rct/dev.txt"
    raw_test_path: str = "data/external/pubmed-rct/test.txt"
    processed_path: str = "data/processed/processed.jsonl"
    dedupe: bool = True
    min_sentences: int = 2
    max_samples_per_split: int = 0


class GenerationConfig(BaseModel):
    backend: Literal["auto", "mock", "llama_cli"] = "auto"
    model_id: str = "Qwen2.5-0.5B-Instruct"
    model_path: str = "data/external/models/qwen2.5-0.5b-instruct.gguf"
    fallback_model_id: str = "TinyLlama-1.1B-Chat-v1.0"
    fallback_model_path: str = "data/external/models/tinyllama-1.1b-chat.gguf"
    output_path: str = "data/processed/generated.jsonl"
    enabled_splits: list[str] = Field(default_factory=lambda: ["train", "val", "test"])
    temperature: float = 0.2
    max_new_tokens: int = 220
    system_prompt_style: str = "biomedical_abstract"


class AdversaryConfig(BaseModel):
    backend: Literal["auto", "mock", "llama_cli"] = "auto"
    output_path: str = "data/processed/attacked.jsonl"
    enabled_splits: list[str] = Field(default_factory=lambda: ["train", "val", "test"])
    attacks: list[str] = Field(default_factory=lambda: ["paraphrase", "compress_expand", "style_transfer"])
    strength: Literal["light", "medium", "heavy"] = "medium"
    max_variants_per_row: int = 1


class DetectorConfig(BaseModel):
    kind: Literal["baseline_tfidf_lr", "hybrid_cpu"] = "hybrid_cpu"
    model_artifact: str = "artifacts/models/hybrid_cpu.joblib"
    threshold_artifact: str = "artifacts/models/hybrid_cpu_threshold.json"
    tfidf_word_ngram_max: int = 2
    tfidf_char_ngram_max: int = 5
    max_word_features: int = 20000
    max_char_features: int = 12000
    logistic_c: float = 3.0
    calibration_cv: int = 3
    threshold_metric: Literal["f1", "balanced_accuracy"] = "f1"
    use_embeddings: bool = True
    embedding_model_name: str = "allenai/scibert_scivocab_uncased"
    embedding_dim_fallback: int = 64
    embedding_batch_size: int = 8
    local_files_only: bool = True


class EvalConfig(BaseModel):
    prediction_output: str = "artifacts/reports/predictions.jsonl"
    metrics_output: str = "artifacts/reports/metrics.json"
    report_output: str = "artifacts/reports/report.md"
    plots_prefix: str = "artifacts/figures/eval"
    evaluate_splits: list[str] = Field(default_factory=lambda: ["test"])
    positive_label: Literal["synthetic"] = "synthetic"


class LoggingConfig(BaseModel):
    level: str = "INFO"


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str = "experiment"
    description: str = ""
    seed: int = 13
    paths: PathsConfig = Field(default_factory=PathsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    adversary: AdversaryConfig = Field(default_factory=AdversaryConfig)
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)


class ProcessedRow(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    split: str
    label: Literal["human", "synthetic"]
    source: str
    generator: str | None = None
    attack: str | None = None
    parent_id: str | None = None
    text: str
    meta: dict[str, Any] = Field(default_factory=dict)

    @field_validator("text")
    @classmethod
    def _validate_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("text must not be empty")
        return cleaned

    def binary_label(self) -> int:
        return 1 if self.label == "synthetic" else 0


class PredictionRow(BaseModel):
    id: str
    split: str
    label: Literal["human", "synthetic"]
    prediction: Literal["human", "synthetic"]
    probability_synthetic: float
    source: str
    generator: str | None = None
    attack: str | None = None
    parent_id: str | None = None
    text: str
    meta: dict[str, Any] = Field(default_factory=dict)


class MetricBundle(BaseModel):
    name: str
    values: dict[str, float]


class RunMetadata(BaseModel):
    run_id: str
    config_path: str
    resolved_config_path: str
    command: str
    output_dir: str

    def output_path(self) -> Path:
        return Path(self.output_dir)

