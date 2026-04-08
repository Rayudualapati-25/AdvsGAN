# BioFake — Architecture & Framework Reference

## Overview

**BioFake** is a research pipeline for studying the adversarial robustness of biomedical text deepfake detectors. It simulates a full attack-defence loop:

1. A **Generative Agent** synthesizes fake biomedical abstracts from real PubMed RCT text.
2. An **Adversarial Agent** rewrites those fakes to evade detection.
3. A **Detector** is trained and evaluated against both clean and adversarially-rewritten text.

The pipeline is CLI-first, stage-oriented, and fully reproducible via YAML/TOML config files.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| ML / DL | PyTorch 2.2+, scikit-learn 1.6+ |
| NLP models | Hugging Face Transformers (SciBERT) |
| Config validation | Pydantic v2 |
| CLI | Typer |
| Demo UI | Streamlit |
| Serialisation | joblib, JSON, JSONL, YAML, TOML |
| Build / packaging | setuptools, pyproject.toml |
| Testing | pytest |

---

## Pipeline Stages

```
[Raw PubMed-RCT text]
        │
        ▼
  1. data prepare          ← loaders, cleaners, split assignment
        │
        ▼
  2. generate synth        ← LLM/mock backend produces one fake per real abstract
        │
        ▼
  3. attack rewrite        ← adversarial rewrites (paraphrase / compress / style-transfer)
        │
        ▼
  4. train detector        ← fits baseline_tfidf_lr OR hybrid_cpu model
        │
        ▼
  5. eval run              ← predictions, metrics, robustness report
        │
        ▼
  6. demo build-assets     ← packages examples for Streamlit app
```

All six stages can be run individually or end-to-end via `biofake pipeline run`.

---

## Module Map

### Root (`src/biofake/`)

| File | Purpose |
|---|---|
| `__init__.py` | Package init; exposes `__version__` |
| `cli.py` | Typer CLI — registers all sub-commands (`data`, `generate`, `attack`, `train`, `eval`, `demo`, `pipeline`) |
| `schemas.py` | Pydantic models for the entire config tree (`ExperimentConfig`, `DataConfig`, `DetectorConfig`, …) and core data rows (`ProcessedRow`, `PredictionRow`, `RunMetadata`) |
| `io.py` | Low-level I/O helpers — YAML/JSON/JSONL read-write, config loading with key-value overrides, run directory initialisation |
| `seed.py` | Global reproducibility helper — sets seeds for Python `random`, NumPy, and PyTorch |

---

### `data/` — Data Ingestion & Preparation

| File | Purpose |
|---|---|
| `loaders.py` | Top-level `prepare_dataset()` — orchestrates loading, cleaning, splitting, and writing `processed.jsonl` |
| `pubmed_rct.py` | Parser for the PubMed-RCT raw text format; falls back to a built-in mini corpus when no external file is found |
| `cleaners.py` | Text normalisation (whitespace, unicode) applied before splitting |
| `sampling.py` | Optional down-sampling to `max_samples_per_split` per class per split |
| `splits.py` | Deterministic train/val/test assignment by hashing record IDs |

---

### `generation/` — Synthetic Text Generation (Generative Agent)

| File | Purpose |
|---|---|
| `synthesize.py` | Entry point — iterates human rows and calls the active backend to produce one synthetic abstract each |
| `backends.py` | Backend selector (`auto` → tries `llama_cli`, falls back to `mock`); implements the `mock` backend with deterministic template text |
| `local_llm.py` | `llama_cli` backend — shells out to a local GGUF model (Qwen2.5-0.5B or TinyLlama) via the `llama.cpp` CLI |
| `prompts.py` | System and user prompt templates for the `biomedical_abstract` generation style |
| `provenance.py` | Attaches `generator` and `parent_id` fields to every synthetic row for full lineage tracking |
| `schema.py` | Pydantic models specific to generation config and output |
| `synthetic.py` | Utility functions for constructing and validating synthetic `ProcessedRow` dicts |
| `seqgan_legacy.py` | Legacy SeqGAN-based generation path (kept for reproducibility of older runs) |

---

### `adversary/` — Adversarial Rewrites (Adversarial Agent)

| File | Purpose |
|---|---|
| `rewrite_agent.py` | Entry point — applies enabled attacks to every synthetic row and writes `attacked.jsonl` |
| `attacks.py` | Loads per-attack YAML/JSON configs and instantiates attack objects via the registry |
| `registry.py` | Central factory — maps attack name strings to concrete attack classes |
| `base.py` | Abstract `Attack` base class defining the `rewrite(text) → str` contract |
| `paraphrase.py` | Paraphrase attack — sentence-level synonym/reorder rewriting |
| `compression.py` | Compress-expand attack — sentence removal (compress) or expansion with filler sentences |
| `style_transfer.py` | Style-transfer attack — shifts register/vocabulary toward a target style |
| `constraints.py` | Budget constraints — enforces length and semantic-similarity bounds on rewrites |
| `schema.py` | Pydantic models for adversary config and attack payloads |

---

### `features/` — Feature Engineering

| File | Purpose |
|---|---|
| `lexical.py` | `LexicalFeatureBuilder` — builds TF-IDF word-ngram and char-ngram sparse matrices |
| `embeddings.py` | `FrozenTransformerEmbeddings` — mean-pools SciBERT (or fallback random) sentence embeddings |
| `citations.py` | Citation-pattern features — counts and normalises reference markers (e.g. `[1]`, `et al.`) |
| `readability.py` | Readability scores — Flesch-Kincaid, sentence length stats, vocab richness |
| `stylometric.py` | Stylometric signals — function-word ratios, punctuation density, type-token ratio |

---

### `models/` — Detectors

| File | Purpose |
|---|---|
| `baseline.py` | `BaselineDetector` — TF-IDF (word + char ngrams) → Logistic Regression; fast CPU-only detector |
| `hybrid.py` | `HybridDetector` — TF-IDF + SciBERT embeddings + readability + stylometric + citation features → calibrated SGD classifier; adversarially-robust detector |
| `calibrate.py` | `pick_threshold()` — selects the decision threshold that maximises F1 or balanced accuracy on the validation set |
| `persistence.py` | `save_artifact()` / `load_artifact()` — joblib serialisation for both detector types |

---

### `evaluation/` — Metrics & Reporting

| File | Purpose |
|---|---|
| `metrics.py` | Core metric computation — accuracy, F1, AUC-ROC, attack success rate, robustness delta |
| `robustness.py` | `build_prediction_rows()` and `evaluate_prediction_sets()` — compares clean vs. attacked performance |
| `ablations.py` | Splits attacked rows by attack type to enable per-attack ablation comparisons |
| `ablation.py` | Helper dataclasses and grouping logic for ablation scenarios |
| `report.py` | `render_full_report()` — formats all metrics into a human-readable Markdown report |
| `reporting.py` | Lower-level table/section builders used by `report.py` |
| `error_analysis.py` | False-positive / false-negative breakdown by source, generator, and attack type |

---

### `demo/` — Streamlit Interface

| File | Purpose |
|---|---|
| `app.py` | Streamlit app entry point — renders the interactive deepfake detection demo |
| `explain.py` | SHAP-style feature attribution display — explains why a sample was classified as fake |
| `example_texts.py` | Curated human and synthetic text examples shown in the demo sidebar |

---

## Config System (`configs/`)

| Path | Purpose |
|---|---|
| `configs/base.yaml` | Default values for all pipeline stages; every experiment extends this |
| `configs/data/pubmed_rct.yaml` | Dataset paths and split parameters for PubMed-RCT |
| `configs/data/splits_default.yaml` | Train/val/test split ratios |
| `configs/generation/default.toml` | Default LLM backend, temperature, and prompt style |
| `configs/generation/qwen_small.yaml` | Qwen2.5-0.5B-Instruct GGUF generation config |
| `configs/generation/seqgan_legacy.yaml` | Legacy SeqGAN generation config |
| `configs/detector/baseline_tfidf_lr.yaml` | TF-IDF + LR detector hyperparameters |
| `configs/detector/hybrid_cpu.yaml` | Hybrid detector hyperparameters (SciBERT + TF-IDF + features) |
| `configs/adversary/paraphrase.json` | Paraphrase attack strength and sentence-level settings |
| `configs/adversary/compression.json` | Compress-expand attack config |
| `configs/adversary/style_transfer.json` | Style-transfer attack config |
| `configs/adversary/rewrite_cpu.yaml` | CPU-only rewrite agent settings |
| `configs/eval/default.yaml` | Evaluation split selection and output paths |
| `configs/experiments/full_report.yaml` | Full experiment: hybrid detector + all attacks |
| `configs/experiments/robust_hybrid.yaml` | Robustness-focused experiment variant |
| `configs/experiments/screenshot_baseline.yaml` | Quick baseline experiment for screenshots |
| `configs/experiments/full_improved.yaml` | Improved full experiment with tuned parameters |
| `configs/experiments/improved_quick.yaml` | Fast version of the improved experiment |
| `configs/logging.yaml` | Log level and handler configuration |
| `configs/paths.yaml` | Overridable root paths for data, artifacts, and reports |

---

## Artifact Layout (`artifacts/`)

| Path | Contents |
|---|---|
| `artifacts/models/*.joblib` | Serialised detector objects |
| `artifacts/models/*_threshold.json` | Optimal decision threshold for each model |
| `artifacts/reports/predictions.jsonl` | Per-row prediction results |
| `artifacts/reports/metrics.json` | Aggregated evaluation metrics |
| `artifacts/reports/report.md` | Full human-readable evaluation report |
| `artifacts/runs/<run-id>/` | Per-run copies of all outputs for reproducibility |

---

## Data Flow (Row Schema)

Every record passing through the pipeline carries these fields:

| Field | Type | Description |
|---|---|---|
| `id` | `str` | Unique record identifier (hash-based) |
| `split` | `str` | `train`, `val`, or `test` |
| `label` | `str` | `human` or `synthetic` |
| `source` | `str` | Origin dataset name (e.g. `pubmed_rct`) |
| `generator` | `str \| None` | Generation backend used (e.g. `qwen_small`, `mock`) |
| `attack` | `str \| None` | Attack applied (e.g. `paraphrase`, `compress_expand`) |
| `parent_id` | `str \| None` | ID of the source human row this was derived from |
| `text` | `str` | The abstract text |
| `meta` | `dict` | Arbitrary provenance metadata |

---

## Tests (`tests/`)

| Path | Purpose |
|---|---|
| `tests/conftest.py` | Top-level fixtures shared across all test suites |
| `tests/fixtures/` | Minimal JSONL fixtures (mini authentic, generated, rewritten) |
| `tests/unit/` | Per-module unit tests (features, models, adversary, data, evaluation) |
| `tests/integration/test_pipeline_end_to_end.py` | Full pipeline smoke integration test |
| `tests/regression/test_metric_ranges.py` | Guards metric values against unexpected drift |
| `tests/smoke/test_cli_and_demo.py` | CLI invocation and Streamlit import smoke checks |

---

## Key Design Decisions

- **Stage isolation** — each pipeline stage reads from and writes to a well-defined JSONL file, so stages can be re-run independently.
- **Dual detector design** — `baseline_tfidf_lr` is fast and interpretable; `hybrid_cpu` adds SciBERT embeddings and handcrafted features for adversarial robustness.
- **Attack registry pattern** — new attacks can be added by implementing `base.Attack` and registering in `registry.py` without touching the CLI or pipeline.
- **Config-first reproducibility** — every run records the resolved config and CLI args to `artifacts/runs/<run-id>/`, enabling exact reproduction.
- **CPU-first deployment** — SciBERT embeddings can be replaced by a random fallback (`embedding_dim_fallback`) so the full pipeline runs on a laptop without a GPU.
