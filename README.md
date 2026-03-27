# BioFake

`biofake` is a CPU-first biomedical text deepfake robustness project built for the topic `Adversarial Agent vs Generative Agent for Deepfake Robustness`.

It implements:

- `PubMed 20k RCT`-style data preparation with a built-in fallback demo corpus
- local synthetic abstract generation with a deterministic fallback when no `llama.cpp` model is configured
- adversarial rewriting with paraphrase, compression/expansion, and style-transfer attacks
- two detectors:
  - `baseline_tfidf_lr`
  - `hybrid_cpu` = TF-IDF + stylometric/readability/citation features + frozen-embedding fallback
- evaluation, ablations, reports, and a Streamlit demo

## Quickstart

```bash
PYTHONPATH=src python3 -m biofake.cli pipeline run --config configs/experiments/full_report.yaml
streamlit run src/biofake/demo/app.py -- --config configs/experiments/robust_hybrid.yaml
```

## Main Commands

```bash
PYTHONPATH=src python3 -m biofake.cli data prepare --config configs/experiments/robust_hybrid.yaml
PYTHONPATH=src python3 -m biofake.cli generate synth --config configs/experiments/robust_hybrid.yaml
PYTHONPATH=src python3 -m biofake.cli attack rewrite --config configs/experiments/robust_hybrid.yaml
PYTHONPATH=src python3 -m biofake.cli train detector --config configs/experiments/robust_hybrid.yaml
PYTHONPATH=src python3 -m biofake.cli eval run --config configs/experiments/robust_hybrid.yaml
PYTHONPATH=src python3 -m biofake.cli eval ablate --config configs/experiments/full_report.yaml
```

## Dataset Notes

- The original screenshot references `MedFake Text Corpus`, but that could not be verified as a practical machine-readable benchmark.
- The implemented project uses `PubMed 20k RCT`-style abstracts as the core real-text source and generates the synthetic side locally.
- Auxiliary biomedical benchmarks such as `SciFact` or `CliniFact` are left as transfer-evaluation extensions rather than primary AI-text labels.

## Model Notes

- If you have a compatible local environment and cached transformer weights, set `BIOFAKE_ENABLE_TORCH=1` to enable transformer embeddings.
- In this workspace, the default path is the deterministic hashed-embedding fallback so tests and the full pipeline run without downloading models.

## Outputs

Each run writes:

- `artifacts/runs/<run_id>/resolved_config.yaml`
- `artifacts/runs/<run_id>/command.txt`
- stage summaries, metrics, thresholds, and report artifacts

Primary shared artifacts are written under:

- `artifacts/models/`
- `artifacts/reports/`
- `artifacts/figures/`

## Validation

```bash
PYTHONPATH=src pytest tests/unit -q
PYTHONPATH=src pytest tests/integration -q
PYTHONPATH=src pytest tests/regression -q
PYTHONPATH=src pytest tests/smoke -q
```

