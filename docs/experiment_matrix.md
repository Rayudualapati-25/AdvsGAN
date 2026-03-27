# Experiment Matrix

## Baselines

- `screenshot_baseline`
  - synthetic generation: `seqgan_legacy`
  - detector: `baseline_tfidf_lr`
- `robust_hybrid`
  - synthetic generation: local LLM or deterministic fallback
  - attacks: paraphrase, compression/expansion, style transfer
  - detector: `hybrid_cpu`

## Main Comparisons

- clean test performance
- attacked test performance
- attack success rate
- robustness gap
- leave-one-family-out ablations

