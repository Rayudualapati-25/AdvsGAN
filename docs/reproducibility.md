# Reproducibility

- Default seed is controlled by config and can be overridden with `--seed`.
- Each run writes its resolved configuration and command line into `artifacts/runs/<run_id>/`.
- The project defaults to deterministic local generation and hashed embeddings so the pipeline can run without external downloads.
- To opt into local transformer embeddings, set `BIOFAKE_ENABLE_TORCH=1` and ensure compatible cached model weights are available.

