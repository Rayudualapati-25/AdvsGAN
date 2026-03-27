# Architecture

The pipeline is CLI-first and stage-oriented:

1. `data prepare`
   Loads PubMed-style RCT abstracts from raw text if present, otherwise uses a built-in fallback corpus.
2. `generate synth`
   Builds one synthetic abstract per human record using a local backend or deterministic fallback.
3. `attack rewrite`
   Applies paraphrase, compression/expansion, and style-transfer rewrites to synthetic rows only.
4. `train detector`
   Trains either `baseline_tfidf_lr` or `hybrid_cpu`.
5. `eval run`
   Writes predictions, classification metrics, and robustness summaries.
6. `demo build-assets`
   Packages recent predictions for the Streamlit app.

Canonical row schema:

- `id`
- `split`
- `label`
- `source`
- `generator`
- `attack`
- `parent_id`
- `text`
- `meta`

