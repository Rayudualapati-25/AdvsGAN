# Biofake Data Slice

This directory documents the data-processing slice for `biofake`, a CPU-first biomedical text deepfake robustness project.

## Scope

The data layer is designed around PubMed RCT-style abstracts. It does three things:

1. Reconstruct sectioned abstracts into a canonical structure.
2. Clean and normalize text deterministically.
3. Assign stable train, validation, and test splits from a row key.

## Processed row shape

The normalized row object is `biofake.data.ProcessedRow`.

Core fields:

- `row_id`
- `source_id`
- `pmid`
- `split`
- `label`
- `title`
- `abstract`
- `text`
- `sections`
- `source`
- `schema_version`
- `provenance`

`ProcessedRow.to_dict()` produces a flat record that is safe for CSV or JSONL export. `ProcessedRow.from_dict()` accepts common aliases such as `id`, `doc_id`, `title`, `abstract_text`, `subset`, and `metadata`.

## PubMed RCT cleaning

`process_pubmed_rct_record()`:

- normalizes Unicode and whitespace,
- reconstructs sectioned abstract content into `BACKGROUND`, `METHODS`, `RESULTS`, and `CONCLUSION`,
- keeps a full `text` field by combining title and abstract,
- assigns a deterministic split when `split="auto"`.

`split_processed_rows()` hashes each row key and assigns a stable split based on the configured ratios.

## Expected usage

Use `process_pubmed_rct_record()` for raw rows, `iter_processed_rows()` for mixed iterables, and `split_processed_rows()` when you need reproducible dataset partitioning without depending on external state.
