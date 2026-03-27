from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from biofake.data.pubmed_rct import ProcessedRow as PubmedProcessedRow
from biofake.data.pubmed_rct import process_pubmed_rct_record
from biofake.io import write_jsonl
from biofake.schemas import ExperimentConfig

from .sampling import limit_per_split
from .splits import normalize_split_name


FALLBACK_RAW_RECORDS: list[dict[str, Any]] = [
    {
        "pmid": "demo-001",
        "title": "Nurse-guided monitoring for hypertension",
        "abstract": "BACKGROUND: Hypertension remains a major cardiovascular risk factor.\nMETHODS: We enrolled 120 adults and compared nurse-led monitoring with standard follow-up.\nRESULTS: Systolic pressure fell significantly in the intervention group after 12 weeks.\nCONCLUSION: Structured monitoring improved short-term blood pressure control.",
        "split": "train",
    },
    {
        "pmid": "demo-002",
        "title": "Biomarker triage for sepsis",
        "abstract": "BACKGROUND: Early sepsis detection is challenging in emergency settings.\nMETHODS: We prospectively evaluated a triage biomarker panel in 84 patients with suspected infection.\nRESULTS: The combined panel improved sensitivity relative to physician assessment alone.\nCONCLUSION: Biomarker-guided triage may support earlier sepsis recognition.",
        "split": "train",
    },
    {
        "pmid": "demo-003",
        "title": "Multimodal prophylaxis after laparoscopic surgery",
        "abstract": "BACKGROUND: Postoperative nausea delays recovery after laparoscopic surgery.\nMETHODS: This randomized study assigned 62 participants to standard care or multimodal prophylaxis.\nRESULTS: Multimodal prophylaxis reduced nausea scores and rescue medication use.\nCONCLUSION: Combined prophylaxis improved early postoperative comfort.",
        "split": "val",
    },
    {
        "pmid": "demo-004",
        "title": "Digital coaching for insulin initiation",
        "abstract": "BACKGROUND: Glycemic instability is common after intensive insulin initiation.\nMETHODS: Investigators compared digital coaching with usual diabetes education in 90 adults.\nRESULTS: Time in range improved and nocturnal hypoglycemia decreased with digital coaching.\nCONCLUSION: Remote coaching supported safer insulin adjustment.",
        "split": "test",
    },
]


def parse_pubmed_rct_txt(path: str | Path, split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source = Path(path)
    if not source.exists():
        return rows

    current_id: str | None = None
    sentences: list[tuple[str, str]] = []

    def flush() -> None:
        nonlocal current_id, sentences
        if not sentences:
            return
        abstract = "\n".join(f"{label}: {text}" for label, text in sentences)
        rows.append(
            {
                "pmid": current_id or f"{source.stem}-{len(rows)}",
                "title": "",
                "abstract": abstract,
                "split": split,
            }
        )
        current_id = None
        sentences = []

    for raw_line in source.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            flush()
            continue
        if line.startswith("###"):
            current_id = line[3:].strip()
            continue
        if "\t" in line:
            label, text = line.split("\t", 1)
            sentences.append((label.strip(), text.strip()))
    flush()
    return rows


def canonicalize_row(row: PubmedProcessedRow) -> dict[str, Any]:
    split = normalize_split_name(row.split)
    return {
        "id": row.row_id or row.source_id,
        "split": split,
        "label": "human",
        "source": row.source,
        "generator": None,
        "attack": None,
        "parent_id": None,
        "text": row.text,
        "meta": {
            "title": row.title,
            "abstract": row.abstract,
            "pmid": row.pmid,
            "sections": list(row.sections),
            "provenance": dict(row.provenance),
            "schema_version": row.schema_version,
        },
    }


def deduplicate_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for record in records:
        text = str(record.get("text", "")).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        deduped.append(record)
    return deduped


def load_or_build_human_records(config: ExperimentConfig) -> list[dict[str, Any]]:
    raw_rows: list[dict[str, Any]] = []
    path_pairs = [
        (config.data.raw_train_path, "train"),
        (config.data.raw_val_path, "val"),
        (config.data.raw_test_path, "test"),
    ]
    for path, split in path_pairs:
        raw_rows.extend(parse_pubmed_rct_txt(path, split))
    if not raw_rows:
        raw_rows = list(FALLBACK_RAW_RECORDS)

    processed: list[dict[str, Any]] = []
    for raw in raw_rows:
        row = process_pubmed_rct_record(raw, split=normalize_split_name(str(raw.get("split", "train"))))
        processed.append(canonicalize_row(row))
    if config.data.dedupe:
        processed = deduplicate_records(processed)
    processed = [record for record in processed if len(record["text"].split()) >= config.data.min_sentences]
    processed = limit_per_split(processed, config.data.max_samples_per_split)
    return processed


def summarize_splits(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for record in records:
        counts[str(record.get("split", "train"))] += 1
    return dict(counts)


def prepare_dataset(config: ExperimentConfig) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = load_or_build_human_records(config)
    write_jsonl(config.data.processed_path, records)
    summary = {
        "dataset_name": config.data.dataset_name,
        "processed_path": config.data.processed_path,
        "record_count": len(records),
        "split_counts": summarize_splits(records),
        "used_fallback_dataset": not Path(config.data.raw_train_path).exists(),
    }
    return records, summary

