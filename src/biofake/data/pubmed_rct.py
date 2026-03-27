from __future__ import annotations

import csv
import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


PROCESSED_ROW_SCHEMA_VERSION = "1.0"
AUTO_SPLIT = "auto"

_CANONICAL_SECTION_ORDER = (
    "background",
    "objective",
    "methods",
    "results",
    "conclusion",
)

_SECTION_ALIASES = {
    "background": "background",
    "objective": "objective",
    "aim": "objective",
    "purpose": "objective",
    "methods": "methods",
    "method": "methods",
    "materials and methods": "methods",
    "participants": "methods",
    "results": "results",
    "findings": "results",
    "conclusion": "conclusion",
    "conclusions": "conclusion",
    "interpretation": "conclusion",
}

_SECTION_MARKER_RE = re.compile(
    r"(?i)\b(background|objective|aim|purpose|methods|method|"
    r"materials and methods|participants|results|findings|conclusion|"
    r"conclusions|interpretation)\s*[:\-]\s*"
)


@dataclass(frozen=True)
class ProcessedRow:
    """Normalized row compatible with downstream text processing."""

    row_id: str
    source_id: str
    pmid: str | None
    split: str
    label: str | None
    title: str
    abstract: str
    text: str
    sections: tuple[tuple[str, str], ...] = ()
    source: str = "pubmed_rct"
    schema_version: str = PROCESSED_ROW_SCHEMA_VERSION
    provenance: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "row_id": self.row_id,
            "source_id": self.source_id,
            "pmid": self.pmid,
            "split": self.split,
            "label": self.label,
            "title": self.title,
            "abstract": self.abstract,
            "text": self.text,
            "sections_json": json.dumps(list(self.sections), ensure_ascii=True, sort_keys=True),
            "provenance_json": json.dumps(dict(self.provenance), ensure_ascii=True, sort_keys=True),
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "ProcessedRow":
        sections = _coerce_sections(record)
        provenance = _coerce_provenance(record)
        return cls(
            row_id=_coerce_identifier(record, ("row_id", "id", "doc_id", "uid", "pmid")),
            source_id=_coerce_identifier(record, ("source_id", "pmid", "id", "doc_id", "uid")),
            pmid=_coerce_optional_str(record.get("pmid") or record.get("PMID") or record.get("doc_id")),
            split=str(record.get("split") or record.get("subset") or AUTO_SPLIT),
            label=_coerce_optional_str(record.get("label") or record.get("target") or record.get("y")),
            title=_coerce_text(record.get("title") or record.get("article_title") or record.get("headline")),
            abstract=_coerce_text(record.get("abstract") or record.get("abstract_text") or record.get("summary")),
            text=_coerce_text(record.get("text") or record.get("document_text") or record.get("body")),
            sections=sections,
            source=_coerce_optional_str(record.get("source") or record.get("dataset") or "pubmed_rct") or "pubmed_rct",
            schema_version=str(record.get("schema_version") or PROCESSED_ROW_SCHEMA_VERSION),
            provenance=provenance,
        )


def clean_pubmed_rct_text(text: str) -> str:
    """Normalize whitespace and punctuation while preserving the abstract content."""

    if not text:
        return ""
    normalized = unicodedata.normalize("NFKC", str(text))
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\u00a0", " ")
    normalized = re.sub(r"[ \t\f\v]+", " ", normalized)
    normalized = re.sub(r"\s*\n\s*", "\n", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    normalized = re.sub(r"(?<=\w)-\s+(?=\w)", "-", normalized)
    normalized = re.sub(r"\s+([,.;:?!])", r"\1", normalized)
    normalized = normalized.strip()
    return normalized


def parse_pubmed_rct_abstract(raw_abstract: str) -> tuple[tuple[str, str], ...]:
    """Reconstruct sectioned RCT-style abstracts into a canonical section order."""

    if not raw_abstract:
        return ()

    text = clean_pubmed_rct_text(raw_abstract)
    if not text:
        return ()

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if not lines:
        return ()

    sections: list[tuple[str, list[str]]] = []
    current_header = "abstract"
    current_body: list[str] = []

    def flush() -> None:
        nonlocal current_header, current_body
        if current_body:
            sections.append((current_header, current_body.copy()))
        current_body = []

    for line in lines:
        markers = list(_SECTION_MARKER_RE.finditer(line))
        if not markers:
            current_body.append(line)
            continue

        if len(markers) == 1 and markers[0].start() == 0:
            flush()
            current_header = _canonical_section_name(markers[0].group(1))
            body = line[markers[0].end() :].strip(" :-")
            if body:
                current_body.append(body)
            continue

        if len(markers) > 1:
            flush()
            for idx, marker in enumerate(markers):
                header = _canonical_section_name(marker.group(1))
                start = marker.end()
                end = markers[idx + 1].start() if idx + 1 < len(markers) else len(line)
                body = line[start:end].strip(" :-")
                if body:
                    sections.append((header, [body]))
            current_header = "abstract"
            current_body = []
            continue

        current_body.append(line)

    flush()

    collapsed: dict[str, list[str]] = {}
    fallback: list[str] = []
    for header, body_lines in sections:
        body = clean_pubmed_rct_text(" ".join(body_lines))
        if not body:
            continue
        if header == "abstract":
            fallback.append(body)
            continue
        collapsed.setdefault(header, []).append(body)

    ordered: list[tuple[str, str]] = []
    for header in _CANONICAL_SECTION_ORDER:
        if header in collapsed:
            ordered.append((header, clean_pubmed_rct_text(" ".join(collapsed[header]))))
    for header in sorted(k for k in collapsed if k not in _CANONICAL_SECTION_ORDER):
        ordered.append((header, clean_pubmed_rct_text(" ".join(collapsed[header]))))
    if fallback and not ordered:
        ordered.append(("abstract", clean_pubmed_rct_text(" ".join(fallback))))
    return tuple(ordered)


def deterministic_split_for_key(
    key: str,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> str:
    """Assign a stable split using a hash bucket."""

    ratios = [max(0.0, float(train_ratio)), max(0.0, float(validation_ratio)), max(0.0, float(test_ratio))]
    total = sum(ratios)
    if total <= 0:
        raise ValueError("At least one split ratio must be positive")
    train_ratio, validation_ratio, test_ratio = [ratio / total for ratio in ratios]
    digest = hashlib.sha256(str(key).encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") / float(1 << 64)
    if bucket < train_ratio:
        return "train"
    if bucket < train_ratio + validation_ratio:
        return "validation"
    return "test"


def process_pubmed_rct_record(
    record: Mapping[str, Any],
    *,
    split: str = AUTO_SPLIT,
    source: str = "pubmed_rct",
) -> ProcessedRow:
    """Convert a raw PubMed RCT row into a normalized processed row."""

    title = _coerce_text(record.get("title") or record.get("article_title") or "")
    raw_abstract = _coerce_text(record.get("abstract") or record.get("abstract_text") or record.get("summary") or "")
    sections = parse_pubmed_rct_abstract(raw_abstract)
    if sections:
        abstract = "\n\n".join(f"{header.upper()}: {body}" for header, body in sections)
    else:
        abstract = clean_pubmed_rct_text(raw_abstract)

    title = clean_pubmed_rct_text(title)
    abstract = clean_pubmed_rct_text(abstract)
    text = clean_pubmed_rct_text("\n\n".join(part for part in (title, abstract) if part))

    row_id = _coerce_identifier(record, ("row_id", "id", "doc_id", "uid", "pmid"))
    source_id = _coerce_identifier(record, ("source_id", "pmid", "id", "doc_id", "uid"))
    pmid = _coerce_optional_str(record.get("pmid") or record.get("PMID") or record.get("doc_id"))
    split_value = split if split != AUTO_SPLIT else deterministic_split_for_key(source_id)
    label = _coerce_optional_str(record.get("label") or record.get("target") or record.get("y"))

    provenance = {
        "source": source,
        "raw_keys": sorted(str(key) for key in record.keys()),
        "section_count": len(sections),
    }
    return ProcessedRow(
        row_id=row_id,
        source_id=source_id,
        pmid=pmid,
        split=split_value,
        label=label,
        title=title,
        abstract=abstract,
        text=text,
        sections=sections,
        source=source,
        provenance=provenance,
    )


def split_processed_rows(
    records: Iterable[Mapping[str, Any] | ProcessedRow],
    *,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> dict[str, list[ProcessedRow]]:
    """Split a sequence of processed rows or raw records deterministically."""

    splits = {"train": [], "validation": [], "test": []}
    for record in records:
        row = record if isinstance(record, ProcessedRow) else process_pubmed_rct_record(record, split=AUTO_SPLIT)
        split_name = deterministic_split_for_key(
            row.source_id,
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
            test_ratio=test_ratio,
        )
        splits[split_name].append(
            ProcessedRow(
                row_id=row.row_id,
                source_id=row.source_id,
                pmid=row.pmid,
                split=split_name,
                label=row.label,
                title=row.title,
                abstract=row.abstract,
                text=row.text,
                sections=row.sections,
                source=row.source,
                schema_version=row.schema_version,
                provenance=dict(row.provenance, split=split_name),
            )
        )
    return splits


def iter_processed_rows(records: Iterable[Mapping[str, Any] | ProcessedRow]) -> Iterator[ProcessedRow]:
    for record in records:
        yield record if isinstance(record, ProcessedRow) else process_pubmed_rct_record(record)


def load_raw_pubmed_rct_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_processed_rows_csv(path: str | Path) -> Iterator[ProcessedRow]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield ProcessedRow.from_dict(row)


def dump_processed_rows_csv(rows: Iterable[ProcessedRow], path: str | Path) -> None:
    rows = list(rows)
    fieldnames = list(rows[0].to_dict().keys()) if rows else list(ProcessedRow.from_dict({}).to_dict().keys())
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())


def _canonical_section_name(name: str) -> str:
    normalized = clean_pubmed_rct_text(name).lower().strip(":.- ")
    return _SECTION_ALIASES.get(normalized, normalized)


def _coerce_identifier(record: Mapping[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = record.get(key)
        if value is None:
            continue
        text = clean_pubmed_rct_text(str(value))
        if text:
            return text
    return "unknown"


def _coerce_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = clean_pubmed_rct_text(str(value))
    return text or None


def _coerce_text(value: Any) -> str:
    return clean_pubmed_rct_text(str(value)) if value is not None else ""


def _coerce_provenance(record: Mapping[str, Any]) -> dict[str, Any]:
    value = record.get("provenance") or record.get("metadata") or record.get("info")
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return {}
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {"value": value}
        return dict(parsed) if isinstance(parsed, Mapping) else {"value": parsed}
    return {"value": value}


def _coerce_sections(record: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    value = record.get("sections")
    if value is None and record.get("sections_json"):
        try:
            value = json.loads(str(record["sections_json"]))
        except json.JSONDecodeError:
            value = None
    if value is None:
        headers = record.get("section_headers")
        texts = record.get("section_texts")
        if headers is not None and texts is not None:
            pairs = []
            for header, body in zip(headers, texts):
                pairs.append((clean_pubmed_rct_text(str(header)).lower(), clean_pubmed_rct_text(str(body))))
            return tuple((header, body) for header, body in pairs if header and body)
        return ()
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return ()
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return (("abstract", clean_pubmed_rct_text(value)),)
        value = parsed
    if isinstance(value, Mapping):
        return tuple((clean_pubmed_rct_text(str(k)).lower(), clean_pubmed_rct_text(str(v))) for k, v in value.items())
    if isinstance(value, Sequence):
        pairs: list[tuple[str, str]] = []
        for item in value:
            if isinstance(item, Mapping):
                header = item.get("header") or item.get("section") or item.get("name")
                body = item.get("text") or item.get("body") or item.get("value")
                if header is None or body is None:
                    continue
                pairs.append((clean_pubmed_rct_text(str(header)).lower(), clean_pubmed_rct_text(str(body))))
            elif isinstance(item, Sequence) and len(item) >= 2:
                pairs.append((clean_pubmed_rct_text(str(item[0])).lower(), clean_pubmed_rct_text(str(item[1]))))
        return tuple((header, body) for header, body in pairs if header and body)
    return ()
