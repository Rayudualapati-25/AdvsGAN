"""Shared row and text helpers.

The helpers here are intentionally schema-flexible so they can work with a
shared processed-row format from another branch without depending on a concrete
IO layer.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import re
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

TEXT_FIELD_CANDIDATES: tuple[str, ...] = (
    "text",
    "content",
    "document",
    "prompt",
    "input_text",
)
ID_FIELD_CANDIDATES: tuple[str, ...] = ("id", "row_id", "sample_id", "doc_id")
LABEL_FIELD_CANDIDATES: tuple[str, ...] = ("label", "target", "class", "y")

_WHITESPACE_RE = re.compile(r"\s+")


def row_copy(row: Any) -> dict[str, Any]:
    if isinstance(row, Mapping):
        return dict(row)
    if is_dataclass(row):
        return asdict(row)
    if hasattr(row, "__dict__"):
        return dict(vars(row))
    raise TypeError(f"Unsupported row type: {type(row)!r}")


def resolve_field_name(
    row: Mapping[str, Any] | Any,
    candidates: Sequence[str],
    default: str | None = None,
) -> str:
    if isinstance(row, Mapping):
        for candidate in candidates:
            if candidate in row:
                return candidate
    else:
        for candidate in candidates:
            if hasattr(row, candidate):
                return candidate
    if default is not None:
        return default
    raise KeyError(f"None of the candidate fields are present: {candidates!r}")


def extract_value(
    row: Mapping[str, Any] | Any,
    candidates: Sequence[str],
    default: Any = None,
) -> Any:
    try:
        field = resolve_field_name(row, candidates)
    except KeyError:
        return default
    if isinstance(row, Mapping):
        return row.get(field, default)
    return getattr(row, field, default)


def extract_text(row: Mapping[str, Any] | Any, default: str = "") -> str:
    value = extract_value(row, TEXT_FIELD_CANDIDATES, default=default)
    if value is None:
        return default
    return str(value)


def normalize_whitespace(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "f", "no", "n", "off", ""}:
            return False
    return bool(value)


def count_token_differences(original: str, updated: str) -> int:
    original_tokens = original.split()
    updated_tokens = updated.split()
    shared = min(len(original_tokens), len(updated_tokens))
    delta = sum(1 for index in range(shared) if original_tokens[index] != updated_tokens[index])
    delta += abs(len(original_tokens) - len(updated_tokens))
    return delta


def count_character_differences(original: str, updated: str) -> int:
    return sum(1 for left, right in zip(original, updated) if left != right) + abs(
        len(original) - len(updated)
    )


def attach_attack_metadata(
    row: Mapping[str, Any] | Any,
    updated_text: str,
    metadata: Mapping[str, Any],
    *,
    text_field: str | None = None,
    metadata_field: str = "attack_metadata",
    original_text_field: str = "original_text",
) -> dict[str, Any]:
    """Return a copied row with attack metadata attached.

    The function updates the discovered text field in place and also writes a
    stable ``adversarial_text`` alias for downstream consumers that expect it.
    """

    copied = row_copy(row)
    resolved_text_field = text_field or resolve_field_name(copied, TEXT_FIELD_CANDIDATES, default="text")
    original_text = copied.get(resolved_text_field, "")
    copied[resolved_text_field] = updated_text
    copied["adversarial_text"] = updated_text
    copied.setdefault(original_text_field, original_text)
    copied[metadata_field] = dict(metadata)
    copied["attack_name"] = metadata.get("attack_name")
    copied["attack_family"] = metadata.get("attack_family")
    copied["attack_applied"] = not metadata.get("fallback_used", False)
    return copied
