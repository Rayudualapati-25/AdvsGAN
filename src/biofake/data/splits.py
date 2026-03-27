from __future__ import annotations


def normalize_split_name(value: str) -> str:
    lowered = value.strip().lower()
    if lowered in {"validation", "valid", "dev"}:
        return "val"
    if lowered in {"testing"}:
        return "test"
    return lowered

