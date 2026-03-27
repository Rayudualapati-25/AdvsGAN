from __future__ import annotations

from collections import defaultdict


def limit_per_split(records: list[dict], max_samples_per_split: int) -> list[dict]:
    if max_samples_per_split <= 0:
        return list(records)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("split", "train"))].append(record)
    limited: list[dict] = []
    for split_name in ("train", "val", "test"):
        limited.extend(grouped.get(split_name, [])[:max_samples_per_split])
    for split_name, rows in grouped.items():
        if split_name not in {"train", "val", "test"}:
            limited.extend(rows[:max_samples_per_split])
    return limited

