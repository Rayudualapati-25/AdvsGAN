"""Ablation helpers for attack subsets."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

from biofake.adversary.schema import coerce_bool, extract_value, row_copy


def filter_rows_by_attack_metadata(
    rows: Iterable[Mapping[str, Any] | Any],
    *,
    families: Sequence[str] | None = None,
    attack_names: Sequence[str] | None = None,
    fallback_used: bool | None = None,
) -> list[dict[str, Any]]:
    family_set = {str(item) for item in families} if families is not None else None
    name_set = {str(item) for item in attack_names} if attack_names is not None else None
    output: list[dict[str, Any]] = []
    for row in rows:
        copied = row_copy(row)
        family = str(extract_value(copied, ("attack_family",), default=""))
        name = str(extract_value(copied, ("attack_name",), default=""))
        row_fallback_used = coerce_bool(extract_value(copied, ("fallback_used",), default=False))
        if family_set is not None and family not in family_set:
            continue
        if name_set is not None and name not in name_set:
            continue
        if fallback_used is not None and row_fallback_used is not fallback_used:
            continue
        output.append(copied)
    return output


def leave_one_family_out(
    rows: Iterable[Mapping[str, Any] | Any],
    *,
    family_field: str = "attack_family",
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    copied_rows = [row_copy(row) for row in rows]
    for row in copied_rows:
        family = str(extract_value(row, (family_field,), default="unknown"))
        grouped[family].append(row)

    scenarios: dict[str, list[dict[str, Any]]] = {}
    families = set(grouped)
    for family in families:
        scenarios[f"drop_{family}"] = [
            row for row in copied_rows if str(extract_value(row, (family_field,), default="unknown")) != family
        ]
    scenarios["keep_all"] = copied_rows
    return scenarios


def build_ablation_scenarios(
    rows: Iterable[Mapping[str, Any] | Any],
    *,
    family_field: str = "attack_family",
) -> dict[str, list[dict[str, Any]]]:
    rows = [row_copy(row) for row in rows]
    scenarios = leave_one_family_out(rows, family_field=family_field)
    scenarios["only_comparison_rows"] = [
        row for row in rows if str(extract_value(row, (family_field,), default="")) in {"paraphrase", "compression_expansion", "style_transfer"}
    ]
    return scenarios
