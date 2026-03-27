"""Base classes for adversarial attacks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .schema import (
    attach_attack_metadata,
    count_character_differences,
    count_token_differences,
    extract_text,
    normalize_whitespace,
    row_copy,
)


@dataclass(slots=True)
class AttackOutcome:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class AdversaryAttack(ABC):
    """Deterministic attack base class."""

    family: str = "generic"
    name: str = "generic_attack"

    def __init__(self, **config: Any) -> None:
        self.config = dict(config)

    def __call__(self, row: Mapping[str, Any] | Any) -> dict[str, Any]:
        return self.attack_row(row)

    def attack_row(self, row: Mapping[str, Any] | Any) -> dict[str, Any]:
        source_text = extract_text(row, default="")
        updated_text, metadata = self.apply_text(source_text, row=row)
        source_text = normalize_whitespace(source_text)
        updated_text = normalize_whitespace(updated_text)
        metadata = dict(metadata)
        metadata.setdefault("attack_name", self.name)
        metadata.setdefault("attack_family", self.family)
        metadata.setdefault("input_text", source_text)
        metadata.setdefault("output_text", updated_text)
        metadata.setdefault("input_char_count", len(source_text))
        metadata.setdefault("output_char_count", len(updated_text))
        metadata.setdefault("changed_char_count", count_character_differences(source_text, updated_text))
        metadata.setdefault("changed_token_count", count_token_differences(source_text, updated_text))
        metadata.setdefault("fallback_used", source_text == updated_text)
        metadata.setdefault(
            "fallback_reason",
            "no-op transformation" if source_text == updated_text else None,
        )
        metadata.setdefault("parameters", dict(self.config))
        return attach_attack_metadata(row, updated_text, metadata)

    def attack_rows(self, rows: Iterable[Mapping[str, Any] | Any]) -> list[dict[str, Any]]:
        return [self.attack_row(row) for row in rows]

    @abstractmethod
    def apply_text(
        self,
        text: str,
        *,
        row: Mapping[str, Any] | Any | None = None,
    ) -> tuple[str, dict[str, Any]]:
        raise NotImplementedError

