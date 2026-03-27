"""Deterministic style transfer attack."""

from __future__ import annotations

import re
from typing import Any, Mapping

from .base import AdversaryAttack
from .schema import normalize_whitespace

STYLE_MAPS: dict[str, tuple[tuple[str, str], ...]] = {
    "clinical": (
        ("shows", "is consistent with"),
        ("show", "indicates"),
        ("causes", "is associated with"),
        ("cause", "is associated with"),
        ("improves", "is associated with improvement in"),
        ("important", "clinically relevant"),
        ("bad", "adverse"),
        ("good", "favorable"),
        ("we found", "the analysis indicates"),
        ("i think", "the findings suggest"),
    ),
    "plain": (
        ("utilize", "use"),
        ("approximately", "about"),
        ("subsequent", "next"),
        ("prior to", "before"),
        ("commence", "start"),
        ("terminate", "end"),
    ),
    "formal": (
        ("can't", "cannot"),
        ("don't", "do not"),
        ("it's", "it is"),
        ("we found", "the analysis found"),
        ("very", "substantially"),
        ("really", "materially"),
    ),
}


def _preserve_case(source: str, replacement: str) -> str:
    if source.isupper():
        return replacement.upper()
    if source[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _apply_phrase_map(text: str, replacements: tuple[tuple[str, str], ...]) -> tuple[str, list[str]]:
    working = text
    transformations: list[str] = []
    for source, target in replacements:
        pattern = re.compile(re.escape(source), re.IGNORECASE)
        updated, count = pattern.subn(lambda match: _preserve_case(match.group(0), target), working)
        if count:
            transformations.append(f"{source}->{target}")
            working = updated
    return working, transformations


class StyleTransferAttack(AdversaryAttack):
    family = "style_transfer"
    name = "rule_based_style_transfer"

    def apply_text(
        self,
        text: str,
        *,
        row: Mapping[str, Any] | Any | None = None,
    ) -> tuple[str, dict[str, Any]]:
        normalized = normalize_whitespace(text)
        if not normalized:
            return text, {"fallback_used": True, "fallback_reason": "empty text", "transformations": []}

        style = str(self.config.get("style", "clinical")).lower()
        replacements = STYLE_MAPS.get(style)
        if replacements is None:
            return text, {
                "fallback_used": True,
                "fallback_reason": f"unsupported style: {style}",
                "transformations": [],
            }

        working, transformations = _apply_phrase_map(normalized, replacements)
        if self.config.get("add_closing_clause", True):
            if style == "clinical":
                suffix = "This formulation is clinically framed and intentionally more cautious."
            elif style == "plain":
                suffix = "This keeps the wording direct and easy to follow."
            else:
                suffix = "This wording adopts a more formal register."
            if not working.endswith((".", "!", "?")):
                working += "."
            working = f"{working} {suffix}"
            transformations.append("append_style_closing_clause")

        working = normalize_whitespace(working)
        if working == normalized:
            return text, {
                "fallback_used": True,
                "fallback_reason": "no eligible style transformation",
                "transformations": [],
            }

        return working, {
            "fallback_used": False,
            "fallback_reason": None,
            "transformations": transformations,
            "style": style,
        }

