"""Rule-based paraphrase attack."""

from __future__ import annotations

import re
from typing import Any, Mapping

from .base import AdversaryAttack
from .schema import normalize_whitespace

PHRASE_MAP: tuple[tuple[str, str], ...] = (
    ("in order to", "to"),
    ("due to the fact that", "because"),
    ("as a result of", "because of"),
    ("in the event that", "if"),
    ("has been shown to", "appears to"),
    ("can be seen as", "may be viewed as"),
)

WORD_MAP: dict[str, str] = {
    "important": "significant",
    "significant": "substantial",
    "increase": "rise",
    "decrease": "reduction",
    "show": "demonstrate",
    "shows": "demonstrates",
    "shown": "demonstrated",
    "high": "elevated",
    "low": "reduced",
    "risk": "probability",
    "evidence": "indication",
    "patient": "individual",
    "patients": "individuals",
    "cause": "lead to",
    "causes": "leads to",
    "method": "approach",
    "methods": "approaches",
    "result": "finding",
    "results": "findings",
    "use": "utilize",
    "used": "utilized",
    "analysis": "assessment",
}

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _preserve_case(source: str, replacement: str) -> str:
    if source.isupper():
        return replacement.upper()
    if source[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _replace_phrase(text: str, source: str, target: str) -> tuple[str, bool]:
    pattern = re.compile(re.escape(source), re.IGNORECASE)

    def repl(match: re.Match[str]) -> str:
        return _preserve_case(match.group(0), target)

    updated, count = pattern.subn(repl, text)
    return updated, count > 0


def _replace_words(text: str, replacements: Mapping[str, str]) -> tuple[str, list[str]]:
    transformed = text
    changes: list[str] = []
    for source, target in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        pattern = re.compile(rf"\b{re.escape(source)}\b", re.IGNORECASE)

        def repl(match: re.Match[str]) -> str:
            return _preserve_case(match.group(0), target)

        transformed, count = pattern.subn(repl, transformed)
        if count:
            changes.append(f"{source}->{target}")
    return transformed, changes


class ParaphraseAttack(AdversaryAttack):
    family = "paraphrase"
    name = "rule_based_paraphrase"

    def apply_text(
        self,
        text: str,
        *,
        row: Mapping[str, Any] | Any | None = None,
    ) -> tuple[str, dict[str, Any]]:
        normalized = normalize_whitespace(text)
        if not normalized:
            return text, {"fallback_used": True, "fallback_reason": "empty text", "transformations": []}

        working = normalized
        transformations: list[str] = []

        if self.config.get("replace_phrases", True):
            for source, target in PHRASE_MAP:
                working, changed = _replace_phrase(working, source, target)
                if changed:
                    transformations.append(f"phrase:{source}->{target}")

        working, word_changes = _replace_words(working, self.config.get("synonym_map", WORD_MAP))
        transformations.extend(f"word:{change}" for change in word_changes)

        if self.config.get("reorder_sentences", True):
            sentences = [segment.strip() for segment in _SENTENCE_SPLIT_RE.split(working) if segment.strip()]
            if len(sentences) > 2:
                reordered = [sentences[-1], *sentences[:-1]]
                working = " ".join(reordered)
                transformations.append("sentence_reorder:last_to_front")
            elif len(sentences) == 2 and self.config.get("swap_two_sentences", False):
                working = " ".join(reversed(sentences))
                transformations.append("sentence_reorder:swap_two")

        if working == normalized:
            return text, {
                "fallback_used": True,
                "fallback_reason": "no eligible paraphrase substitutions",
                "transformations": [],
            }

        return working, {
            "fallback_used": False,
            "fallback_reason": None,
            "transformations": transformations,
            "paraphrase_strength": float(self.config.get("strength", 0.7)),
        }

