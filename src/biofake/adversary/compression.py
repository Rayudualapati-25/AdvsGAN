"""Compression and expansion attack family."""

from __future__ import annotations

import re
from typing import Any, Mapping

from .base import AdversaryAttack
from .schema import normalize_whitespace

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_PARENTHETICAL_RE = re.compile(r"\([^)]*\)")
_FILLER_PHRASES = (
    ("it is worth noting that ", ""),
    ("it should be noted that ", ""),
    ("in order to ", ""),
    ("in the context of ", ""),
    ("as shown by ", ""),
    ("the results suggest that ", ""),
)


class CompressionExpansionAttack(AdversaryAttack):
    family = "compression_expansion"
    name = "rule_based_compression_expansion"

    def apply_text(
        self,
        text: str,
        *,
        row: Mapping[str, Any] | Any | None = None,
    ) -> tuple[str, dict[str, Any]]:
        normalized = normalize_whitespace(text)
        if not normalized:
            return text, {"fallback_used": True, "fallback_reason": "empty text", "transformations": []}

        mode = str(self.config.get("mode", "compress")).lower()
        transformations: list[str] = []

        if mode == "compress":
            working = normalized
            for source, target in _FILLER_PHRASES:
                updated = working.replace(source, target).replace(source.capitalize(), target.capitalize())
                if updated != working:
                    transformations.append(f"remove_phrase:{source.strip()}")
                    working = updated
            working = _PARENTHETICAL_RE.sub("", working)
            if working != normalized:
                transformations.append("remove_parenthetical")
            sentences = [segment.strip() for segment in _SENTENCE_SPLIT_RE.split(working) if segment.strip()]
            max_sentences = int(self.config.get("max_sentences", 2))
            if len(sentences) > max_sentences:
                working = " ".join(sentences[:max_sentences])
                transformations.append(f"truncate_sentences:{max_sentences}")
            working = normalize_whitespace(working)
            if working and not working.endswith((".", "!", "?")):
                working += "."
        elif mode == "expand":
            base = normalized
            if base and base[-1] not in ".!?":
                base += "."
            template = str(
                self.config.get(
                    "expansion_sentence",
                    "This wording provides additional context without changing the underlying finding.",
                )
            ).strip()
            working = f"{base} {template}".strip()
            transformations.append("append_expansion_sentence")
        else:
            return text, {
                "fallback_used": True,
                "fallback_reason": f"unsupported mode: {mode}",
                "transformations": [],
            }

        if working == normalized:
            return text, {
                "fallback_used": True,
                "fallback_reason": "no eligible compression/expansion transformation",
                "transformations": [],
            }

        return working, {
            "fallback_used": False,
            "fallback_reason": None,
            "transformations": transformations,
            "mode": mode,
        }

