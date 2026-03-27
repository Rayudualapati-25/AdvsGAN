from __future__ import annotations

from typing import Any, Mapping


def build_generation_prompt(row: Mapping[str, Any], style: str = "biomedical_abstract") -> str:
    meta = row.get("meta", {}) or {}
    title = meta.get("title") or "Untitled biomedical study"
    text = str(row.get("text", "")).strip()
    sections = meta.get("sections") or []
    section_names = ", ".join(section[0] for section in sections) if sections else "background, methods, results, conclusion"
    if style == "seqgan_legacy":
        return (
            "Generate one short biomedical sentence that sounds plausible but machine-produced.\n"
            f"Seed text: {text}\n"
            "Return only one sentence."
        )
    return (
        "You are generating a biomedical research abstract for robustness testing.\n"
        f"Title: {title}\n"
        f"Section skeleton: {section_names}\n"
        "Write a concise abstract that preserves the same topic but does not copy the source text verbatim.\n"
        f"Source abstract:\n{text}"
    )


def build_attack_prompt(row: Mapping[str, Any], attack_name: str, strength: str = "medium") -> str:
    return (
        f"Rewrite the following biomedical abstract to evade AI-text detection.\n"
        f"Attack family: {attack_name}\n"
        f"Strength: {strength}\n"
        "Preserve the scientific meaning and topic.\n"
        f"Abstract:\n{row.get('text', '')}"
    )

