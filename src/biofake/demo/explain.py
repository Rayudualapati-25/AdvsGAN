from __future__ import annotations

import re


def explain_text(text: str) -> list[str]:
    reasons: list[str] = []
    lowered = text.lower()
    if "background:" in lowered and "methods:" in lowered and "results:" in lowered:
        reasons.append("Structured abstract markers are present.")
    if len(re.findall(r"\b\d+\b", text)) >= 2:
        reasons.append("The text contains several numeric clinical cues.")
    if len(text.split()) < 45:
        reasons.append("The sample is short, which can make classifier confidence unstable.")
    if any(phrase in lowered for phrase in ["this wording", "deterministic", "template-driven"]):
        reasons.append("The wording resembles synthetic fallback or style-transfer phrasing.")
    if not reasons:
        reasons.append("No single dominant cue was detected; the model is relying on mixed lexical signals.")
    return reasons

