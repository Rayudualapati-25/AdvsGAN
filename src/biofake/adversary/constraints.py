from __future__ import annotations

from typing import Mapping


def passes_basic_constraints(original_row: Mapping[str, object], attacked_row: Mapping[str, object]) -> bool:
    original_text = str(original_row.get("text", "")).strip()
    attacked_text = str(attacked_row.get("adversarial_text") or attacked_row.get("text", "")).strip()
    if not original_text or not attacked_text:
        return False
    if len(attacked_text.split()) < max(8, len(original_text.split()) // 3):
        return False
    return True

