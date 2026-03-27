from __future__ import annotations

from typing import Any, Mapping


def generation_metadata(
    source_row: Mapping[str, Any],
    *,
    backend_name: str,
    prompt: str,
    prompt_style: str,
    fallback_reason: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "source_id": source_row.get("id"),
        "prompt_style": prompt_style,
        "backend": backend_name,
        "prompt": prompt,
        "fallback_reason": fallback_reason,
        **dict(extra or {}),
    }

