from __future__ import annotations

from typing import Any, Mapping


def simulate_seqgan_text(text: str) -> str:
    tokens = text.split()
    if len(tokens) < 12:
        return text
    short = tokens[: min(20, len(tokens))]
    reordered = short[::2] + short[1::2]
    output = " ".join(reordered)
    if output and output[-1] not in ".!?":
        output += "."
    return output


def generate_seqgan_legacy_rows(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    synthetic_rows: list[dict[str, Any]] = []
    for row in rows:
        synthetic_rows.append(
            {
                "id": f"{row['id']}_seqgan",
                "split": row["split"],
                "label": "synthetic",
                "source": row.get("source", "pubmed_rct"),
                "generator": "seqgan_legacy",
                "attack": None,
                "parent_id": row["id"],
                "text": simulate_seqgan_text(str(row.get("text", ""))),
                "meta": {
                    "prompt_style": "seqgan_legacy",
                    "source_meta": row.get("meta", {}),
                },
            }
        )
    return synthetic_rows

