from __future__ import annotations

from typing import Any

from biofake.generation.local_llm import build_local_llm
from biofake.generation.prompts import build_generation_prompt
from biofake.generation.provenance import generation_metadata
from biofake.generation.seqgan_legacy import generate_seqgan_legacy_rows
from biofake.io import write_jsonl
from biofake.schemas import ExperimentConfig


def generate_synthetic_rows(
    rows: list[dict[str, Any]],
    config: ExperimentConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if config.generation.model_id == "seqgan_legacy":
        generated = generate_seqgan_legacy_rows(rows)
        write_jsonl(config.generation.output_path, generated)
        return generated, {"backend": "seqgan_legacy", "count": len(generated)}

    llm = build_local_llm(config.generation.model_dump())
    generated: list[dict[str, Any]] = []
    for row in rows:
        if row["split"] not in config.generation.enabled_splits:
            continue
        prompt = build_generation_prompt(row, style=config.generation.system_prompt_style)
        result = llm.generate(
            prompt,
            max_tokens=config.generation.max_new_tokens,
            temperature=config.generation.temperature,
            seed=config.seed,
            metadata={"source_id": row["id"]},
        )
        provenance = generation_metadata(
            row,
            backend_name=result.provenance.backend,
            prompt=prompt,
            prompt_style=config.generation.system_prompt_style,
            fallback_reason=result.provenance.fallback_reason,
            extra=result.to_dict()["provenance"],
        )
        generated.append(
            {
                "id": f"{row['id']}_synthetic",
                "split": row["split"],
                "label": "synthetic",
                "source": row.get("source", "pubmed_rct"),
                "generator": result.provenance.model_name or result.provenance.backend,
                "attack": None,
                "parent_id": row["id"],
                "text": result.text,
                "meta": provenance,
            }
        )
    write_jsonl(config.generation.output_path, generated)
    return generated, {"backend": llm.backend.__class__.__name__, "count": len(generated)}

